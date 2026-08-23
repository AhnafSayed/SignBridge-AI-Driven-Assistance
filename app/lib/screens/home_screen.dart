import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show HapticFeedback;
import 'package:flutter_native_splash/flutter_native_splash.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';

import '../app_theme.dart';
import '../services/asl_vocabulary.dart';
import '../services/bdsl_glossary.dart';
import '../services/control_gesture.dart';
import '../services/direct_static_classifier.dart';
import '../services/phrase_store.dart';
import '../services/sign_category.dart';
import '../services/sign_classifier.dart';
import '../services/tflite_sign_classifier.dart';
import '../services/ui_strings.dart';
import '../services/word_completer.dart';
import '../state/app_state.dart';
import '../widgets/brand_title.dart';
import '../widgets/landmark_painter.dart';
import 'about_screen.dart';
import 'phrases_sheet.dart';
import 'sos_sheet.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => HomeScreenState();
}

enum _Stage { requestingPermission, permissionDenied, initializing, ready, noCamera, cameraError }

/// Public (not `_HomeScreenState`) so [RootShell] can hold a [GlobalKey] to
/// it and call [pause]/[resume] directly, instead of destroying and
/// recreating this whole widget on every tab switch. Camera + MediaPipe
/// plugin teardown is async but [State.dispose] can't be awaited, so
/// swapping this widget out and a new one in quickly (e.g. flicking
/// between tabs) could start a new camera/plugin init before the old one
/// finished tearing down -- a real native-level race that could crash.
/// Keeping one instance alive for the app's lifetime and just pausing its
/// camera stream removes that race entirely.
class HomeScreenState extends State<HomeScreen>
    with WidgetsBindingObserver, AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  // How long a sign must hold steady before it's added to the sentence, and
  // how long a control handshape (fist / thumbs-down) must hold before it
  // fires -- long enough that it isn't triggered by a hand just passing
  // through that shape on its way to something else.
  static const _wordHoldDuration = Duration(milliseconds: 700);
  static const _backspaceHoldDuration = Duration(milliseconds: 900);
  static const _speakHoldDuration = Duration(milliseconds: 900);
  static const _clearHoldDuration = Duration(milliseconds: 1400);
  // How long with no hand in frame before the finished sentence is spoken.
  static const _silenceBeforeSpeak = Duration(milliseconds: 1800);

  CameraController? _controller;
  HandLandmarkerPlugin? _plugin;
  StreamSubscription<List<Hand>>? _landmarkSubscription;
  CameraLensDirection _lensDirection = CameraLensDirection.front;
  bool _loggedFrameError = false;

  // Both languages load their real trained model asynchronously; the
  // finger-count prototype is the fallback shown until (or unless) a
  // model finishes loading -- see _classifierFor.
  final SignClassifier _prototype = FingerCountClassifier();
  TfliteSignClassifier? _aslClassifier;
  TfliteSignClassifier? _bdslClassifier;
  // ASL digits/letters only -- see direct_static_classifier.dart's own
  // comment for why this exists as a second classifier rather than a
  // replacement, and why BdSL's digits/letters deliberately stay on
  // _bdslClassifier instead of getting an equivalent (measured worse on
  // held-out signers, unlike ASL's static sources which have none to
  // hold out).
  DirectStaticClassifier? _aslStaticClassifier;
  final ControlGestureDetector _controlDetector = ControlGestureDetector();
  final FlutterTts _tts = FlutterTts();

  List<Hand> _hands = const [];
  ClassificationResult? _liveResult;
  ControlGesture _activeControl = ControlGesture.none;
  double _stabilityProgress = 0;
  _Stage _stage = _Stage.requestingPermission;
  String? _cameraErrorMessage;

  String? _wordStableLabel;
  DateTime? _wordStableSince;
  String? _lastCommittedWord;
  ControlGesture _controlStableGesture = ControlGesture.none;
  DateTime? _controlStableSince;
  ControlGesture _controlFiredFor = ControlGesture.none;
  final ControlArbiter _controlArbiter = ControlArbiter();
  DateTime? _lastHandSeenAt;
  bool _autoSpokenForSentence = false;

  // Only reapplied when either actually changes (not every frame) --
  // setAllowedCategories rebuilds a mask over the full label list, cheap
  // but pointless to redo 30x/sec for no reason.
  SignClassifier? _categoryAppliedTo;
  SignCategory? _lastAppliedCategory;

  // Fingerspelling: letters committed in Letters mode accumulate here
  // instead of becoming separate auto-spaced sentence entries, so
  // spelling "C" "A" "T" builds "CAT" instead of "C A T" (see
  // _commitWord). Flushed into the sentence as one word on mode switch,
  // Clear/Speak, or accepting a suggestion.
  String _spellBuffer = '';
  List<String> _suggestions = const [];
  WordCompleter? _englishWordCompleter;
  WordCompleter? _banglaWordCompleter;

  WordCompleter? _completerFor(SignLanguage language) =>
      language == SignLanguage.bdsl ? _banglaWordCompleter : _englishWordCompleter;

  SignClassifier _classifierFor(SignLanguage language, SignCategory category) {
    if (language == SignLanguage.bdsl) {
      return _bdslClassifier ?? _prototype;
    }
    // ASL digits/letters route to the direct single-frame classifier
    // (measured win over forcing statics through the sequence contract --
    // see direct_static_classifier.dart); ASL words stay on the sequence
    // model, which is what dynamic signs actually need. Falls back to the
    // sequence classifier, not the finger-count prototype, if the static
    // model hasn't finished loading yet -- still real recognition, just
    // not yet the faster path.
    if (category != SignCategory.word) {
      return _aslStaticClassifier ?? _aslClassifier ?? _prototype;
    }
    return _aslClassifier ?? _prototype;
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Splash stays up until camera + MediaPipe plugin + both trained models
    // are *all* ready, not just whichever finishes first -- so the app opens
    // immediately usable instead of showing a spinner or a "prototype
    // engine" badge that flips to the real model a second later. The word
    // dictionary is tiny (10k short lines) and loads near-instantly, so it
    // isn't worth gating the splash on -- fire and forget.
    Future.wait([_bootstrap(), _loadAslModel(), _loadAslStaticModel(), _loadBdslModel()])
        .whenComplete(FlutterNativeSplash.remove);
    // Neither load had error handling before -- a failure (bad asset path,
    // malformed encoding, anything) would silently leave the completer
    // null forever, with suggestions just never appearing and no error
    // anywhere to explain why. At minimum this makes that failure visible.
    WordCompleter.loadEnglish().then((c) {
      if (mounted) setState(() => _englishWordCompleter = c);
    }).catchError((e) {
      debugPrint('Failed to load English word list: $e');
    });
    WordCompleter.loadBangla().then((c) {
      if (mounted) setState(() => _banglaWordCompleter = c);
    }).catchError((e) {
      debugPrint('Failed to load Bangla word list: $e');
    });
    // Without this, a TTS engine failure (missing voice, engine crash) is
    // silent -- the "Speak" button just does nothing with no indication
    // why, indistinguishable from the recognition pipeline itself being
    // dead. At minimum this makes the failure visible in logs instead of
    // vanishing.
    _tts.setErrorHandler((msg) => debugPrint('TTS error: $msg'));
  }

  Future<void> _loadAslModel() async {
    try {
      final classifier = await TfliteSignClassifier.create(
        modelAsset: 'assets/models/asl/model.tflite',
        labelsAsset: 'assets/models/asl/labels.json',
        displayName: 'ASL 250-word CNN (trained)',
      );
      if (mounted) setState(() => _aslClassifier = classifier);
    } catch (e) {
      debugPrint('Failed to load ASL model: $e');
    }
  }

  Future<void> _loadAslStaticModel() async {
    try {
      final classifier = await DirectStaticClassifier.create(
        modelAsset: 'assets/models/asl_static/model.tflite',
        labelsAsset: 'assets/models/asl_static/labels.json',
        displayName: 'ASL digits/letters direct MLP (trained)',
      );
      if (mounted) setState(() => _aslStaticClassifier = classifier);
    } catch (e) {
      debugPrint('Failed to load ASL static model: $e');
    }
  }

  Future<void> _loadBdslModel() async {
    try {
      final classifier = await TfliteSignClassifier.create(
        modelAsset: 'assets/models/bdsl/model.tflite',
        labelsAsset: 'assets/models/bdsl/labels.json',
        displayName: 'BdSL 60-word CNN (trained)',
      );
      if (mounted) setState(() => _bdslClassifier = classifier);
    } catch (e) {
      debugPrint('Failed to load BdSL model: $e');
    }
  }

  Future<void> _bootstrap() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _stage = _Stage.permissionDenied);
      return;
    }
    setState(() => _stage = _Stage.initializing);

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      setState(() => _stage = _Stage.noCamera);
      return;
    }

    final camera = cameras.firstWhere(
      (cam) => cam.lensDirection == _lensDirection,
      orElse: () => cameras.first,
    );

    CameraController? controller;
    try {
      controller = CameraController(camera, ResolutionPreset.medium, enableAudio: false);
      // Fire camera hardware init first (native work proceeds on its own
      // thread) *before* the blocking plugin setup below, so the two
      // overlap instead of running fully back-to-back.
      final initFuture = controller.initialize();

      // HandLandmarkerPlugin.create() is synchronous under the hood (it
      // loads the MediaPipe model and sets up the GPU delegate via a
      // blocking JNI call) -- there's no async version to await. Creating
      // it once and reusing it across camera switches means this blocking
      // cost is paid only on first launch (while the splash screen is
      // still covering it), never again per switch.
      //
      // numHands: 2 -- the trained models now use both hands (see
      // ml/README.md's two-hand section); requesting only 1 here would
      // silently starve them of a real signal for every two-handed sign.
      // (This also happens to be the plugin's own default.)
      //
      // delegate: cpu, not gpu -- kept from the last round: GPU delegate
      // init can fail without a catchable exception on hardware lacking a
      // working GPU compute path, and CPU is slower but supported
      // everywhere. This model is small enough that CPU inference is still
      // real-time.
      //
      // minHandDetectionConfidence: left at the plugin's own default
      // (0.5), not raised. An earlier build raised this to 0.6 on the
      // theory that stricter detection would reduce false positives, but
      // shipped with no way to verify that against a real device -- and
      // real-device testing then reported hand tracking itself getting
      // *worse* ("does not detect anything"), consistent with 0.6 simply
      // rejecting real hands 0.5 would have accepted. Reverted since there
      // was never actual evidence 0.6 helped, only a plausible-sounding
      // guess.
      _plugin ??= HandLandmarkerPlugin.create(
        numHands: 2,
        delegate: HandLandmarkerDelegate.cpu,
      );
      _landmarkSubscription ??= _plugin!.landmarkStream.listen(_onHands);

      await initFuture;
      await controller.startImageStream(_onCameraFrame);

      _controller = controller;
      if (mounted) setState(() => _stage = _Stage.ready);
    } catch (e) {
      // Some devices (extra lenses, driver quirks) fail to start a specific
      // camera. Fail loudly with a way back, never a silent stuck spinner.
      await controller?.dispose();
      debugPrint('Camera init failed ($_lensDirection): $e');
      if (!mounted) return;
      final name = _lensDirection == CameraLensDirection.back ? 'back' : 'front';
      setState(() {
        _cameraErrorMessage = "Couldn't start the $name camera on this device.";
        _stage = _Stage.cameraError;
      });
    }
  }

  Future<void> _switchCamera() async {
    // Only the camera controller needs to change -- the hand-landmarker
    // plugin's config doesn't depend on which camera feeds it, so it stays
    // alive and switching no longer re-pays its startup cost.
    final target =
        _lensDirection == CameraLensDirection.front ? CameraLensDirection.back : CameraLensDirection.front;
    await _controller?.stopImageStream();
    await _controller?.dispose();
    _controller = null;
    _lensDirection = target;
    if (mounted) setState(() => _stage = _Stage.initializing);
    await _bootstrap();
  }

  bool _wasStreamingBeforePause = false;

  /// Stops the camera image stream without tearing down the controller or
  /// the (expensive to recreate) MediaPipe plugin. Called whenever this
  /// screen is hidden but not destroyed -- switching to the Listen tab, or
  /// opening the About screen -- so recognition (and any auto-speak it
  /// would trigger) doesn't keep running behind a screen you can't see the
  /// camera feed on.
  Future<void> pause() async {
    // Idempotent: with three tabs now (Sign/Listen/Dictionary) instead of
    // two, this can be called twice in a row with no resume() in between
    // (e.g. Sign -> Listen -> Dictionary both pause Home). A second call
    // while already paused must be a no-op -- re-reading
    // isStreamingImages (now false) and writing that into
    // _wasStreamingBeforePause would silently overwrite the flag that
    // still correctly remembers the *original* paused-from state,
    // and resume() back on the Sign tab would then wrongly do nothing,
    // leaving the camera preview frozen with no image stream feeding
    // MediaPipe at all -- exactly "camera shows, but nothing is ever
    // detected", with no error anywhere to explain why.
    if (!(_controller?.value.isStreamingImages ?? false)) return;
    _wasStreamingBeforePause = true;
    await _controller?.stopImageStream();
  }

  /// Restarts the stream if [pause] had actually stopped one.
  Future<void> resume() async {
    if (_wasStreamingBeforePause && mounted && _stage == _Stage.ready) {
      await _controller?.startImageStream(_onCameraFrame);
    }
  }

  Future<void> _openAbout() async {
    await pause();
    if (!mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const AboutScreen()),
    );
    await resume();
  }

  /// Used by the camera-error screen to go back to whichever lens was
  /// working before the failed switch, instead of retrying the broken one.
  Future<void> _revertCamera() async {
    _lensDirection =
        _lensDirection == CameraLensDirection.front ? CameraLensDirection.back : CameraLensDirection.front;
    if (mounted) setState(() => _stage = _Stage.initializing);
    await _bootstrap();
  }

  void _onCameraFrame(CameraImage image) {
    final controller = _controller;
    if (controller == null) return;
    try {
      _plugin?.processFrame(image, controller.description.sensorOrientation);
    } catch (e) {
      // An occasional dropped frame is not worth surfacing to the user,
      // but this swallowed the exception message entirely -- if
      // processFrame is failing on *every* frame instead of rarely, this
      // silent catch is exactly what would make that look identical to
      // "no hand ever detected" from the outside, with nothing in the UI
      // to distinguish the two. Logged once, not every frame (this runs
      // at camera frame rate), so real-device logs can actually show
      // which failure mode this is instead of a wall of repeats.
      if (!_loggedFrameError) {
        _loggedFrameError = true;
        debugPrint('processFrame failed: $e');
      }
    }
  }

  // Confidence gate for control-vs-word arbitration: a raw control shape
  // (fist / thumbs-up / thumbs-down) is only *acted on* when the sign
  // classifier isn't confidently seeing a real word in that same shape.
  // Without this, closed-fist and thumb-only shapes would be unrecognizable
  // as words forever, since the control detector would always intercept
  // them first. The finger-count prototype's confidence is a hardcoded 1.0
  // (not a real signal), so this only helps once a trained model -- ASL or
  // BdSL -- is loaded and its confidence is real.
  // Minimum confidence to auto-commit a recognized sign as a word at all.
  // Without this, a resting/ambiguous hand shape the model isn't actually
  // confident about still gets *some* label (softmax always picks a best
  // guess), and if that guess is stable across frames it gets committed
  // just like a real sign -- the "same word keeps showing up out of
  // nowhere" failure mode. Below this threshold, treat it as no result
  // rather than inject a low-confidence guess into the sentence.
  //
  // Split by category, not one flat number, because they measurably don't
  // behave the same (see ml/eval_confidence.py, run against real held-out
  // data): a *correct* word prediction is usually confident (BdSL: 97% of
  // correct words still clear 0.60; ASL: 68%), but a correct digit or
  // letter prediction often isn't, simply because those vocabularies are
  // smaller sets of more visually similar handshapes -- at 0.55, BdSL
  // digit recall of correct predictions was only 78% and letter recall
  // only 85%, for barely any precision gain over a lower bar. The one flat
  // 0.55 used before this measurement silently discarded a lot of real,
  // correct digit/letter recognitions as if the sign had failed outright.
  //
  // FOLLOW-UP NEEDED: ml/eval_confidence.py computes confidence as
  // `scores.max(axis=1)` over the *entire* unmasked output (no category
  // restriction), while TfliteSignClassifier now renormalizes confidence
  // within the active category's allowed subset (see its _runInference
  // comment for why -- real-device testing found letters/digits
  // essentially never committing, root-caused to exactly this mismatch).
  // Renormalizing only increases a correct winner's reported confidence,
  // so these thresholds are not miscalibrated in the dangerous direction
  // (nothing that used to correctly commit will now be rejected), but
  // they were tuned against a different metric than the one now gating
  // real decisions. eval_confidence.py should be updated to mask-then-
  // renormalize per category before these numbers are retuned properly.
  static double _minCommitConfidenceFor(SignCategory category) {
    switch (category) {
      case SignCategory.word:
        return 0.55;
      case SignCategory.digit:
      case SignCategory.letter:
        return 0.4;
    }
  }

  void _onHands(List<Hand> hands) {
    final appState = context.read<AppState>();
    final classifier = _classifierFor(appState.language, appState.activeCategory);
    if (_categoryAppliedTo != classifier || _lastAppliedCategory != appState.activeCategory) {
      classifier.setAllowedCategories({appState.activeCategory});
      _categoryAppliedTo = classifier;
      _lastAppliedCategory = appState.activeCategory;
    }
    final rawResult = classifier.classify(hands);
    // Gate auto-commit (and the control-override check below) behind a
    // confidence bar, but keep showing rawResult as the live label
    // regardless -- see _minCommitConfidenceFor's comment. Nulling the
    // display too (an earlier build did this) means the app shows zero
    // feedback the instant a real, correctly-tracked sign just isn't
    // *quite* confident enough yet, which reads as "detects nothing" even
    // while it's actually working.
    final commitCandidate = rawResult != null &&
            rawResult.confidence >= _minCommitConfidenceFor(appState.activeCategory)
        ? rawResult
        : null;
    // Control handshapes (closed fist, thumb up/down) are checked in every
    // category, not just Words -- an earlier build restricted them to
    // Words mode only, on the theory that several real letters/digits
    // (ASL A/S/E/M/N/T, for instance) are closed-fist shapes that would
    // always lose the fist-shape tiebreak to backspace. That blanket
    // restriction went too far in practice: it silently disabled *all
    // three* hands-free gestures (backspace, clear, speak) any time the
    // user was in Letters or Digits mode, which is most of the time while
    // fingerspelling -- read by real-device testing as "the gestures
    // don't work," not as a deliberate scoping decision.
    //
    // The actual fix is confidence-based, same idea as the original
    // word-vs-control arbitration, now workable because letters/digits
    // are measured (ml/eval_confidence.py) to be extremely confident when
    // correctly formed (~99% recall at this same threshold) -- so a real,
    // deliberate letter/digit attempt reliably outscores a merely
    // fist-shaped "I meant to delete" hand and wins the tiebreak, while a
    // generic relaxed fist that doesn't closely match any trained pose
    // stays low-confidence and lets backspace fire. Clear/speak
    // (thumb clearly extended, pointing up or down, no other fingers) are
    // shaped nothing like any trained letter or digit pose, so they were
    // never really at risk -- but the same single mechanism covers all
    // three gestures in every category rather than special-casing each.
    final rawControl =
        appState.controlGesturesEnabled ? _controlDetector.detect(hands) : ControlGesture.none;
    // wordIsConfident used to require an *additional*, flat 0.6 bar on top
    // of commitCandidate's own category-specific threshold (0.4 for
    // letters/digits, 0.55 for words) -- caught live on a real device: a
    // held "S" scored in that 0.4-0.6 gap (good enough to eventually
    // commit as a letter, not good enough to block control), so the raw
    // fist-shape reading won the race and backspace deleted a previous
    // word while the user was mid-sign, not mid-delete. There is no
    // principled reason for control-suppression to demand a *higher* bar
    // than the app already uses to decide "this is a real enough sign to
    // commit" -- if a frame clears that bar, it should also be enough to
    // stop an incidental fist-shape collision from winning. Tying the two
    // together removes the gap instead of just narrowing it.
    final wordIsConfident = !classifier.isPrototype && commitCandidate != null;
    final now = DateTime.now();
    // See ControlArbiter's docstring: a per-frame confidence check alone
    // isn't enough to protect closed-fist letters (ASL "S"/"E"/"Q" and
    // others measure as a fist shape on the overwhelming majority of their
    // own training samples), because the backspace hold needs ~900ms of
    // *continuous* low confidence to fire, and real-device confidence can
    // dip that long during genuine signing. The arbiter bridges brief dips
    // via hysteresis instead of loosening the geometry check or the
    // confidence bar itself.
    final control = _controlArbiter.arbitrate(
      wordIsConfident: wordIsConfident,
      rawControl: rawControl,
      now: now,
    );

    if (hands.isEmpty) {
      // A stale "recently confident" timestamp from before the hand left
      // frame must not suppress an unrelated control gesture made after
      // it returns.
      _controlArbiter.reset();
    } else {
      _lastHandSeenAt = now;
      _autoSpokenForSentence = false;
    }

    if (appState.autoAssist) {
      _updateControlStability(control, appState, now);
      // A control handshape (fist/thumbs-down) takes priority over word
      // recognition -- they're deliberately distinct from real signs, so
      // if one is showing, don't also try to commit it as a word.
      if (control == ControlGesture.none) {
        _updateWordStability(commitCandidate, appState);
      } else {
        _wordStableLabel = null;
        _wordStableSince = null;
      }
      _maybeAutoSpeak(appState, now);
    }

    if (!mounted) return;
    setState(() {
      _hands = hands;
      _liveResult = rawResult;
      _activeControl = control;
      _stabilityProgress = _computeStabilityProgress(commitCandidate, control, now);
    });
  }

  void _updateWordStability(ClassificationResult? result, AppState appState) {
    if (result == null) {
      _wordStableLabel = null;
      _wordStableSince = null;
      _lastCommittedWord = null;
      return;
    }
    if (result.label != _wordStableLabel) {
      _wordStableLabel = result.label;
      _wordStableSince = DateTime.now();
      return;
    }
    final since = _wordStableSince;
    if (since == null) return;
    if (DateTime.now().difference(since) >= _wordHoldDuration && result.label != _lastCommittedWord) {
      _commitWord(result.label, appState);
      _lastCommittedWord = result.label;
    }
  }

  /// In Letters mode, committed letters accumulate into [_spellBuffer]
  /// instead of becoming their own auto-spaced sentence entry -- spelling
  /// "C" "A" "T" builds "CAT", not "C A T". Everywhere else, unchanged:
  /// a recognized word goes straight into the sentence.
  void _commitWord(String label, AppState appState) {
    final word =
        appState.language == SignLanguage.bdsl ? bengaliFor(label) : aslDisplayFor(label);
    if (appState.activeCategory == SignCategory.letter) {
      setState(() {
        _spellBuffer += word;
        _suggestions = _completerFor(appState.language)?.suggest(_spellBuffer) ?? const [];
      });
      HapticFeedback.mediumImpact();
      return;
    }
    appState.appendWord(word);
    HapticFeedback.mediumImpact();
  }

  /// Commits whatever's been spelled so far as one word and clears the
  /// buffer. Called on mode switch, Clear, and before Speak -- never left
  /// implicitly abandoned.
  void _flushSpellBuffer(AppState appState) {
    if (_spellBuffer.isEmpty) return;
    appState.appendWord(_spellBuffer);
    setState(() {
      _spellBuffer = '';
      _suggestions = const [];
    });
  }

  /// Replaces the in-progress spelling with a tapped completion suggestion.
  void _acceptSuggestion(AppState appState, String word) {
    appState.appendWord(word);
    setState(() {
      _spellBuffer = '';
      _suggestions = const [];
    });
    HapticFeedback.mediumImpact();
  }

  /// Backspace while spelling removes the last *letter* from the buffer,
  /// not the previous already-committed sentence word -- matches how a
  /// real keyboard's backspace behaves while mid-word.
  void _handleBackspace(AppState appState) {
    if (_spellBuffer.isNotEmpty) {
      setState(() {
        _spellBuffer = _spellBuffer.substring(0, _spellBuffer.length - 1);
        _suggestions = _spellBuffer.isEmpty
            ? const []
            : (_completerFor(appState.language)?.suggest(_spellBuffer) ?? const []);
      });
      return;
    }
    if (appState.sentence.isNotEmpty) appState.backspace();
  }

  void _handleClearAll(AppState appState) {
    setState(() {
      _spellBuffer = '';
      _suggestions = const [];
    });
    if (appState.sentence.isNotEmpty) appState.clearSentence();
  }

  Future<void> _handleSpeak(AppState appState) async {
    _flushSpellBuffer(appState);
    final text = appState.sentenceText;
    // Record on speak rather than on every committed word, so history
    // holds complete utterances instead of every partial state the
    // sentence passed through on the way there.
    unawaited(context.read<PhraseStore>().recordSpoken(appState.language, text));
    await _speak(text, appState.language);
  }

  /// Speaks a stored phrase directly, without disturbing the sentence the
  /// user may be part-way through building. Deliberately does not touch
  /// the spell buffer or the sentence: reaching for a stock phrase
  /// mid-utterance should not destroy the utterance.
  Future<void> _speakPhrase(String phrase) async {
    final appState = context.read<AppState>();
    await _speak(phrase, appState.language);
  }

  /// Loads a stored phrase into the sentence area, replacing whatever was
  /// there, so it can be extended or edited before speaking.
  void _usePhrase(String phrase) {
    final appState = context.read<AppState>();
    setState(() {
      _spellBuffer = '';
      _suggestions = const [];
    });
    appState.replaceSentence(phrase);
    _lastCommittedWord = null;
  }

  Future<void> _openPhrases() async {
    final appState = context.read<AppState>();
    _flushSpellBuffer(appState);
    await PhrasesSheet.show(
      context,
      currentSentence: appState.sentenceText,
      onUsePhrase: _usePhrase,
      onSpeakPhrase: _speakPhrase,
    );
  }

  void _updateControlStability(ControlGesture control, AppState appState, DateTime now) {
    if (control == ControlGesture.none) {
      _controlStableGesture = ControlGesture.none;
      _controlStableSince = null;
      _controlFiredFor = ControlGesture.none;
      return;
    }
    if (control != _controlStableGesture) {
      _controlStableGesture = control;
      _controlStableSince = now;
      return;
    }
    final since = _controlStableSince;
    if (since == null) return;
    final required = _requiredHoldFor(control);
    if (now.difference(since) >= required && control != _controlFiredFor) {
      _controlFiredFor = control;
      if (control == ControlGesture.backspace &&
          (appState.sentence.isNotEmpty || _spellBuffer.isNotEmpty)) {
        _handleBackspace(appState);
        HapticFeedback.heavyImpact();
      } else if (control == ControlGesture.clearAll &&
          (appState.sentence.isNotEmpty || _spellBuffer.isNotEmpty)) {
        _handleClearAll(appState);
        HapticFeedback.heavyImpact();
      } else if (control == ControlGesture.speak &&
          (appState.sentence.isNotEmpty || _spellBuffer.isNotEmpty)) {
        HapticFeedback.heavyImpact();
        _autoSpokenForSentence = true; // don't also fire the pause-based auto-speak right after
        _handleSpeak(appState);
      }
    }
  }

  Duration _requiredHoldFor(ControlGesture control) {
    switch (control) {
      case ControlGesture.clearAll:
        return _clearHoldDuration;
      case ControlGesture.speak:
        return _speakHoldDuration;
      case ControlGesture.backspace:
      case ControlGesture.none:
        return _backspaceHoldDuration;
    }
  }

  void _maybeAutoSpeak(AppState appState, DateTime now) {
    if (_autoSpokenForSentence || (appState.sentence.isEmpty && _spellBuffer.isEmpty)) return;
    final lastSeen = _lastHandSeenAt;
    if (lastSeen != null && now.difference(lastSeen) >= _silenceBeforeSpeak) {
      _autoSpokenForSentence = true;
      _handleSpeak(appState);
    }
  }

  double _computeStabilityProgress(ClassificationResult? result, ControlGesture control, DateTime now) {
    if (control != ControlGesture.none && _controlStableSince != null) {
      final required = _requiredHoldFor(control);
      return (now.difference(_controlStableSince!).inMilliseconds / required.inMilliseconds).clamp(0, 1);
    }
    if (result != null && result.label == _wordStableLabel && _wordStableSince != null) {
      return (now.difference(_wordStableSince!).inMilliseconds / _wordHoldDuration.inMilliseconds).clamp(0, 1);
    }
    return 0;
  }

  Future<void> _speak(String text, SignLanguage language) async {
    if (text.trim().isEmpty) return;
    final requested = language == SignLanguage.bdsl ? 'bn-BD' : 'en-US';
    // Confirm the voice actually exists on this device before committing
    // to it -- setLanguage on a missing voice can silently no-op on some
    // devices/engines rather than error, which otherwise looks identical
    // to "Speak does nothing" with no clue why.
    var localeId = requested;
    try {
      final available = await _tts.isLanguageAvailable(requested);
      if (available != true) {
        debugPrint('TTS: $requested unavailable on this device, falling back to en-US');
        localeId = 'en-US';
      }
    } catch (e) {
      debugPrint('TTS: isLanguageAvailable check failed ($e), trying $requested anyway');
    }
    await _tts.setLanguage(localeId);
    await _tts.setSpeechRate(0.45);
    await _tts.speak(text);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.stopImageStream();
    _controller?.dispose();
    _landmarkSubscription?.cancel();
    _plugin?.dispose();
    _aslClassifier?.dispose();
    _aslStaticClassifier?.dispose();
    _bdslClassifier?.dispose();
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    super.build(context); // required by AutomaticKeepAliveClientMixin
    final appState = context.watch<AppState>();
    final rawLabel = _liveResult?.label;
    final displayLabel = rawLabel == null
        ? null
        : (appState.language == SignLanguage.bdsl ? bengaliFor(rawLabel) : aslDisplayFor(rawLabel));
    return Scaffold(
      appBar: AppBar(
        title: const BrandTitle('SignBridge'),
        actions: [
          // An emergency action gets a direct one-tap icon, not a spot in
          // the overflow menu -- the two extra taps a menu costs are
          // exactly the wrong tradeoff for something meant to be used
          // under stress. Auto mode and the hand-shape-gesture toggle stay
          // in the overflow menu below (a labeled row above the camera
          // was tried and reverted -- it shrank the camera preview, which
          // matters more than saving one menu tap for two settings that
          // are not used every session).
          IconButton(
            icon: const Icon(Icons.emergency_outlined),
            color: AppColors.coral,
            tooltip: 'Emergency SOS',
            onPressed: () => SosSheet.show(context),
          ),
          IconButton(
            icon: const Icon(Icons.bookmarks_outlined),
            tooltip: 'My phrases — saved and recently spoken',
            onPressed: _openPhrases,
          ),
          IconButton(
            icon: const Icon(Icons.cameraswitch_outlined),
            tooltip: 'Switch camera',
            onPressed: _stage == _Stage.ready || _stage == _Stage.cameraError ? _switchCamera : null,
          ),
          // The less-frequent controls live in an overflow menu rather than
          // as more icons: five icon buttons plus the title overflow the
          // app bar on a ~360dp-wide phone, which is squarely inside this
          // app's target range.
          PopupMenuButton<String>(
            tooltip: 'More options',
            onSelected: (value) {
              if (value == 'theme') {
                // Cycles rather than a picker dialog -- three states, one
                // tap each.
                const order = [ThemeMode.system, ThemeMode.light, ThemeMode.dark];
                final next = order[(order.indexOf(appState.themeMode) + 1) % order.length];
                appState.setThemeMode(next);
              } else if (value == 'ui_language') {
                appState.toggleUiLanguage();
              } else if (value == 'gestures') {
                appState.toggleControlGestures();
              } else if (value == 'auto_mode') {
                appState.toggleAutoAssist();
              } else if (value == 'about') {
                _openAbout();
              }
            },
            itemBuilder: (context) => [
              PopupMenuItem(
                value: 'theme',
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(switch (appState.themeMode) {
                    ThemeMode.light => Icons.light_mode,
                    ThemeMode.dark => Icons.dark_mode,
                    ThemeMode.system => Icons.brightness_auto,
                  }),
                  title: Text(switch (appState.themeMode) {
                    ThemeMode.light => 'Theme: Light',
                    ThemeMode.dark => 'Theme: Dark',
                    ThemeMode.system => 'Theme: Match device',
                  }),
                  subtitle: const Text('Tap to switch', style: TextStyle(fontSize: 11.5)),
                ),
              ),
              PopupMenuItem(
                value: 'ui_language',
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: const Icon(Icons.translate),
                  // Independent of the ASL/BdSL segmented control above --
                  // that picks which sign language the camera reads, this
                  // picks what script the app's own buttons/menus use.
                  title: Text(appState.uiBangla ? 'App language: বাংলা' : 'App language: English'),
                  subtitle: const Text('Tap to switch', style: TextStyle(fontSize: 11.5)),
                ),
              ),
              PopupMenuItem(
                value: 'auto_mode',
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(
                    appState.autoAssist ? Icons.auto_awesome : Icons.auto_awesome_outlined,
                    color: appState.autoAssist ? AppColors.amber : null,
                  ),
                  title: Text(appState.autoAssist ? 'Auto mode: On' : 'Auto mode: Off'),
                  subtitle: Text(
                    appState.autoAssist
                        ? 'Hold signs to add, pause to speak'
                        : 'Use the buttons below instead',
                    style: const TextStyle(fontSize: 11.5),
                  ),
                ),
              ),
              PopupMenuItem(
                value: 'gestures',
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(
                    appState.controlGesturesEnabled ? Icons.back_hand : Icons.back_hand_outlined,
                    color: appState.controlGesturesEnabled ? AppColors.amber : null,
                  ),
                  title: Text(appState.controlGesturesEnabled
                      ? 'Hand-shape gestures: On'
                      : 'Hand-shape gestures: Off'),
                  subtitle: Text(
                    appState.controlGesturesEnabled
                        ? 'Fist / thumbs-up / thumbs-down control the sentence'
                        : 'Use the Delete / Clear / Speak buttons instead',
                    style: const TextStyle(fontSize: 11.5),
                  ),
                ),
              ),
              PopupMenuItem(
                value: 'about',
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: const Icon(Icons.info_outline),
                  title: Text(uiText(appState.uiBangla, 'about_this_build', 'About this build')),
                ),
              ),
            ],
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            _LanguageModeBar(appState: appState),
            _CategoryModeBar(
              appState: appState,
              onChanged: (category) {
                _flushSpellBuffer(appState);
                appState.setActiveCategory(category);
              },
            ),
            Expanded(child: _buildCameraArea()),
            _CaptionPanel(
              appState: appState,
              liveResult: _liveResult,
              displayLabel: displayLabel,
              isPrototype: _classifierFor(appState.language, appState.activeCategory).isPrototype,
              activeControl: _activeControl,
              stabilityProgress: _stabilityProgress,
              spellBuffer: _spellBuffer,
              suggestions: _suggestions,
              onCommit: () {
                final label = _liveResult?.label;
                if (label != null) _commitWord(label, appState);
              },
              onAcceptSuggestion: (word) => _acceptSuggestion(appState, word),
              onBackspace: () => _handleBackspace(appState),
              onClear: () => _handleClearAll(appState),
              onSpeak: () => _handleSpeak(appState),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraArea() {
    switch (_stage) {
      case _Stage.requestingPermission:
      case _Stage.initializing:
        return const Center(child: CircularProgressIndicator());
      case _Stage.permissionDenied:
        return _CameraMessage(
          icon: Icons.videocam_off_outlined,
          title: 'Camera access needed',
          message:
              'SignBridge reads hand shapes through the camera to recognize signs. '
              'Please grant camera access in system settings to continue.',
          action: FilledButton(
            onPressed: openAppSettings,
            child: const Text('Open settings'),
          ),
        );
      case _Stage.noCamera:
        return const _CameraMessage(
          icon: Icons.no_photography_outlined,
          title: 'No camera found',
          message: 'This device has no usable camera.',
        );
      case _Stage.cameraError:
        return _CameraMessage(
          icon: Icons.error_outline,
          title: 'Camera unavailable',
          message: _cameraErrorMessage ?? 'Could not start the camera.',
          action: FilledButton.icon(
            onPressed: _revertCamera,
            icon: const Icon(Icons.cameraswitch_outlined, size: 18),
            label: const Text('Switch camera'),
          ),
        );
      case _Stage.ready:
        final controller = _controller!;
        final previewSize = controller.value.previewSize!;
        final aspect = previewSize.height / previewSize.width;
        // A colored frame keyed to the active sign language -- a real,
        // functional at-a-glance signal for which language is currently
        // being read (easy to lose track of after switching modes a few
        // times), not just decoration.
        final languageAccent = context.read<AppState>().language == SignLanguage.bdsl
            ? AppColors.bdslAccent
            : AppColors.primary;
        return Center(
          child: AspectRatio(
            aspectRatio: aspect,
            child: Container(
              padding: const EdgeInsets.all(2.5),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: languageAccent, width: 2.5),
                boxShadow: [
                  BoxShadow(color: languageAccent.withValues(alpha: 0.25), blurRadius: 10, spreadRadius: 1),
                ],
              ),
              child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(controller),
                  CustomPaint(
                    size: Size.infinite,
                    painter: LandmarkPainter(
                      hands: _hands,
                      previewSize: previewSize,
                      lensDirection: controller.description.lensDirection,
                      sensorOrientation: controller.description.sensorOrientation,
                    ),
                  ),
                  Positioned(
                    top: 10,
                    left: 10,
                    child: _TrackingBadge(tracking: _hands.isNotEmpty),
                  ),
                  if (_activeControl != ControlGesture.none)
                    Positioned(
                      top: 10,
                      right: 10,
                      child: _ControlHintBadge(gesture: _activeControl),
                    ),
                ],
              ),
            ),
            ),
          ),
        );
    }
  }
}

class _CameraMessage extends StatelessWidget {
  const _CameraMessage({
    required this.icon,
    required this.title,
    required this.message,
    this.action,
  });

  final IconData icon;
  final String title;
  final String message;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 48, color: AppColors.inkSoft),
            const SizedBox(height: 12),
            Text(title, style: Theme.of(context).textTheme.titleMedium, textAlign: TextAlign.center),
            const SizedBox(height: 8),
            Text(message, textAlign: TextAlign.center, style: const TextStyle(color: AppColors.inkSoft)),
            if (action != null) ...[const SizedBox(height: 16), action!],
          ],
        ),
      ),
    );
  }
}

class _TrackingBadge extends StatelessWidget {
  const _TrackingBadge({required this.tracking});
  final bool tracking;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(99),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: tracking ? const Color(0xFF4ADE80) : Colors.white38,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            tracking ? 'Tracking' : 'No hand',
            style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

class _ControlHintBadge extends StatelessWidget {
  const _ControlHintBadge({required this.gesture});
  final ControlGesture gesture;

  @override
  Widget build(BuildContext context) {
    final String label;
    final Color color;
    switch (gesture) {
      case ControlGesture.backspace:
        label = 'Hold to delete word';
        color = AppColors.coral;
      case ControlGesture.clearAll:
        label = 'Hold to clear all';
        color = AppColors.coral;
      case ControlGesture.speak:
        label = 'Hold to speak now';
        color = AppColors.primary;
      case ControlGesture.none:
        label = '';
        color = AppColors.coral;
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(99),
      ),
      child: Text(
        label,
        style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600),
      ),
    );
  }
}

class _LanguageModeBar extends StatelessWidget {
  const _LanguageModeBar({required this.appState});
  final AppState appState;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: Row(
        children: [
          // Core/Full mode selection is hidden for now: both languages
          // currently ship exactly one trained vocabulary each, so the
          // choice has no effect yet. It'll come back once there's an
          // actual larger vs. curated vocabulary to choose between (see
          // ml/README.md).
          Expanded(
            child: SegmentedButton<SignLanguage>(
              segments: const [
                ButtonSegment(value: SignLanguage.asl, label: Text('ASL · English')),
                ButtonSegment(value: SignLanguage.bdsl, label: Text('BdSL · বাংলা')),
              ],
              selected: {appState.language},
              onSelectionChanged: (s) => appState.setLanguage(s.first),
            ),
          ),
        ],
      ),
    );
  }
}

/// Restricts recognition to one [SignCategory] at a time -- Words,
/// Digits (BdSL only), or Letters -- instead of every trained class
/// competing against every other at once. This is what stops a held-
/// steady letter from losing to a superficially similar word (or vice
/// versa) purely because both live in the same model's output; see
/// TfliteSignClassifier.setAllowedCategories.
class _CategoryModeBar extends StatelessWidget {
  const _CategoryModeBar({required this.appState, required this.onChanged});
  final AppState appState;
  final ValueChanged<SignCategory> onChanged;

  @override
  Widget build(BuildContext context) {
    final available = appState.language.availableCategories;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
      child: SegmentedButton<SignCategory>(
        segments: [
          for (final category in available)
            ButtonSegment(value: category, label: Text(category.displayName)),
        ],
        selected: {appState.activeCategory},
        onSelectionChanged: (s) => onChanged(s.first),
      ),
    );
  }
}


class _CaptionPanel extends StatelessWidget {
  const _CaptionPanel({
    required this.appState,
    required this.liveResult,
    required this.displayLabel,
    required this.isPrototype,
    required this.activeControl,
    required this.stabilityProgress,
    required this.spellBuffer,
    required this.suggestions,
    required this.onCommit,
    required this.onAcceptSuggestion,
    required this.onBackspace,
    required this.onClear,
    required this.onSpeak,
  });

  final AppState appState;
  final ClassificationResult? liveResult;
  final String? displayLabel;
  final bool isPrototype;
  final ControlGesture activeControl;
  final double stabilityProgress;
  final String spellBuffer;
  final List<String> suggestions;
  final VoidCallback onCommit;
  final ValueChanged<String> onAcceptSuggestion;
  final VoidCallback onBackspace;
  final VoidCallback onClear;
  final VoidCallback onSpeak;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final progressColor = activeControl != ControlGesture.none ? AppColors.coral : AppColors.amber;
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      decoration: BoxDecoration(
        color: scheme.surface,
        border: Border(top: BorderSide(color: scheme.outlineVariant)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: Text(
                  displayLabel ?? '—',
                  style: const TextStyle(fontSize: 34, fontWeight: FontWeight.w800, color: AppColors.amber),
                ),
              ),
              if (isPrototype)
                Container(
                  margin: const EdgeInsets.only(bottom: 8),
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: scheme.secondaryContainer,
                    borderRadius: BorderRadius.circular(99),
                  ),
                  child: const Text(
                    'PROTOTYPE ENGINE',
                    style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, letterSpacing: 0.4),
                  ),
                ),
              IconButton.filled(
                iconSize: 24,
                onPressed: liveResult == null ? null : onCommit,
                icon: const Icon(Icons.add),
                tooltip: 'Add to sentence now',
              ),
            ],
          ),
          ClipRRect(
            borderRadius: BorderRadius.circular(99),
            child: LinearProgressIndicator(
              value: stabilityProgress == 0 ? null : stabilityProgress,
              minHeight: 4,
              backgroundColor: scheme.surfaceContainerHighest,
              color: stabilityProgress == 0 ? Colors.transparent : progressColor,
            ),
          ),
          const SizedBox(height: 8),
          _SentenceBox(
            scheme: scheme,
            sentenceText: appState.sentence.isNotEmpty ? appState.sentenceText : null,
            spellBuffer: spellBuffer,
          ),
          const SizedBox(height: 8),
          // Fixed-height slot, always present (even with zero suggestions)
          // rather than only inserted when there's something to show --
          // real-device testing found the camera preview above visibly
          // growing and shrinking every time a suggestion chip row
          // appeared/disappeared, because the whole caption panel sits
          // beside the camera area inside an Expanded, so any change in
          // this panel's own height changes how much space "remaining"
          // the camera preview is given. Reserving the space unconditionally
          // keeps the panel's height constant regardless of suggestion
          // count, so the camera preview never resizes because of typing.
          SizedBox(
            height: 36,
            // Horizontally scrollable, not a Wrap -- a Wrap can spill onto
            // a second line once its content is wide enough (longer
            // Bangla words render wider per character than English in
            // this app's font), and a second line would just get clipped
            // by this box's fixed height rather than actually showing.
            // Scrolling sideways instead guarantees exactly one line no
            // matter how long any suggestion is, so the fixed-height
            // reservation above always holds.
            child: suggestions.isEmpty
                ? null
                : ListView(
                    scrollDirection: Axis.horizontal,
                    children: [
                      for (final word in suggestions)
                        Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: ActionChip(
                            label: Text(word),
                            onPressed: () => onAcceptSuggestion(word),
                          ),
                        ),
                    ],
                  ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 6),
                    minimumSize: const Size(0, 42),
                  ),
                  onPressed: appState.sentence.isEmpty && spellBuffer.isEmpty ? null : onBackspace,
                  icon: const Icon(Icons.backspace_outlined, size: 16),
                  label: Text(
                    uiText(appState.uiBangla, 'delete', 'Delete'),
                    style: const TextStyle(fontSize: 12),
                    overflow: TextOverflow.ellipsis,
                    maxLines: 1,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 6),
                    minimumSize: const Size(0, 42),
                  ),
                  onPressed: appState.sentence.isEmpty && spellBuffer.isEmpty ? null : onClear,
                  icon: const Icon(Icons.clear, size: 16),
                  label: Text(
                    uiText(appState.uiBangla, 'clear', 'Clear'),
                    style: const TextStyle(fontSize: 12),
                    overflow: TextOverflow.ellipsis,
                    maxLines: 1,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                flex: 2,
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(minimumSize: const Size(0, 42)),
                  onPressed: appState.sentence.isEmpty && spellBuffer.isEmpty ? null : onSpeak,
                  icon: const Icon(Icons.volume_up_rounded, size: 18),
                  label: Text(
                    uiText(appState.uiBangla, 'speak', 'Speak'),
                    style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// The sentence display box. Bounded-height and internally scrollable
/// rather than growing to fit its content -- an earlier version had no
/// height cap at all, so a long enough sentence (there is no limit on how
/// many words a user can add, see AppState._sentence) would keep wrapping
/// onto more lines and shrinking the camera preview above it, since both
/// sit in a plain Column with only the camera area as an Expanded: any
/// growth here came directly out of the camera's share of the screen.
/// Auto-scrolls to the newest content on every change, so the word just
/// added is always the one in view without the user having to swipe.
class _SentenceBox extends StatefulWidget {
  const _SentenceBox({required this.scheme, required this.sentenceText, required this.spellBuffer});

  final ColorScheme scheme;
  final String? sentenceText;
  final String spellBuffer;

  @override
  State<_SentenceBox> createState() => _SentenceBoxState();
}

class _SentenceBoxState extends State<_SentenceBox> {
  final ScrollController _scrollController = ScrollController();

  // A fixed cap tall enough for roughly three lines at this text's font
  // size (18) plus the box's own padding -- generous for a normal
  // sentence, while guaranteeing the camera above it never shrinks no
  // matter how long the sentence grows.
  static const double _maxHeight = 104;

  @override
  void didUpdateWidget(_SentenceBox oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.sentenceText != widget.sentenceText || oldWidget.spellBuffer != widget.spellBuffer) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!_scrollController.hasClients) return;
        _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
      });
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isEmpty = widget.sentenceText == null && widget.spellBuffer.isEmpty;
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(minHeight: 44, maxHeight: _maxHeight),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: widget.scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(10),
      ),
      child: isEmpty
          ? const Text(
              'Recognized words appear here…',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600, color: AppColors.inkSoft),
            )
          : SingleChildScrollView(
              controller: _scrollController,
              child: RichText(
                text: TextSpan(
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: Colors.black,
                  ),
                  children: [
                    if (widget.sentenceText != null) TextSpan(text: '${widget.sentenceText} '),
                    // Still-being-spelled letters, visually distinct from
                    // already-committed words (see
                    // HomeScreenState._spellBuffer) -- not yet a real
                    // sentence entry, so it reads differently on purpose.
                    if (widget.spellBuffer.isNotEmpty)
                      TextSpan(
                        text: widget.spellBuffer,
                        style: const TextStyle(color: AppColors.primary, letterSpacing: 1),
                      ),
                  ],
                ),
              ),
            ),
    );
  }
}
