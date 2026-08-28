import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart' show rootBundle;
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'gesture_segmenter.dart';
import 'sign_category.dart';
import 'sign_classifier.dart';

/// Runs a trained landmark-sequence `.tflite` model (see ../../../ml/train.py)
/// over a rolling window of the last [seqLen] hand-tracking frames.
///
/// Inference runs on a background isolate ([IsolateInterpreter]), never on
/// the UI thread: `classify()` is called once per camera frame and must
/// return immediately or the camera preview itself starts to stutter. It
/// kicks off at most one inference at a time and returns whatever the last
/// completed inference produced -- a frame or two of lag, which is far
/// cheaper than blocking the UI thread on every single frame.
///
/// Known limitation: training resampled each *whole gesture* to exactly
/// [seqLen] frames regardless of how long it took, so the model learned
/// motion shape independent of speed. This classifier instead uses a plain
/// rolling window of the last [seqLen] raw camera frames -- there is no
/// real-time gesture segmentation yet (see the plan's Phase 4 "debounce
/// predictions" task, not implemented). In practice this still produces a
/// meaningful signal as long as the sign is held for roughly the window's
/// duration, which is why the UI has the user hold a sign steady rather
/// than committing every frame's prediction.
class TfliteSignClassifier implements SignClassifier {
  TfliteSignClassifier._(
    this._interpreter,
    this._isolateInterpreter,
    this._labels,
    this._name,
    this.seqLen,
    this._expectsAnchor,
  ) {
    _allowedMask = List.filled(_labels.length, true);
  }

  final Interpreter _interpreter;
  final IsolateInterpreter _isolateInterpreter;
  final List<String> _labels;
  final String _name;
  final int seqLen;

  /// Whether this model was trained with the trailing in-frame position
  /// anchor row (43 rows per frame rather than 42) -- read from the
  /// model's own input tensor at load time rather than hardcoded, so a
  /// model built either way runs correctly without a matching code
  /// change. Feeding the wrong row count is not a graceful failure: it is
  /// a shape mismatch at best and silently misaligned features at worst.
  final bool _expectsAnchor;
  final Queue<List<List<double>>> _buffer = Queue();
  ClassificationResult? _latestResult;
  bool _inferring = false;

  // Precomputed once per setAllowedCategories call (not per-frame): which
  // label indices the argmax in _runInference is allowed to consider.
  // Defaults to every label, so a classifier nobody has restricted yet
  // behaves exactly as before the mode toggle existed.
  late List<bool> _allowedMask;

  static Future<TfliteSignClassifier> create({
    required String modelAsset,
    required String labelsAsset,
    required String displayName,
    int seqLen = 40,
  }) async {
    final interpreter = await Interpreter.fromAsset(modelAsset);
    final isolateInterpreter = await IsolateInterpreter.create(address: interpreter.address);
    final labelsJson = await rootBundle.loadString(labelsAsset);
    final labels = (jsonDecode(labelsJson) as List).cast<String>();

    // Input shape is [batch, seqLen, rowsPerFrame, 3]. rowsPerFrame is 42
    // for the two-hand landmark block alone, or 43 when the training
    // pipeline also appended the position anchor (ml/src/landmarks.py's
    // frame_anchor). Detect it instead of assuming.
    final inputShape = interpreter.getInputTensor(0).shape;
    final rowsPerFrame = inputShape.length >= 3 ? inputShape[inputShape.length - 2] : _numLandmarks * 2;
    final expectsAnchor = rowsPerFrame == _numLandmarks * 2 + 1;

    return TfliteSignClassifier._(
      interpreter, isolateInterpreter, labels, displayName, seqLen, expectsAnchor,
    );
  }

  @override
  String get modelName => _name;

  @override
  bool get isPrototype => false;

  @override
  void setAllowedCategories(Set<SignCategory> categories) {
    _allowedMask = _labels.map((l) => categories.contains(categoryOf(l))).toList();
    // Digits and letters are single-hand-trained in both ASL and BdSL (see
    // _buildTwoHandFrame); only word signs genuinely use a second hand.
    // The mode toggle already restricts the active category to exactly
    // one at a time, so "word is not among the allowed categories" is
    // exactly "the active mode is single-hand-only".
    _singleHandOnly = !categories.contains(SignCategory.word);
  }

  bool _singleHandOnly = false;

  static const int _numLandmarks = 21;

  // --- Gesture segmentation -------------------------------------------
  //
  // Training resamples each *complete, pre-segmented* gesture to exactly
  // seqLen frames, so the model learned motion shape independent of how
  // long the sign took. Inference originally fed it a rolling window of
  // the last seqLen raw camera frames instead -- a window that can
  // straddle two signs, or hold a half-finished one, and whose effective
  // playback speed depends entirely on the camera's frame rate. That is a
  // real train/inference mismatch, and it is why the interface had to ask
  // the user to hold every sign steady.
  //
  // The fix is to find the gesture's boundaries and resample what is
  // between them, exactly as training does. A sign is movement followed
  // by a settle, so per-frame motion energy separates the two: while the
  // hand is moving we accumulate; once motion stays below a threshold for
  // a few consecutive frames the sign is treated as finished, the
  // accumulated frames are resampled to seqLen, and that is what gets
  // classified.
  //
  // The rolling window is deliberately KEPT as a fallback rather than
  // replaced. Continuous or fidgety signing may never produce a clean
  // settle, and a segmenter that waits forever for one would make the app
  // stop responding entirely -- a far worse failure than a slightly
  // misaligned window. Whichever path produces a result, it is one
  // seqLen-frame tensor of the same shape.

  // The detector itself lives in GestureSegmenter, which is pure logic
  // over landmark frames and therefore unit-testable without a model or a
  // camera (see test/gesture_segmenter_test.dart).
  final GestureSegmenter _segmenter = GestureSegmenter();

  @override
  ClassificationResult? classify(List<Hand> hands) {
    final detected = hands.where((h) => h.landmarks.length >= _numLandmarks).toList();
    // In Digits/Letters mode, a second hand in frame (most often the one
    // holding the phone) is noise, not signal -- feeding it into slot 1
    // below would re-anchor the normalization origin/scale to the wrong
    // hand entirely (see _buildTwoHandFrame's "prefer slot 1" comment),
    // corrupting the one hand that actually matters. Collapsing to the
    // single most likely signing hand up front matches what training saw
    // and keeps the segmenter's own motion tracking from being confused by
    // a hand that was never part of the sign.
    final valid = _singleHandOnly && detected.length > 1
        ? [primaryHand(detected)!]
        : detected;
    if (valid.isEmpty) {
      // Hand left frame -- itself a gesture boundary, since lowering the
      // hand is the most natural way for a signer to finish.
      final finished = _segmenter.flush(seqLen);
      if (finished != null) _startInference(finished);
      _buffer.clear();
      _segmenter.reset();
      _latestResult = null;
      return null;
    }

    final frame = _buildTwoHandFrame(valid);

    // Rolling-window fallback, maintained in parallel so the live label
    // keeps updating mid-sign and so continuous signing that never settles
    // still produces results.
    _buffer.addLast(frame);
    while (_buffer.length > seqLen) {
      _buffer.removeFirst();
    }

    final completed = _segmenter.add(frame, seqLen);
    if (completed != null) {
      _startInference(completed);
      return _latestResult;
    }

    if (_buffer.length == seqLen) _startInference(_buffer.toList());
    return _latestResult;
  }

  /// Runs at most one inference at a time; a request arriving while one is
  /// in flight is dropped rather than queued, since by the time a stale
  /// window finished it would no longer describe what the hand is doing.
  void _startInference(List<List<List<double>>> window) {
    if (_inferring) return;
    _inferring = true;
    unawaited(_runInference(window));
  }

  Future<void> _runInference(List<List<List<double>>> window) async {
    try {
      final input = [window];
      final output = [List<double>.filled(_labels.length, 0.0)];
      await _isolateInterpreter.run(input, output);

      final scores = output[0];
      // Only consider labels in the currently active category (see
      // setAllowedCategories) -- this is what stops a held-steady letter
      // from losing to a superficially similar word, or vice versa, just
      // because both live in the same underlying model's output.
      var bestIdx = -1;
      var allowedSum = 0.0;
      for (var i = 0; i < scores.length; i++) {
        if (!_allowedMask[i]) continue;
        allowedSum += scores[i];
        if (bestIdx == -1 || scores[i] > scores[bestIdx]) bestIdx = i;
      }
      // Confidence is renormalized within the allowed subset, not the raw
      // softmax value from the full 286-/107-way output. Real-device
      // testing found letters and digits essentially never committing at
      // all -- confidence stuck below even the 0.4 bar -- while words
      // committed normally. The cause: the model's softmax is computed
      // over every trained class at once, and masking restricts which
      // class can *win* the argmax but was leaving its *score* exactly as
      // reported for the full, unmasked distribution. Words are 250 of
      // 286 ASL classes (60 of 107 BdSL), so a winning word tends to
      // retain a substantial share of the total probability mass even
      // when the softmax is diffuse; letters (26) and digits (10) are a
      // small slice of the same space, so under real-world input --
      // noisier than the clean training photos this was tuned against --
      // the correct letter can be overwhelmingly the best answer *among
      // letters* while its raw share of all 286 classes still sits under
      // 0.4, because probability mass "leaked" to unrelated word classes
      // was being counted against it. Dividing by the sum of only the
      // allowed scores makes confidence reflect what the UI and category
      // masking already claim: competition restricted to the active mode,
      // not the whole model's output.
      final confidence = bestIdx == -1 || allowedSum <= 0 ? 0.0 : scores[bestIdx] / allowedSum;
      _latestResult = bestIdx == -1 ? null : ClassificationResult(_labels[bestIdx], confidence);
    } finally {
      _inferring = false;
    }
  }

  /// Builds one (42, 3) two-hand frame from up to 2 detected hands, sorted
  /// into slots by wrist x-coordinate -- mirrors
  /// ml/src/landmarks.py's sort_hands_by_x + combine_two_hands_by_x +
  /// normalize_two_hand_frame exactly. This plugin's [Hand] carries no
  /// left/right handedness at all, so geometric (on-screen position)
  /// ordering is the only slot-assignment rule that can be computed
  /// identically here and in training from the dataset's raw coordinates
  /// -- see landmarks.py's module docstring for why. A lone detected hand
  /// always lands in slot 0, with slot 1 zero-filled, matching the
  /// single-hand static-pose datasets (BdSL digits/letters, the ASL
  /// alphabet) trained the same way.
  List<List<double>> _buildTwoHandFrame(List<Hand> hands) {
    final sorted = [...hands]..sort((a, b) => a.landmarks[0].x.compareTo(b.landmarks[0].x));
    final slot0 = sorted[0].landmarks;
    final slot1 = sorted.length > 1 ? sorted[1].landmarks : null;

    // Shared origin/scale from whichever slot is present, preferring slot
    // 1 -- matches normalize_two_hand_frame's "prefer the right/later
    // slot" reference choice, so the two hands' *relative* position to
    // each other survives normalization instead of each being centered
    // independently.
    final reference = slot1 ?? slot0;
    final origin = reference[0];
    final mid = reference[9];
    final dx = mid.x - origin.x, dy = mid.y - origin.y, dz = mid.z - origin.z;
    final scale = math.sqrt(dx * dx + dy * dy + dz * dz);
    final invScale = scale < 1e-6 ? 1.0 : 1.0 / scale;

    List<double> normalize(Landmark l) =>
        [(l.x - origin.x) * invScale, (l.y - origin.y) * invScale, (l.z - origin.z) * invScale];

    return [
      ...slot0.map(normalize),
      if (slot1 != null) ...slot1.map(normalize) else ...List.generate(_numLandmarks, (_) => const [0.0, 0.0, 0.0]),
      // The position anchor: the raw in-frame origin and apparent hand
      // size that the normalization above divides out. Normalization is
      // what makes a handshape recognizable regardless of where it is
      // held -- and is therefore also what makes signs distinguished only
      // by *where* they are held (at the chin versus the cheek, say)
      // collapse into near-identical inputs. Measured cost: face-located
      // ASL words score 9.9 points worse than other words (see
      // ml/analyze_face_hypothesis.py). Keeping these three numbers hands
      // that information back at zero inference cost, because the plugin
      // already reports the wrist's frame position -- no face-landmark
      // model, and no train/inference mismatch.
      if (_expectsAnchor) [origin.x, origin.y, scale],
    ];
  }

  void dispose() {
    _isolateInterpreter.close();
    _interpreter.close();
  }
}
