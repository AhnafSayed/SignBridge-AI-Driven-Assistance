import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

import '../app_theme.dart';
import '../state/app_state.dart';
import '../widgets/brand_title.dart';

enum _MicStage { requestingPermission, permissionDenied, idle, listening, unavailable }

/// The language the *microphone* listens in. Deliberately separate from
/// [AppState.language] (which is the Sign tab's recognition/output
/// language) -- the person speaking into this tab isn't necessarily using
/// the same language as the impaired user is signing in, so this tab gets
/// its own switch rather than silently inheriting the Sign tab's choice.
enum _ListenLang { english, bangla }

/// The other half of the conversation. The Sign tab is the impaired user
/// signing *to* someone; this tab is for whoever they're talking to --
/// hand them the phone (or set it between you) and their speech becomes
/// large on-screen text the impaired user can read, live.
class ListenScreen extends StatefulWidget {
  const ListenScreen({super.key});

  @override
  State<ListenScreen> createState() => ListenScreenState();
}

/// Public so [RootShell] can call [pause] via a [GlobalKey] when this tab
/// is swiped/tapped away from, instead of destroying and recreating this
/// widget (see the matching note on `HomeScreenState`).
class ListenScreenState extends State<ListenScreen>
    with AutomaticKeepAliveClientMixin, WidgetsBindingObserver {
  @override
  bool get wantKeepAlive => true;

  final stt.SpeechToText _speech = stt.SpeechToText();
  _MicStage _stage = _MicStage.requestingPermission;
  String _finalText = '';
  String _partialText = '';

  // speech_to_text's locales() is known-unreliable on Android -- there is
  // no OS API that actually lists installed offline speech languages, so
  // this can wrongly report Bangla as absent even when the recognizer
  // supports it (github.com/csdcorp/speech_to_text#505). It's kept only to
  // drive the informational banner below; actual recognition always tries
  // 'bn-BD' directly regardless of what this found (see _toggleListening).
  String? _banglaLocaleId;
  bool _localesChecked = false;
  bool _started = false;

  // Defaults to whatever the Sign tab is set to, but the user can flip it
  // independently via the toggle in build() -- synced from AppState.language
  // exactly once, on the first build, via [_listenLangInitialized].
  _ListenLang _listenLang = _ListenLang.english;
  bool _listenLangInitialized = false;

  // Set once an actual listen attempt in Bangla comes back with real
  // recognized words -- proof positive that Bangla works on this device,
  // which permanently hides the "couldn't confirm" banner regardless of
  // what the unreliable locales() check says.
  bool _banglaConfirmedWorking = false;
  bool _bannerDismissed = false;

  /// Requests the microphone and initializes speech recognition, but only
  /// the first time this is called -- deliberately *not* wired to
  /// [initState], since this tab is built (kept alive in a [PageView])
  /// before the user has ever opened it. Asking for microphone permission
  /// at app launch, before the user has seen this tab at all, would be a
  /// confusing surprise. [RootShell] calls this the first time you swipe
  /// or tap to this tab.
  Future<void> ensureStarted() async {
    WidgetsBinding.instance.addObserver(this);
    if (_started) return;
    _started = true;
    await _bootstrap();
  }

  Future<void> _bootstrap() async {
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      if (mounted) setState(() => _stage = _MicStage.permissionDenied);
      return;
    }
    final available = await _speech.initialize(
      onStatus: _onStatus,
      onError: (e) => debugPrint('speech_to_text error: $e'),
    );
    if (available) await _checkBanglaLocale();
    if (mounted) {
      setState(() {
        _stage = available ? _MicStage.idle : _MicStage.unavailable;
        _localesChecked = true;
      });
    }
  }

  Future<void> _checkBanglaLocale() async {
    final locales = await _speech.locales();
    final bangla = locales.where((l) => l.localeId.toLowerCase().startsWith('bn')).toList();
    final resolved = bangla.isEmpty ? null : bangla.first.localeId;
    if (mounted) setState(() => _banglaLocaleId = resolved);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // If Bangla wasn't available, re-check whenever the user comes back to
    // the app -- most likely because they just went to Settings and
    // installed a Bangla voice, and the "not available" banner should
    // disappear on its own rather than needing a manual refresh.
    if (state == AppLifecycleState.resumed && _localesChecked && _banglaLocaleId == null) {
      _checkBanglaLocale();
    }
  }

  void _onStatus(String status) {
    if (!mounted) return;
    if (status == 'listening') {
      setState(() => _stage = _MicStage.listening);
    } else if (_stage == _MicStage.listening) {
      setState(() => _stage = _MicStage.idle);
    }
  }

  Future<void> _toggleListening() async {
    if (_stage == _MicStage.listening) {
      await _speech.stop();
      return;
    }

    // No popup here on purpose: the persistent banner in build() already
    // says this once, plainly, whenever it's actually relevant -- a
    // SnackBar on every single tap of the mic would be far more
    // disruptive than helpful.
    //
    // Always try 'bn-BD' directly for Bangla rather than gating on
    // _banglaLocaleId -- locales() misses installed languages on many
    // Android builds (see the field doc above), so trusting it here would
    // wrongly force English on phones that can actually recognize Bangla.
    final bangla = _listenLang == _ListenLang.bangla;
    final localeId = bangla ? 'bn-BD' : 'en_US';

    setState(() => _partialText = '');
    await _speech.listen(
      onResult: (result) {
        setState(() {
          if (bangla && result.recognizedWords.isNotEmpty) {
            _banglaConfirmedWorking = true;
          }
          if (result.finalResult) {
            final words = result.recognizedWords;
            if (words.isNotEmpty) {
              _finalText = _finalText.isEmpty ? words : '$_finalText $words';
            }
            _partialText = '';
          } else {
            _partialText = result.recognizedWords;
          }
        });
      },
      listenOptions: stt.SpeechListenOptions(
        partialResults: true,
        listenMode: stt.ListenMode.dictation,
        localeId: localeId,
        pauseFor: const Duration(seconds: 4),
        listenFor: const Duration(minutes: 2),
      ),
    );
  }

  void _clear() => setState(() {
        _finalText = '';
        _partialText = '';
      });

  /// Stops any in-progress listening. Called when this tab is swiped/tapped
  /// away from, so the microphone doesn't keep recording behind a screen
  /// you're no longer looking at.
  Future<void> pause() async {
    if (_stage == _MicStage.listening) await _speech.stop();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _speech.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    super.build(context); // required by AutomaticKeepAliveClientMixin
    final appState = context.watch<AppState>();
    if (!_listenLangInitialized) {
      _listenLangInitialized = true;
      _listenLang = appState.language == SignLanguage.bdsl ? _ListenLang.bangla : _ListenLang.english;
    }
    final showBanglaBanner = _listenLang == _ListenLang.bangla &&
        _localesChecked &&
        _banglaLocaleId == null &&
        !_banglaConfirmedWorking &&
        !_bannerDismissed;
    return Scaffold(
      appBar: AppBar(title: const BrandTitle('Listen')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
          child: Column(
            children: [
              const Text(
                "Hand the phone to whoever's speaking, or set it between you.",
                textAlign: TextAlign.center,
                style: TextStyle(color: AppColors.inkSoft),
              ),
              const SizedBox(height: 10),
              _ListenLangBar(
                selected: _listenLang,
                onChanged: (lang) => setState(() => _listenLang = lang),
              ),
              if (showBanglaBanner)
                Container(
                  margin: const EdgeInsets.only(top: 10),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: AppColors.amber.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: AppColors.amber.withValues(alpha: 0.4)),
                  ),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          const Expanded(
                            child: Text(
                              "Couldn't confirm a Bangla voice is installed -- the mic will "
                              "still try Bangla first when you tap it.",
                              textAlign: TextAlign.center,
                              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                            ),
                          ),
                          InkWell(
                            onTap: () => setState(() => _bannerDismissed = true),
                            child: const Padding(
                              padding: EdgeInsets.all(4),
                              child: Icon(Icons.close, size: 16, color: AppColors.inkSoft),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      const Text(
                        "If it comes back empty or in English instead: Settings -> System -> "
                        "Languages & input -> Voice input (or On-device speech recognition) "
                        "-> add Bangla. This banner disappears on its own once Bangla actually "
                        "works or you dismiss it.",
                        textAlign: TextAlign.center,
                        style: TextStyle(fontSize: 12, color: AppColors.inkSoft),
                      ),
                      TextButton(
                        onPressed: _checkBanglaLocale,
                        child: const Text('Check again', style: TextStyle(fontSize: 12)),
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: 16),
              Expanded(child: _buildBody(appState)),
              const SizedBox(height: 16),
              _buildMicArea(appState),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBody(AppState appState) {
    switch (_stage) {
      case _MicStage.requestingPermission:
        return const Center(child: CircularProgressIndicator());
      case _MicStage.permissionDenied:
        return _MicMessage(
          icon: Icons.mic_off_outlined,
          title: 'Microphone access needed',
          message: 'SignBridge needs the microphone to caption what the other person says.',
          action: FilledButton(onPressed: openAppSettings, child: const Text('Open settings')),
        );
      case _MicStage.unavailable:
        return const _MicMessage(
          icon: Icons.error_outline,
          title: 'Speech recognition unavailable',
          message: 'This device does not support on-device speech recognition.',
        );
      case _MicStage.idle:
      case _MicStage.listening:
        final display = [_finalText, _partialText].where((s) => s.isNotEmpty).join(' ');
        return Container(
          width: double.infinity,
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(16),
          ),
          child: SingleChildScrollView(
            reverse: true,
            child: Text(
              display.isEmpty ? 'Tap the microphone and start talking…' : display,
              style: TextStyle(
                fontSize: 26,
                fontWeight: FontWeight.w600,
                color: display.isEmpty ? AppColors.inkSoft : null,
              ),
            ),
          ),
        );
    }
  }

  Widget _buildMicArea(AppState appState) {
    final listening = _stage == _MicStage.listening;
    final enabled = _stage == _MicStage.idle || listening;
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _MicButton(
              listening: listening,
              enabled: enabled,
              onPressed: enabled ? _toggleListening : null,
            ),
          ],
        ),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          decoration: BoxDecoration(
            color: (listening ? AppColors.coral : AppColors.primary).withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            listening ? 'Listening…' : 'Tap to talk',
            style: TextStyle(
              color: listening ? AppColors.coral : AppColors.primary,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        const SizedBox(height: 8),
        TextButton.icon(
          onPressed: _finalText.isEmpty && _partialText.isEmpty ? null : _clear,
          icon: const Icon(Icons.clear, size: 18),
          label: const Text('Clear'),
        ),
      ],
    );
  }
}

class _ListenLangBar extends StatelessWidget {
  const _ListenLangBar({required this.selected, required this.onChanged});
  final _ListenLang selected;
  final ValueChanged<_ListenLang> onChanged;

  @override
  Widget build(BuildContext context) {
    return SegmentedButton<_ListenLang>(
      segments: const [
        ButtonSegment(value: _ListenLang.english, label: Text('English')),
        ButtonSegment(value: _ListenLang.bangla, label: Text('বাংলা')),
      ],
      selected: {selected},
      onSelectionChanged: (s) => onChanged(s.first),
    );
  }
}

class _MicButton extends StatelessWidget {
  const _MicButton({required this.listening, required this.enabled, required this.onPressed});
  final bool listening;
  final bool enabled;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: listening ? AppColors.coral : AppColors.primary,
      shape: const CircleBorder(),
      elevation: 3,
      child: InkWell(
        onTap: onPressed,
        customBorder: const CircleBorder(),
        child: Padding(
          padding: const EdgeInsets.all(22),
          child: Icon(
            listening ? Icons.stop_rounded : Icons.mic_rounded,
            size: 34,
            color: Colors.white,
          ),
        ),
      ),
    );
  }
}

class _MicMessage extends StatelessWidget {
  const _MicMessage({required this.icon, required this.title, required this.message, this.action});
  final IconData icon;
  final String title;
  final String message;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 44, color: AppColors.inkSoft),
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
