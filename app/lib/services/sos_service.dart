import 'dart:async';

import 'package:flutter/services.dart' show HapticFeedback;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:url_launcher/url_launcher.dart';

/// The alarm half of the emergency feature: a loud, repeating spoken
/// alert plus a vibration pulse, until explicitly stopped.
///
/// Deliberately does not send anything on its own -- see
/// [buildSmsLaunchUri]'s own comment for why sending is left to a
/// pre-filled native SMS intent the user still has to tap "send" on,
/// rather than a silently automatic send.
///
/// Uses its own [FlutterTts] instance, separate from the one HomeScreen
/// uses for normal sentence-speaking, so an SOS alert can't be silently
/// cancelled by (or itself cancel) an unrelated "speak the sentence" call
/// racing it on a shared engine handle.
class SosService {
  final FlutterTts _tts = FlutterTts();
  Timer? _loop;
  bool _active = false;

  bool get isActive => _active;

  Future<void> start({String message = 'Emergency. I need help. Please assist me.'}) async {
    if (_active) return;
    _active = true;
    await _tts.setVolume(1.0);
    await _tts.setSpeechRate(0.45);
    unawaited(_tts.speak(message));
    HapticFeedback.heavyImpact();
    // Repeats the alert rather than speaking it once -- the point of a
    // siren is that it keeps going until someone deals with it, not that
    // it announces itself a single time and falls silent.
    _loop = Timer.periodic(const Duration(seconds: 4), (_) {
      if (!_active) return;
      unawaited(_tts.speak(message));
      HapticFeedback.heavyImpact();
    });
  }

  Future<void> stop() async {
    _active = false;
    _loop?.cancel();
    _loop = null;
    await _tts.stop();
  }

  void dispose() {
    _loop?.cancel();
    _tts.stop();
  }
}

/// Builds an `sms:` URI with the message pre-filled but NOT sent. Opening
/// it hands control to the phone's own messaging app, where the user
/// still has to tap send themselves.
///
/// This is a deliberate safety choice, not a limitation worked around.
/// Automatically transmitting a message with no further confirmation --
/// the pattern a native SEND_SMS plugin would enable -- means a false
/// trigger (an accidental tap, a confused first-time user, a phone
/// bumped in a pocket) silently sends a real distress message to a real
/// person with no chance to cancel. Requiring one more tap, in the
/// app the user already trusts to send texts, costs very little time in
/// a genuine emergency and removes that entire failure mode. It also
/// means this feature needs no SEND_SMS permission at all.
Uri buildSmsLaunchUri(String phoneNumber, String message) {
  final encodedBody = Uri.encodeComponent(message);
  // "sms:" (not "smsto:") is the scheme url_launcher's own docs recommend
  // checking first; Android resolves either to the default messaging app.
  return Uri.parse('sms:$phoneNumber?body=$encodedBody');
}

Future<bool> launchPrefilledSms(String phoneNumber, String message) async {
  final uri = buildSmsLaunchUri(phoneNumber, message);
  return launchUrl(uri);
}
