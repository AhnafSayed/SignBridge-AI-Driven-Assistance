import 'dart:async';

import 'package:flutter/services.dart' show HapticFeedback;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:url_launcher/url_launcher.dart';

/// The alarm half of the emergency feature: a loud, repeating spoken
/// alert plus a vibration pulse, until explicitly stopped.
///
/// Deliberately does not send anything on its own -- see
/// [buildSmsLaunchUri], [buildWhatsAppLaunchUri], and [buildCallLaunchUri]
/// for why reaching the contact is always left to a pre-filled native
/// intent (text, WhatsApp, or the dialer) the user still has to tap
/// "send" or "call" on themselves, rather than a silently automatic send
/// or an automatically placed call.
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

/// Builds a WhatsApp "click to chat" link with the message pre-filled but
/// NOT sent -- same reasoning as [buildSmsLaunchUri]: this hands off to
/// WhatsApp's own compose screen, where sending is still one deliberate
/// tap away, never an app-triggered side effect.
///
/// WhatsApp's click-to-chat spec wants the number as plain digits, country
/// code included, with no `+`, spaces, dashes, or parentheses -- unlike
/// `sms:`/`tel:`, which both accept a normally-formatted number. Stripping
/// everything but digits here means the one number already stored for the
/// emergency contact (in whatever format the user typed it, or the
/// contact picker returned it) works for every launcher, without asking
/// the user to store it twice in two formats.
Uri buildWhatsAppLaunchUri(String phoneNumber, String message) {
  final digitsOnly = phoneNumber.replaceAll(RegExp(r'[^0-9]'), '');
  final encodedText = Uri.encodeComponent(message);
  return Uri.parse('https://wa.me/$digitsOnly?text=$encodedText');
}

Future<bool> launchPrefilledWhatsApp(String phoneNumber, String message) async {
  final uri = buildWhatsAppLaunchUri(phoneNumber, message);
  // externalApplication, not the default: an https:// link would otherwise
  // just as happily open in an in-app browser tab showing wa.me's own
  // redirect page instead of handing off to the WhatsApp app itself.
  return launchUrl(uri, mode: LaunchMode.externalApplication);
}

/// Builds a `tel:` URI that opens the phone's own dialer with the number
/// already entered -- it does NOT place the call itself. Placing a call
/// automatically (Android's `CALL_PHONE` permission plus `ACTION_CALL`)
/// was deliberately not used, for the same reason [buildSmsLaunchUri]
/// hands off to the messaging app instead of transmitting directly: a
/// false trigger must never be able to silently start a real phone call
/// with no chance to cancel, and `tel:` (which resolves to `ACTION_DIAL`)
/// has never required that permission at all, so this costs the feature
/// nothing extra to ask the user for.
Uri buildCallLaunchUri(String phoneNumber) => Uri.parse('tel:$phoneNumber');

Future<bool> launchCall(String phoneNumber) async {
  final uri = buildCallLaunchUri(phoneNumber);
  return launchUrl(uri);
}
