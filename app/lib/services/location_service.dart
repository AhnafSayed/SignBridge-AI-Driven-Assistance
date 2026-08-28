import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';

/// Fetches a one-shot Google Maps link for the device's current location,
/// for attaching to an SOS message -- or `null` if that isn't possible
/// right now, for any reason at all (permission denied, location services
/// off, no fix within [timeout], no GPS hardware). Every one of those
/// cases is handled the same way: fall back to sending the emergency
/// message without a location rather than blocking the alert, or crashing
/// it, over something that was never the point of the feature. A message
/// that arrives late or without a location is far better than one that
/// never arrives because a permission dialog was dismissed.
///
/// Deliberately a single on-demand lookup, not a subscription: nothing in
/// this app tracks location in the background or between SOS taps, so
/// there is nothing here beyond what firing the alert itself asks for.
Future<String?> currentLocationLinkOrNull({Duration timeout = const Duration(seconds: 6)}) async {
  try {
    // permission_handler, the same package/pattern this app already uses
    // for camera/microphone/contacts, rather than geolocator's own
    // permission API -- one way of asking for a permission across the
    // whole app, not two.
    final status = await Permission.locationWhenInUse.request();
    if (!status.isGranted) return null;

    if (!await Geolocator.isLocationServiceEnabled()) return null;

    final position = await Geolocator.getCurrentPosition(
      locationSettings: LocationSettings(accuracy: LocationAccuracy.medium, timeLimit: timeout),
    );
    return 'https://maps.google.com/?q=${position.latitude},${position.longitude}';
  } catch (_) {
    // Covers a timeout, a denied-forever permission, location services
    // toggled off mid-request, or any platform-specific failure --
    // all the same outcome from the caller's side: no location this time.
    return null;
  }
}

/// Appends [locationLink] to [message] as a new line, or returns [message]
/// unchanged if no location was available. Pulled out as a pure function,
/// separate from the async lookup above, so the message-composition logic
/// itself is unit-testable without a device, a permission prompt, or a
/// GPS fix.
String appendLocationLink(String message, String? locationLink) {
  if (locationLink == null || locationLink.isEmpty) return message;
  return '$message\nMy location: $locationLink';
}
