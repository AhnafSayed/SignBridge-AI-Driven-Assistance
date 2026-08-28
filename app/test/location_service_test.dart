import 'package:flutter_test/flutter_test.dart';
import 'package:sign_bridge/services/location_service.dart';

/// Covers `appendLocationLink`, the pure half of the SOS location feature
/// -- the actual GPS fetch (`currentLocationLinkOrNull`) needs a device
/// and a permission prompt, so it isn't unit-tested directly, but the
/// message-composition logic it feeds into is, same as this project's
/// existing pattern of pulling the untestable I/O apart from the testable
/// decision it feeds (see GestureSegmenter, ControlArbiter).
void main() {
  group('appendLocationLink', () {
    test('no location -> message unchanged', () {
      expect(appendLocationLink('Emergency. I need help.', null), 'Emergency. I need help.');
    });

    test('empty location string -> message unchanged', () {
      // Defensive: currentLocationLinkOrNull never actually returns '',
      // only null or a real link, but the function's own contract should
      // not depend on that -- an empty string is exactly as "no location"
      // as null is.
      expect(appendLocationLink('Emergency. I need help.', ''), 'Emergency. I need help.');
    });

    test('a real link is appended on a new line', () {
      final result = appendLocationLink(
        'Emergency. I need help.',
        'https://maps.google.com/?q=23.777,90.399',
      );
      expect(result, 'Emergency. I need help.\nMy location: https://maps.google.com/?q=23.777,90.399');
    });
  });
}
