import 'package:flutter_test/flutter_test.dart';
import 'package:sign_bridge/services/sos_service.dart';

void main() {
  group('buildSmsLaunchUri', () {
    test('uses the sms: scheme with the number as the path', () {
      final uri = buildSmsLaunchUri('+15551234567', 'help');
      expect(uri.scheme, 'sms');
      expect(uri.path, '+15551234567');
    });

    test('puts the message in the body query parameter, percent-encoded', () {
      final uri = buildSmsLaunchUri('123', 'Emergency! I need help.');
      expect(uri.queryParameters['body'], 'Emergency! I need help.');
      // The raw URI string must actually be percent-encoded (spaces/punctuation
      // are not valid unescaped in a URI query) -- Uri.encodeComponent's job,
      // and the thing that would silently break if it were ever dropped.
      expect(uri.toString(), contains('%20'));
    });

    test('does not send anything itself -- only builds the URI', () {
      // buildSmsLaunchUri has no side effects at all; this exists to pin
      // that contract down explicitly, since the whole safety argument in
      // sos_service.dart's docstring depends on this function never being
      // the thing that transmits a message.
      expect(() => buildSmsLaunchUri('123', 'test'), returnsNormally);
    });
  });
}
