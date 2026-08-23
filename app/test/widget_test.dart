// AppState is pure Dart with no platform plugins, so it's the one thing in
// this project that's safe to unit test without mocking camera/permission
// channels. Full-screen widget tests would need those channels mocked.
import 'package:flutter_test/flutter_test.dart';
import 'package:sign_bridge/state/app_state.dart';

void main() {
  test('language defaults to ASL and can be switched', () {
    final state = AppState();
    expect(state.language, SignLanguage.asl);

    state.setLanguage(SignLanguage.bdsl);

    expect(state.language, SignLanguage.bdsl);
  });

  test('autoAssist defaults on and can be toggled', () {
    final state = AppState();
    expect(state.autoAssist, isTrue);

    state.toggleAutoAssist();

    expect(state.autoAssist, isFalse);
  });

  test('controlGesturesEnabled defaults on and can be toggled independently of autoAssist', () {
    final state = AppState();
    expect(state.controlGesturesEnabled, isTrue);

    state.toggleControlGestures();

    expect(state.controlGesturesEnabled, isFalse);
    expect(state.autoAssist, isTrue); // unaffected
  });

  test('sentence building: append, backspace, clear', () {
    final state = AppState();
    state.appendWord('3');
    state.appendWord('5');
    expect(state.sentenceText, '3 5');

    state.backspace();
    expect(state.sentenceText, '3');

    state.clearSentence();
    expect(state.sentence, isEmpty);
  });
}
