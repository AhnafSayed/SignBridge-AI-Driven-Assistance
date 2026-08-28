import 'package:flutter_test/flutter_test.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:sign_bridge/services/hand_selection.dart';

/// Verifies `primaryHand`, the single-hand-only guard shared by
/// DirectStaticClassifier (ASL letters/digits), TfliteSignClassifier
/// (BdSL letters/digits), ControlGestureDetector (hands-free gestures),
/// and HomeScreenState's own upstream filter, so a second hand in frame --
/// typically the one holding the phone -- doesn't corrupt recognition or
/// gesture detection for anything that was only ever trained/designed
/// around one hand. See hand_selection.dart's own doc for why bounding-box
/// size, not detection order, is the tiebreaker.
void main() {
  // A square hand bounding box of the given half-width centred at the
  // origin -- real landmark shape doesn't matter for this function, only
  // the overall bounding box does.
  Hand squareHand(double halfSize, {double centerX = 0.5, double centerY = 0.5}) {
    return Hand([
      Landmark(centerX - halfSize, centerY - halfSize, 0),
      Landmark(centerX + halfSize, centerY - halfSize, 0),
      Landmark(centerX + halfSize, centerY + halfSize, 0),
      Landmark(centerX - halfSize, centerY + halfSize, 0),
    ]);
  }

  test('no hands -> null', () {
    expect(primaryHand([]), isNull);
  });

  test('one hand -> that hand, regardless of size', () {
    final only = squareHand(0.05);
    expect(primaryHand([only]), same(only));
  });

  test('two hands -> the larger bounding box wins, not list order', () {
    final small = squareHand(0.05);
    final large = squareHand(0.2);
    expect(primaryHand([small, large]), same(large));
    // Order in the input list must not matter -- this is what stops
    // recognition from flickering between frames where the plugin happens
    // to report the same two hands in a different order.
    expect(primaryHand([large, small]), same(large));
  });

  test('three hands -> the single largest, not just the first or last', () {
    final small = squareHand(0.03);
    final medium = squareHand(0.1);
    final large = squareHand(0.25);
    expect(primaryHand([small, large, medium]), same(large));
  });

  test('equal-size hands -> stable pick (the first one), not arbitrary', () {
    final a = squareHand(0.1);
    final b = squareHand(0.1);
    expect(primaryHand([a, b]), same(a));
  });
}
