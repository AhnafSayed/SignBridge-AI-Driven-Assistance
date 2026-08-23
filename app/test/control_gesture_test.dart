import 'package:flutter_test/flutter_test.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:sign_bridge/services/control_gesture.dart';

/// Verifies the hands-free control gestures (backspace/clearAll/speak)
/// against both real and synthetic landmark data.
///
/// The two closed-fist cases below are copied verbatim from this
/// project's own extracted training data (`ml/data/asl_alphabet_landmarks
/// .npz`'s letter "S", and `ml/data/asl_digit_landmarks.npz`'s digit "0"),
/// not invented -- they're the concrete evidence behind a real bug found
/// and fixed in `control_gesture.dart`: the previous thumb-extension
/// check (`dist(thumb_tip, wrist) > dist(index_mcp, pinky_mcp) * 0.9`)
/// came out *true* for essentially every real closed-fist hand shape
/// (checked directly against this same data), which made backspace
/// effectively unreachable -- any closed fist fell through to clearAll or
/// speak depending on incidental thumb position, never backspace. Both
/// real samples below are regression tests for that exact fix.
void main() {
  final detector = ControlGestureDetector();

  Hand hand(List<List<double>> coords) =>
      Hand(coords.map((c) => Landmark(c[0], c[1], c[2])).toList());

  test('empty hands -> none', () {
    expect(detector.detect([]), ControlGesture.none);
  });

  test('real letter "S" landmarks (closed fist, thumb across front) -> backspace', () {
    final s = hand([
      [0.3773, 0.6722, -0.0000],
      [0.5080, 0.5818, -0.0512],
      [0.5647, 0.4364, -0.0646],
      [0.4995, 0.3345, -0.0751],
      [0.3734, 0.3471, -0.0715],
      [0.5451, 0.3640, 0.0052],
      [0.5029, 0.2496, -0.0497],
      [0.4848, 0.3466, -0.0796],
      [0.5001, 0.3994, -0.0935],
      [0.4533, 0.3719, 0.0022],
      [0.4004, 0.2741, -0.0498],
      [0.4073, 0.3823, -0.0624],
      [0.4305, 0.4025, -0.0602],
      [0.3667, 0.3883, -0.0158],
      [0.3282, 0.3281, -0.0779],
      [0.3481, 0.4356, -0.0786],
      [0.3804, 0.4493, -0.0623],
      [0.2875, 0.4174, -0.0393],
      [0.2774, 0.3949, -0.0642],
      [0.3002, 0.4618, -0.0528],
      [0.3107, 0.4737, -0.0381],
    ]);
    expect(detector.detect([s]), ControlGesture.backspace);
  });

  test('real ASL digit "0" landmarks (closed circle/O shape) -> backspace', () {
    final zero = hand([
      [0.4072, 0.3080, 0.0000],
      [0.5182, 0.3817, 0.0049],
      [0.6479, 0.3769, 0.0078],
      [0.7695, 0.3990, -0.0006],
      [0.8667, 0.3957, -0.0031],
      [0.7335, 0.1594, 0.0729],
      [0.8210, 0.2687, 0.0525],
      [0.8318, 0.3655, 0.0303],
      [0.8302, 0.4375, 0.0148],
      [0.7440, 0.1262, 0.0528],
      [0.8521, 0.2551, 0.0168],
      [0.8509, 0.3619, -0.0092],
      [0.8399, 0.4436, -0.0210],
      [0.7461, 0.1292, 0.0239],
      [0.8676, 0.2530, -0.0197],
      [0.8664, 0.3695, -0.0383],
      [0.8413, 0.4500, -0.0409],
      [0.7404, 0.1663, -0.0059],
      [0.8380, 0.2790, -0.0403],
      [0.8482, 0.3703, -0.0553],
      [0.8423, 0.4426, -0.0600],
    ]);
    expect(detector.detect([zero]), ControlGesture.backspace);
  });

  // The control gestures run identically for BdSL -- ControlGestureDetector
  // and ControlArbiter are language-agnostic geometry over raw landmarks,
  // with no ASL/BdSL branch anywhere in either class. A full sweep of
  // BdSL's own static digit/letter CSV landmarks (ml/src/dataset_bdsl_static
  // .py's loader) found the same collision, wider if anything: 15 of 37
  // letters and 2 of 10 digits exceed a 30% backspace or 15% clear/speak
  // rate, topped by letter "Taa" at 99% backspace and digit "6" at 31%
  // backspace *and* 30% clear/speak (clearAll is the more damaging
  // failure of the two -- it erases the whole sentence, not just the last
  // word). Both real samples below are drawn from that measurement, not
  // invented, mirroring the ASL regression tests above.
  test('real BdSL letter "Taa" landmarks (closed fist) -> backspace', () {
    final taa = hand([
      [0.4872, 0.7416, -0.0001],
      [0.3859, 0.6682, -0.0914],
      [0.3564, 0.5209, -0.1327],
      [0.4213, 0.4321, -0.1581],
      [0.4962, 0.4058, -0.1388],
      [0.4346, 0.3796, -0.0563],
      [0.4500, 0.2464, -0.1604],
      [0.4329, 0.3706, -0.2006],
      [0.4270, 0.4545, -0.2074],
      [0.5041, 0.3980, -0.0577],
      [0.5240, 0.2551, -0.1996],
      [0.4841, 0.4036, -0.2190],
      [0.4740, 0.4746, -0.1834],
      [0.5678, 0.4340, -0.0772],
      [0.5919, 0.3095, -0.1975],
      [0.5376, 0.4511, -0.1934],
      [0.5245, 0.5220, -0.1462],
      [0.6259, 0.4866, -0.1065],
      [0.6352, 0.3989, -0.1787],
      [0.5898, 0.4832, -0.1846],
      [0.5774, 0.5321, -0.1600],
    ]);
    expect(detector.detect([taa]), ControlGesture.backspace);
  });

  test('real BdSL digit "6" landmarks -> speak (the clear/speak collision case)', () {
    final six = hand([
      [0.3846, 0.6771, 0.0000],
      [0.3895, 0.5163, -0.0503],
      [0.4679, 0.3798, -0.0635],
      [0.5364, 0.2741, -0.0777],
      [0.5723, 0.1711, -0.1068],
      [0.5719, 0.4490, -0.0092],
      [0.6384, 0.4187, -0.1162],
      [0.5683, 0.4568, -0.1897],
      [0.4987, 0.4863, -0.2084],
      [0.5958, 0.5304, -0.0163],
      [0.6571, 0.4987, -0.1372],
      [0.5666, 0.5238, -0.1984],
      [0.4949, 0.5465, -0.1934],
      [0.6040, 0.6158, -0.0370],
      [0.6597, 0.5922, -0.1474],
      [0.5693, 0.6018, -0.1897],
      [0.4962, 0.6111, -0.1765],
      [0.5998, 0.7034, -0.0556],
      [0.6334, 0.6818, -0.1326],
      [0.5690, 0.6719, -0.1498],
      [0.5110, 0.6715, -0.1292],
    ]);
    // Not backspace: this specific real sample's thumb geometry reads as
    // the vertical-thumb shape, so the raw detector's answer is "speak" --
    // still a real collision (a genuine digit-6 attempt would show up as
    // an unrequested speak-out-loud), just a different one of the three
    // than the letter case above. The point demonstrated is the same: a
    // real trained handshape can raw-collide with a control gesture at
    // all, in either direction, for BdSL exactly as for ASL.
    expect(detector.detect([six]), ControlGesture.speak);
  });

  // Synthetic from here on -- deliberately exaggerated poses (thumb
  // clearly separated from a curled fist, pointing straight up or down)
  // representing what this project asks the user to actually hold, not a
  // borderline case. Proportioned to a similar hand scale (~0.2-0.3
  // wrist-to-middle-knuckle) as the real samples above.
  List<List<double>> curledFistWithThumb(double thumbTipY) => [
        [0.50, 0.70, 0.0], // 0 wrist
        [0.45, 0.62, 0.0], // 1 thumb_cmc
        [0.43, 0.55, 0.0], // 2 thumb_mcp
        [0.42, (0.55 + thumbTipY) / 2, 0.0], // 3 thumb_ip (between mcp and tip)
        [0.41, thumbTipY, 0.0], // 4 thumb_tip
        [0.55, 0.50, 0.0], // 5 index_mcp
        [0.55, 0.42, 0.0], // 6 index_pip
        [0.55, 0.44, 0.0], // 7 index_dip
        [0.55, 0.46, 0.0], // 8 index_tip (curled: below pip)
        [0.52, 0.48, 0.0], // 9 middle_mcp
        [0.60, 0.40, 0.0], // 10 middle_pip
        [0.60, 0.42, 0.0], // 11 middle_dip
        [0.60, 0.44, 0.0], // 12 middle_tip (curled)
        [0.63, 0.50, 0.0], // 13 ring_mcp
        [0.65, 0.42, 0.0], // 14 ring_pip
        [0.65, 0.44, 0.0], // 15 ring_dip
        [0.65, 0.46, 0.0], // 16 ring_tip (curled)
        [0.68, 0.53, 0.0], // 17 pinky_mcp
        [0.70, 0.46, 0.0], // 18 pinky_pip
        [0.70, 0.48, 0.0], // 19 pinky_dip
        [0.70, 0.50, 0.0], // 20 pinky_tip (curled)
      ];

  test('closed fist, thumb extended straight up -> speak', () {
    final thumbsUp = hand(curledFistWithThumb(0.30)); // tip well above mcp (0.55)
    expect(detector.detect([thumbsUp]), ControlGesture.speak);
  });

  test('closed fist, thumb extended straight down -> clearAll', () {
    final thumbsDown = hand(curledFistWithThumb(0.80)); // tip well below mcp (0.55)
    expect(detector.detect([thumbsDown]), ControlGesture.clearAll);
  });

  test('open hand (index finger extended) -> none, regardless of thumb', () {
    final coords = curledFistWithThumb(0.30); // same thumbs-up-shaped thumb...
    coords[8] = [0.55, 0.20, 0.0]; // ...but index finger clearly extended
    expect(detector.detect([hand(coords)]), ControlGesture.none);
  });

  test('degenerate/zero-scale landmarks -> backspace, not a crash', () {
    final degenerate = hand(List.generate(21, (_) => [0.5, 0.5, 0.0]));
    expect(detector.detect([degenerate]), ControlGesture.backspace);
  });

  controlArbiterTests();
}

/// Covers ControlArbiter: the hysteresis that stops a closed-fist letter
/// (see the sweep in control_gesture.dart's docstring -- S 100%, E/Q 99%,
/// M 91%, A 88%, T 77%, O 70%, P 55% of sampled poses read as `backspace`)
/// from having backspace fire on nothing more than a brief real-world
/// confidence dip mid-sign. Every case below uses fabricated `DateTime`s
/// rather than a real clock, so the 500ms window is tested exactly rather
/// than approximately.
void controlArbiterTests() {
  group('ControlArbiter', () {
    final base = DateTime(2026, 1, 1, 12, 0, 0);

    test('confident frame always suppresses control, regardless of raw reading', () {
      final arbiter = ControlArbiter();
      final result = arbiter.arbitrate(
        wordIsConfident: true,
        rawControl: ControlGesture.backspace,
        now: base,
      );
      expect(result, ControlGesture.none);
    });

    test('non-confident frame with no recent history passes the raw reading through', () {
      final arbiter = ControlArbiter();
      final result = arbiter.arbitrate(
        wordIsConfident: false,
        rawControl: ControlGesture.backspace,
        now: base,
      );
      expect(result, ControlGesture.backspace);
    });

    test('a confidence dip shortly after a confident frame is still suppressed', () {
      final arbiter = ControlArbiter();
      arbiter.arbitrate(wordIsConfident: true, rawControl: ControlGesture.none, now: base);
      // 300ms later, confidence dipped (e.g. a real "S" momentarily below
      // the override bar), but well inside the 500ms cooldown.
      final result = arbiter.arbitrate(
        wordIsConfident: false,
        rawControl: ControlGesture.backspace,
        now: base.add(const Duration(milliseconds: 300)),
      );
      expect(result, ControlGesture.none,
          reason: 'a brief post-confident dip must not let backspace through');
    });

    test('backspace fires again once the cooldown has genuinely elapsed', () {
      final arbiter = ControlArbiter();
      arbiter.arbitrate(wordIsConfident: true, rawControl: ControlGesture.none, now: base);
      // 600ms later -- past the 500ms window -- a real held fist should
      // not be blocked forever by one earlier confident frame.
      final result = arbiter.arbitrate(
        wordIsConfident: false,
        rawControl: ControlGesture.backspace,
        now: base.add(const Duration(milliseconds: 600)),
      );
      expect(result, ControlGesture.backspace);
    });

    test('a custom cooldown is honoured', () {
      final arbiter = ControlArbiter(suppressCooldown: const Duration(milliseconds: 100));
      arbiter.arbitrate(wordIsConfident: true, rawControl: ControlGesture.none, now: base);
      final result = arbiter.arbitrate(
        wordIsConfident: false,
        rawControl: ControlGesture.backspace,
        now: base.add(const Duration(milliseconds: 200)),
      );
      expect(result, ControlGesture.backspace);
    });

    test('reset clears history, so a stale confident timestamp cannot leak across it', () {
      final arbiter = ControlArbiter();
      arbiter.arbitrate(wordIsConfident: true, rawControl: ControlGesture.none, now: base);
      arbiter.reset();
      final result = arbiter.arbitrate(
        wordIsConfident: false,
        rawControl: ControlGesture.clearAll,
        now: base.add(const Duration(milliseconds: 10)),
      );
      expect(result, ControlGesture.clearAll,
          reason: 'reset should mean no recent-confidence memory remains');
    });

    test('a real intentionally-held fist still fires backspace once its own hold completes', () {
      // Simulates: user signs a confident letter, then immediately closes
      // a genuine backspace fist and holds it continuously. Frames arrive
      // roughly every 33ms (~30fps); the fist reads as backspace on every
      // one, non-confident throughout.
      final arbiter = ControlArbiter();
      arbiter.arbitrate(wordIsConfident: true, rawControl: ControlGesture.none, now: base);

      ControlGesture last = ControlGesture.none;
      for (var ms = 33; ms <= 1200; ms += 33) {
        last = arbiter.arbitrate(
          wordIsConfident: false,
          rawControl: ControlGesture.backspace,
          now: base.add(Duration(milliseconds: ms)),
        );
      }
      expect(last, ControlGesture.backspace,
          reason: 'a fist held well past the cooldown must not be permanently suppressed');
    });
  });
}
