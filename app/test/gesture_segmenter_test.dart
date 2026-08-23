import 'package:flutter_test/flutter_test.dart';
import 'package:sign_bridge/services/gesture_segmenter.dart';

/// One landmark frame. Only x/y matter to the segmenter, so a compact
/// stand-in for a real 42- or 43-row frame is sufficient and keeps the
/// expected motion values easy to reason about by hand.
List<List<double>> frameAt(double x, double y, {int rows = 4}) =>
    List.generate(rows, (i) => [x + i * 0.001, y + i * 0.001, 0.0]);

void main() {
  const target = 40;

  group('resample', () {
    test('a sequence already at the target length is unchanged', () {
      final seq = List.generate(target, (i) => frameAt(i * 0.01, 0));
      final out = GestureSegmenter.resample(seq, target);
      expect(out.length, target);
      expect(out.first[0][0], closeTo(seq.first[0][0], 1e-9));
      expect(out.last[0][0], closeTo(seq.last[0][0], 1e-9));
    });

    test('a short sequence is stretched to the target length', () {
      final seq = [frameAt(0, 0), frameAt(1, 0)];
      final out = GestureSegmenter.resample(seq, target);
      expect(out.length, target);
      // Endpoints are preserved and the middle is interpolated.
      expect(out.first[0][0], closeTo(0.0, 1e-9));
      expect(out.last[0][0], closeTo(1.0, 1e-9));
      expect(out[target ~/ 2][0][0], closeTo(0.5, 0.02));
    });

    test('a long sequence is compressed to the target length', () {
      final seq = List.generate(120, (i) => frameAt(i / 119.0, 0));
      final out = GestureSegmenter.resample(seq, target);
      expect(out.length, target);
      expect(out.first[0][0], closeTo(0.0, 1e-9));
      expect(out.last[0][0], closeTo(1.0, 1e-9));
    });

    test('a single frame is repeated, matching the training-side rule', () {
      final out = GestureSegmenter.resample([frameAt(0.3, 0.4)], target);
      expect(out.length, target);
      expect(out.every((f) => (f[0][0] - 0.3).abs() < 1e-9), isTrue);
    });

    test('an empty sequence yields empty rather than throwing', () {
      expect(GestureSegmenter.resample(const [], target), isEmpty);
    });

    test('frame shape is preserved exactly', () {
      final seq = [
        List.generate(43, (i) => [i * 0.01, i * 0.02, i * 0.03]),
        List.generate(43, (i) => [i * 0.02, i * 0.03, i * 0.04]),
      ];
      final out = GestureSegmenter.resample(seq, target);
      expect(out.length, target);
      expect(out.first.length, 43);
      expect(out.first.first.length, 3);
    });
  });

  group('meanDisplacement', () {
    test('identical frames register no motion', () {
      expect(GestureSegmenter.meanDisplacement(frameAt(0.5, 0.5), frameAt(0.5, 0.5)),
          closeTo(0, 1e-9));
    });

    test('a pure translation registers exactly that distance', () {
      expect(GestureSegmenter.meanDisplacement(frameAt(0.0, 0.0), frameAt(0.3, 0.4)),
          closeTo(0.5, 1e-6)); // 3-4-5 triangle
    });

    test('empty input is zero, not a crash or NaN', () {
      expect(GestureSegmenter.meanDisplacement(const [], const []), 0);
    });
  });

  group('segmentation', () {
    test('a still hand alone does not emit -- too few frames to be a sign', () {
      final s = GestureSegmenter();
      List<List<List<double>>>? emitted;
      for (var i = 0; i < 5; i++) {
        emitted ??= s.add(frameAt(0.5, 0.5), target);
      }
      expect(emitted, isNull);
    });

    test('movement followed by a settle emits exactly one gesture', () {
      final s = GestureSegmenter();
      final results = <List<List<List<double>>>>[];

      // Moving: 10 frames of clear displacement.
      for (var i = 0; i < 10; i++) {
        final r = s.add(frameAt(i * 0.1, 0.2), target);
        if (r != null) results.add(r);
      }
      // Settling: hand holds position.
      for (var i = 0; i < GestureSegmenter.stillFramesToSegment + 1; i++) {
        final r = s.add(frameAt(0.9, 0.2), target);
        if (r != null) results.add(r);
      }

      expect(results.length, 1, reason: 'one movement + settle should be one gesture');
      expect(results.first.length, target, reason: 'emitted gesture must be resampled to seqLen');
    });

    test('two movement+settle cycles emit two separate gestures', () {
      final s = GestureSegmenter();
      final results = <List<List<List<double>>>>[];
      void cycle(double base) {
        for (var i = 0; i < 10; i++) {
          final r = s.add(frameAt(base + i * 0.1, 0.2), target);
          if (r != null) results.add(r);
        }
        for (var i = 0; i < GestureSegmenter.stillFramesToSegment + 1; i++) {
          final r = s.add(frameAt(base + 0.9, 0.2), target);
          if (r != null) results.add(r);
        }
      }

      cycle(0.0);
      cycle(0.0);
      expect(results.length, 2);
      expect(results.every((g) => g.length == target), isTrue);
    });

    test('continuous motion force-segments instead of growing without bound', () {
      final s = GestureSegmenter();
      var emitted = 0;
      // Always-moving hand: never settles, so only the length cap can fire.
      for (var i = 0; i < GestureSegmenter.maxSegmentFrames + 5; i++) {
        if (s.add(frameAt((i % 2) * 0.5, 0.2), target) != null) emitted++;
      }
      expect(emitted, greaterThanOrEqualTo(1));
      expect(s.bufferedFrames, lessThan(GestureSegmenter.maxSegmentFrames));
    });

    test('flush emits when the hand leaves after a real gesture', () {
      final s = GestureSegmenter();
      for (var i = 0; i < 10; i++) {
        s.add(frameAt(i * 0.1, 0.2), target);
      }
      final out = s.flush(target);
      expect(out, isNotNull);
      expect(out!.length, target);
    });

    test('flush after only a couple of frames emits nothing', () {
      final s = GestureSegmenter();
      s.add(frameAt(0.1, 0.2), target);
      s.add(frameAt(0.2, 0.2), target);
      expect(s.flush(target), isNull);
    });

    test('reset clears accumulated frames', () {
      final s = GestureSegmenter();
      for (var i = 0; i < 8; i++) {
        s.add(frameAt(i * 0.1, 0.2), target);
      }
      expect(s.bufferedFrames, greaterThan(0));
      s.reset();
      expect(s.bufferedFrames, 0);
      expect(s.flush(target), isNull);
    });

    test('buffer is emptied on emit, so consecutive gestures do not accumulate', () {
      final s = GestureSegmenter();
      for (var i = 0; i < 10; i++) {
        s.add(frameAt(i * 0.1, 0.2), target);
      }
      // Exactly the threshold: the emit fires on the last of these, so
      // nothing has yet been accumulated toward a following gesture.
      for (var i = 0; i < GestureSegmenter.stillFramesToSegment; i++) {
        s.add(frameAt(0.9, 0.2), target);
      }
      expect(s.bufferedFrames, 0);

      // A frame arriving after the boundary starts the next gesture rather
      // than being appended to the one already emitted.
      s.add(frameAt(0.9, 0.2), target);
      expect(s.bufferedFrames, 1);
    });
  });
}
