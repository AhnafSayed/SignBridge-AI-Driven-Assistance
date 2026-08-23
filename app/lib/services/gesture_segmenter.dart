import 'dart:math' as math;

/// Finds where one sign ends and the next begins, from landmark motion
/// alone.
///
/// The problem it solves. Training resamples each *complete,
/// pre-segmented* gesture to exactly `seqLen` frames, so the model learned
/// motion shape independently of how long the sign took. Inference
/// originally fed it a rolling window of the last `seqLen` raw camera
/// frames instead -- a window that can straddle two signs or hold a
/// half-finished one, and whose effective playback speed depends entirely
/// on the camera's frame rate. That is a real train/inference mismatch,
/// and it is why the interface had to ask the user to hold every sign
/// steady rather than simply sign.
///
/// The approach. A sign is movement followed by a settle, so per-frame
/// motion energy separates them: while the hand is moving, frames
/// accumulate; once motion stays below [stillMotionThreshold] for
/// [stillFramesToSegment] consecutive frames, the sign is treated as
/// finished and the accumulated frames are resampled to `seqLen` -- the
/// same linear interpolation ml/src/landmarks.py's `resample_sequence`
/// applies during training. The model therefore receives a complete
/// gesture in the same form its training data took.
///
/// This is deliberately a *fallible* detector, and callers are expected to
/// keep the rolling window as a fallback. Continuous or fidgety signing
/// may never produce a clean settle, and a segmenter that waited forever
/// for one would make the app stop responding entirely -- a far worse
/// failure than a slightly misaligned window.
class GestureSegmenter {
  /// Mean per-landmark displacement between consecutive frames below which
  /// the hand counts as "still", in normalized landmark units.
  static const double stillMotionThreshold = 0.055;

  /// Consecutive still frames that end a gesture. Short enough to feel
  /// immediate; long enough not to fire on the brief pauses within a
  /// single sign.
  static const int stillFramesToSegment = 4;

  /// Shortest accepted gesture. Below this, the "sign" is almost always
  /// the hand entering frame or a stray twitch.
  static const int minSegmentFrames = 6;

  /// Longest accumulation before force-segmenting, so a continuously
  /// moving hand still yields results instead of growing without bound.
  static const int maxSegmentFrames = 90;

  final List<List<List<double>>> _segment = [];
  List<List<double>>? _previous;
  int _stillFrames = 0;

  int get bufferedFrames => _segment.length;

  /// Feeds one frame. Returns a completed, resampled gesture when a
  /// boundary is detected, otherwise null.
  List<List<List<double>>>? add(List<List<double>> frame, int targetLength) {
    // The first frame of a gesture has no predecessor to compare against;
    // treating that as "infinite motion" starts the still-counter cleanly
    // at zero rather than immediately counting it as settled.
    final motion = _previous == null ? double.infinity : meanDisplacement(_previous!, frame);
    _previous = frame;
    _segment.add(frame);

    if (motion.isFinite && motion < stillMotionThreshold) {
      _stillFrames++;
    } else {
      _stillFrames = 0;
    }

    final settled = _stillFrames >= stillFramesToSegment && _segment.length >= minSegmentFrames;
    if (settled || _segment.length >= maxSegmentFrames) {
      return _take(targetLength);
    }
    return null;
  }

  /// Ends the current gesture because the hand left the frame -- the most
  /// natural way for a signer to indicate they have finished. Returns null
  /// if too little was accumulated to be a real sign.
  List<List<List<double>>>? flush(int targetLength) {
    if (_segment.length < minSegmentFrames) {
      reset();
      return null;
    }
    return _take(targetLength);
  }

  void reset() {
    _segment.clear();
    _previous = null;
    _stillFrames = 0;
  }

  List<List<List<double>>>? _take(int targetLength) {
    if (_segment.length < minSegmentFrames) {
      reset();
      return null;
    }
    final out = resample(_segment, targetLength);
    // Only the accumulated frames and still-counter reset; `_previous` is
    // intentionally left in place so motion continues to be measured
    // across the boundary rather than reporting a spurious infinity on the
    // next frame.
    _segment.clear();
    _stillFrames = 0;
    return out;
  }

  /// Mean 2D displacement between corresponding landmarks of two frames.
  static double meanDisplacement(List<List<double>> a, List<List<double>> b) {
    final n = math.min(a.length, b.length);
    if (n == 0) return 0;
    var total = 0.0;
    for (var i = 0; i < n; i++) {
      final dx = a[i][0] - b[i][0];
      final dy = a[i][1] - b[i][1];
      total += math.sqrt(dx * dx + dy * dy);
    }
    return total / n;
  }

  /// Linear interpolation over the frame axis, mirroring
  /// `landmarks.resample_sequence` so a segmented gesture reaches the model
  /// in the same form the training data did.
  static List<List<List<double>>> resample(List<List<List<double>>> seq, int target) {
    if (seq.isEmpty) return const [];
    if (seq.length == target) return List.of(seq);
    if (seq.length == 1 || target == 1) {
      return List.generate(target, (_) => seq.first);
    }
    final rows = seq.first.length;
    final cols = seq.first.first.length;
    final out = <List<List<double>>>[];
    for (var t = 0; t < target; t++) {
      final pos = t * (seq.length - 1) / (target - 1);
      final lo = pos.floor();
      final hi = pos.ceil();
      final frac = pos - lo;
      final frame = <List<double>>[];
      for (var r = 0; r < rows; r++) {
        final row = <double>[];
        for (var c = 0; c < cols; c++) {
          final a = seq[lo][r][c];
          final b = seq[hi][r][c];
          row.add(a + (b - a) * frac);
        }
        frame.add(row);
      }
      out.add(frame);
    }
    return out;
  }
}
