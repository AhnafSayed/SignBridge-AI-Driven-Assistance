import 'dart:math' as math;

import 'package:hand_landmarker/hand_landmarker.dart';

/// Non-linguistic hand shapes that control the sentence instead of adding
/// a word to it. Deliberately just three, all static handshapes (no motion
/// tracking) so they're cheap to detect every frame and easy for a user to
/// hold deliberately:
///
///  - closed fist                    -> remove the last word (backspace)
///  - thumb down, fist otherwise     -> clear the whole sentence
///  - thumb up, fist otherwise       -> speak the sentence right now
///
/// No "space" gesture exists on purpose: each auto-committed word already
/// becomes its own entry in the sentence list, so words come pre-spaced --
/// mirroring how sign-to-text keyboards (e.g. Gboard's sign-typing mode)
/// auto-space between recognized signs rather than requiring an explicit
/// space gesture.
///
/// Checked in every [SignCategory], not just Words -- an earlier build
/// restricted these to Words mode only, since several real letters/digits
/// are themselves closed-fist shapes (ASL A/S/E/M/N/T, for one) that would
/// always lose the fist-shape tiebreak to backspace. That restriction went
/// too far: it silently disabled backspace, clear, *and* speak together
/// any time the active category wasn't Words, which is most of fingerspelling.
/// See HomeScreenState._onHands's `wordIsConfident` check for the actual
/// fix -- a confidence-based tiebreak, workable now that letters/digits are
/// measured to be extremely confident when correctly formed.
enum ControlGesture { none, backspace, clearAll, speak }

class ControlGestureDetector {
  ControlGesture detect(List<Hand> hands) {
    if (hands.isEmpty) return ControlGesture.none;
    final lm = hands.first.landmarks;
    if (lm.length < 21) return ControlGesture.none;

    final fingersUp = [
      lm[8].y < lm[6].y - 0.02,
      lm[12].y < lm[10].y - 0.02,
      lm[16].y < lm[14].y - 0.02,
      lm[20].y < lm[18].y - 0.02,
    ];
    // Any extended finger rules out every one of these three shapes --
    // they're all fist-based (thumb aside), so this is the cheapest,
    // clearest disqualifier and gets checked first.
    if (fingersUp.any((f) => f)) return ControlGesture.none;

    // Which way is the thumb pointing, and is it pointing anywhere at all
    // (as opposed to tucked into the fist)? Both questions in one measure:
    // the thumb's own direction vector (tip relative to its own base
    // joint, not the wrist -- see below for why that distinction matters),
    // normalized by a stable hand-scale reference (wrist-to-middle-knuckle,
    // the same scale ml/src/landmarks.py's normalization uses elsewhere in
    // this project).
    //
    // An earlier version measured thumb extension as
    // `dist(thumb_tip, wrist) > dist(index_mcp, pinky_mcp) * 0.9` --
    // thumb-to-*wrist* distance against palm *width*. Checked against real
    // extracted landmarks from this project's own training data
    // (ml/data/asl_alphabet_landmarks.npz), that came out **true for every
    // single closed-fist letter tested, including the canonical ASL "A"**
    // -- the wrist is structurally far from the thumb regardless of
    // whether the thumb is tucked in or held out, so that comparison was
    // measuring hand size, not thumb curl. The practical effect: `thumbUp`
    // was true almost unconditionally, which made the `!thumbUp` branch
    // below -- backspace -- effectively unreachable for any real hand.
    // That's not a minor miscalibration; it meant holding a closed fist
    // could never register as delete, only ever as clear or speak
    // depending on incidental thumb position, which matches real-device
    // testing reporting the hands-free gestures as broken.
    //
    // The fix measures direction from the thumb's *own* base joint
    // (thumb_mcp) instead of the wrist, and requires the vector to be
    // both large (clearly extended, not just resting) and close to
    // vertical (a deliberate up/down point, not the more horizontal or
    // diagonal thumb positions most fingerspelling letters actually use).
    //
    // This detector is language-agnostic -- it runs on raw hand landmarks
    // with no ASL/BdSL branch anywhere in it -- so it was measured against
    // both trained vocabularies' own static training data, not just one:
    //
    //  - ASL (ml/data/asl_alphabet_landmarks.npz, 26 letters): the
    //    closed-fist disqualifier above reads as a `backspace` shape for
    //    S 100%, E/Q 99%, M 91%, A 88%, T 77%, O 70%, P 55% of sampled
    //    poses. An earlier version of this comment reported only A/M/N,
    //    from having checked those three specifically rather than
    //    sweeping all 26 -- undercounting the real collision surface by
    //    more than double.
    //  - BdSL (ml/src/dataset_bdsl_static.py's CSV loader, 37 letters + 10
    //    digits): wider still -- 15 letters and 2 digits exceed 30%
    //    backspace or 15% clear/speak, topped by letter "Taa" at 99%
    //    backspace and digit "6" at 31% backspace *and* 30% clear/speak.
    //    The digit-6 case matters more than the letter cases: clearAll
    //    erases the whole sentence, not just the last word, so a false
    //    positive there is the single most damaging failure this detector
    //    can produce.
    //
    // The reason any of this is survivable is that HomeScreenState's
    // arbitration doesn't stop at a per-frame confidence tiebreak. Any
    // frame that clears the *same* bar the app already uses to decide a
    // sign is commit-worthy (_minCommitConfidenceFor -- 0.4 for
    // letters/digits, 0.55 for words) suppresses control outright. An
    // earlier version demanded a separate, higher 0.6 bar for suppression
    // specifically, which left a real gap: a frame confident enough to
    // eventually commit as a letter (>=0.4) but under 0.6 could still lose
    // the tiebreak to its own fist-shape reading. Caught live on a real
    // device -- a held "S" landed in that gap and backspace deleted a
    // previous word mid-sign. Real devices are not clean benchmark
    // measurements, and confidence can sit in that kind of band for a run
    // of frames under real lighting/angle conditions. Because the
    // backspace hold itself requires ~900ms of *continuous* low
    // confidence to fire (_updateControlStability), a dip that lasts
    // nearly that long during genuine signing would delete or clear
    // content instead of committing it -- a materially worse failure than
    // a missed recognition, and what real-device testing reported as
    // gestures "colliding" with recognition. `ControlArbiter`'s cooldown
    // hysteresis (see its own
    // docstring) exists specifically to bridge that gap, for both
    // languages at once, since it lives in HomeScreenState's arbitration
    // rather than in either language's model.
    final scale = _dist(lm[0], lm[9]); // wrist -> middle_mcp
    if (scale < 1e-6) return ControlGesture.backspace;
    final dx = lm[4].x - lm[2].x;
    final dy = lm[4].y - lm[2].y;
    final verticalThumb = dx.abs() < dy.abs() * 0.6 && dy.abs() > scale * 0.75;

    if (!verticalThumb) return ControlGesture.backspace;
    return dy > 0 ? ControlGesture.clearAll : ControlGesture.speak; // y grows downward
  }

  double _dist(Landmark a, Landmark b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return math.sqrt(dx * dx + dy * dy);
  }
}

/// Decides whether a detected control handshape should actually be allowed
/// to fire this frame, or suppressed because a real recognition is -- or
/// was, within [suppressCooldown], recently -- confidently underway.
///
/// This exists because several trained letters are closed fists that
/// always read as `ControlGesture.backspace` geometrically (see the
/// measurements in [ControlGestureDetector]'s docstring: ASL "S" reads as
/// a fist shape in 100% of its own training samples, "E"/"Q" in 99%).
/// A per-frame confidence check alone is not enough to protect them: the
/// backspace hold itself requires ~900ms of *continuous* non-confident
/// frames to fire, and on a real device confidence is not the clean
/// offline number it is on a benchmark -- it can dip below the override
/// bar for a run of frames approaching that duration during genuine,
/// correctly-formed signing. When that happens without this class,
/// backspace fires and deletes what the user just signed -- a materially
/// worse failure than a missed recognition.
///
/// The fix is hysteresis rather than a looser geometry check or a lower
/// confidence bar: once a recognition has been confident, treat that as
/// evidence signing is still underway and keep suppressing control through
/// a brief dip. A genuine control gesture -- made only once a sign has
/// actually finished, when the hand is beginning to change shape rather
/// than mid-hold -- is never meaningfully delayed by a 500ms window, since
/// its own 900ms+ hold requirement outlasts the cooldown comfortably.
///
/// Pulled out as its own class, taking `now` as an explicit parameter
/// rather than calling `DateTime.now()` internally, so it is unit-testable
/// with fabricated timestamps -- the same pattern [GestureSegmenter] and
/// [ControlGestureDetector] already use to stay testable without a camera
/// or widget tree.
class ControlArbiter {
  ControlArbiter({this.suppressCooldown = const Duration(milliseconds: 500)});

  final Duration suppressCooldown;
  DateTime? _lastConfidentAt;

  /// Returns [ControlGesture.none] if this frame (or a recent one, within
  /// [suppressCooldown]) was a confident recognition; otherwise returns
  /// [rawControl] unchanged.
  ControlGesture arbitrate({
    required bool wordIsConfident,
    required ControlGesture rawControl,
    required DateTime now,
  }) {
    if (wordIsConfident) _lastConfidentAt = now;
    final recentlyConfident =
        _lastConfidentAt != null && now.difference(_lastConfidentAt!) < suppressCooldown;
    return (wordIsConfident || recentlyConfident) ? ControlGesture.none : rawControl;
  }

  /// Clears the remembered confident-recognition time -- call when the
  /// hand leaves frame, so a stale timestamp from before it left can't
  /// suppress control on an unrelated gesture made after it returns.
  void reset() {
    _lastConfidentAt = null;
  }
}
