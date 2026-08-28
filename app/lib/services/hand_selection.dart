import 'package:hand_landmarker/hand_landmarker.dart';

/// Picks the one hand a single-hand-only context (digits, fingerspelling
/// letters, or a hands-free control gesture -- in both ASL and BdSL, every
/// one of these was trained or designed around exactly one hand) should
/// work from, when the plugin reports more than one.
///
/// A second hand shows up on real devices most often because the signer is
/// holding the phone with it, not because the sign or gesture itself is
/// two-handed. Larger on-screen bounding box wins: the hand deliberately
/// held up to sign is normally closer to the camera and more prominent,
/// while a hand just holding the phone or resting nearby tends to read
/// smaller. This is deterministic and stateless (a pure function of the
/// current frame's landmarks), unlike trusting whichever order the plugin
/// happened to list hands in, which can flip which hand is "first" from
/// one frame to the next with no change in what the signer is doing --
/// that flip is what made both recognition and the hands-free control
/// gestures intermittently fail with two hands in frame, even though only
/// one hand ever mattered.
Hand? primaryHand(List<Hand> hands) {
  if (hands.isEmpty) return null;
  if (hands.length == 1) return hands.first;
  var best = hands.first;
  var bestArea = _bboxArea(best);
  for (final h in hands.skip(1)) {
    final area = _bboxArea(h);
    if (area > bestArea) {
      best = h;
      bestArea = area;
    }
  }
  return best;
}

double _bboxArea(Hand hand) {
  final lm = hand.landmarks;
  if (lm.isEmpty) return 0.0;
  var minX = lm[0].x, maxX = lm[0].x, minY = lm[0].y, maxY = lm[0].y;
  for (final p in lm) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return (maxX - minX) * (maxY - minY);
}
