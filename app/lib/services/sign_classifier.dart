import 'dart:math' as math;

import 'package:hand_landmarker/hand_landmarker.dart';

import 'sign_category.dart';

export 'hand_selection.dart' show primaryHand;

/// One classification result: a label plus a confidence in [0, 1].
class ClassificationResult {
  final String label;
  final double confidence;
  const ClassificationResult(this.label, this.confidence);
}

/// Anything that can turn a frame's detected hands into a recognized sign.
///
/// [FingerCountClassifier] below is the only implementation today. Once the
/// ML pipeline in `ml/` exports a trained `.tflite` word classifier, add a
/// `TfliteSignClassifier implements SignClassifier` that runs the exported
/// model over a sliding window of landmark frames (see ml/README.md) and
/// swap it in via `HomeScreen`'s `classifier` field. Nothing else in the UI
/// needs to change — every screen only talks to this interface.
abstract class SignClassifier {
  /// Shown in the UI so it's always obvious which engine produced a label.
  String get modelName;

  /// True for placeholder/heuristic engines, so the UI can badge them as such.
  bool get isPrototype;

  /// Restricts recognition to labels in [categories] -- e.g. "Letters only"
  /// while fingerspelling, so a held-steady "C" can't lose to a superficially
  /// similar word (or vice versa) just because both live in the same
  /// underlying model's output. Ignored by classifiers with only one
  /// category (like [FingerCountClassifier]).
  void setAllowedCategories(Set<SignCategory> categories);

  ClassificationResult? classify(List<Hand> hands);
}

/// A rule-based placeholder: counts extended fingers on the first detected
/// hand. It exists to prove the camera -> landmark -> classify -> UI pipeline
/// end-to-end *today*, before any dataset has been trained on. It does not
/// implement real ASL/BdSL sign recognition and is labelled as a prototype
/// everywhere it surfaces in the UI.
class FingerCountClassifier implements SignClassifier {
  @override
  String get modelName => 'Finger-count heuristic (prototype)';

  @override
  bool get isPrototype => true;

  // Nothing to restrict: every label this heuristic can produce ("0"-"5")
  // is a single, undifferentiated category, so the mode toggle has no
  // effect on it -- it's only ever shown as a fallback anyway.
  @override
  void setAllowedCategories(Set<SignCategory> categories) {}

  static const _tips = [8, 12, 16, 20];
  static const _pips = [6, 10, 14, 18];

  @override
  ClassificationResult? classify(List<Hand> hands) {
    if (hands.isEmpty) return null;
    final landmarks = hands.first.landmarks;
    if (landmarks.length < 21) return null;

    var count = 0;
    for (var i = 0; i < _tips.length; i++) {
      if (landmarks[_tips[i]].y < landmarks[_pips[i]].y - 0.02) count++;
    }

    final wrist = landmarks[0];
    final thumbTip = landmarks[4];
    final indexMcp = landmarks[5];
    final pinkyMcp = landmarks[17];
    final thumbSpread = _dist(thumbTip, wrist);
    final palmWidth = _dist(indexMcp, pinkyMcp);
    if (palmWidth > 0 && thumbSpread > palmWidth * 0.9) count++;

    return ClassificationResult('$count', 1.0);
  }

  double _dist(Landmark a, Landmark b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return math.sqrt(dx * dx + dy * dy);
  }
}
