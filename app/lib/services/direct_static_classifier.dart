import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart' show rootBundle;
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'sign_category.dart';
import 'sign_classifier.dart';

/// Runs a trained single-frame `.tflite` model (see ../../../ml/
/// train_static_direct.py) directly on one hand's 21 landmarks -- no
/// sequence buffer, no gesture segmentation, no 40-frame wait. Built for
/// digits and fingerspelling letters specifically, which are static poses
/// with no motion to segment in the first place.
///
/// Why this exists alongside [TfliteSignClassifier] rather than replacing
/// it: static poses were being forced through the sequence contract built
/// for *dynamic* word signs -- repeated 40 times and run through the
/// motion-energy gesture segmenter meant to find where a moving sign
/// starts and ends, real overhead and latency for something that never
/// moves. A direct classifier removes that mismatch for the categories
/// where it doesn't belong, while leaving word recognition on the
/// sequence model it actually needs.
///
/// Measured before shipping, not assumed: on ASL's own held-out static
/// data this scores 99.69% (digit 100%, letter 99.57%), matching or
/// beating the sequence model's static-category accuracy under the same
/// evaluation. The equivalent BdSL model was also trained and measured --
/// against a genuine held-out-*signer* split, since BdSL's static sources
/// carry real signer identity -- and came in 20-30 points worse than the
/// shipped BdSL sequence model for signers it had never seen. BdSL's
/// letters/digits therefore stay on [TfliteSignClassifier]; this class is
/// wired up for ASL only (see HomeScreenState._classifierFor).
class DirectStaticClassifier implements SignClassifier {
  DirectStaticClassifier._(this._interpreter, this._isolateInterpreter, this._labels, this._name) {
    _allowedMask = List.filled(_labels.length, true);
  }

  final Interpreter _interpreter;
  final IsolateInterpreter _isolateInterpreter;
  final List<String> _labels;
  final String _name;
  late List<bool> _allowedMask;

  ClassificationResult? _latestResult;
  bool _inferring = false;

  static const int _numLandmarks = 21;
  static const int _wrist = 0;
  static const int _middleMcp = 9;

  static Future<DirectStaticClassifier> create({
    required String modelAsset,
    required String labelsAsset,
    required String displayName,
  }) async {
    final interpreter = await Interpreter.fromAsset(modelAsset);
    final isolateInterpreter = await IsolateInterpreter.create(address: interpreter.address);
    final labelsJson = await rootBundle.loadString(labelsAsset);
    final labels = (jsonDecode(labelsJson) as List).cast<String>();
    return DirectStaticClassifier._(interpreter, isolateInterpreter, labels, displayName);
  }

  @override
  String get modelName => _name;

  @override
  bool get isPrototype => false;

  @override
  void setAllowedCategories(Set<SignCategory> categories) {
    _allowedMask = _labels.map((l) => categories.contains(categoryOf(l))).toList();
  }

  @override
  ClassificationResult? classify(List<Hand> hands) {
    if (hands.isEmpty) {
      _latestResult = null;
      return null;
    }
    // Digits and fingerspelling are one-handed in both ASL and BdSL, so a
    // single detected hand is the whole input -- matches training, which
    // never saw more than one hand per static sample.
    final landmarks = hands.first.landmarks;
    if (landmarks.length < _numLandmarks) return _latestResult;

    final origin = landmarks[_wrist];
    final mid = landmarks[_middleMcp];
    final dx = mid.x - origin.x, dy = mid.y - origin.y, dz = mid.z - origin.z;
    final scale = math.sqrt(dx * dx + dy * dy + dz * dz);
    final invScale = scale < 1e-6 ? 1.0 : 1.0 / scale;

    final input = List<double>.filled(63, 0.0);
    for (var i = 0; i < _numLandmarks; i++) {
      final l = landmarks[i];
      input[i * 3] = (l.x - origin.x) * invScale;
      input[i * 3 + 1] = (l.y - origin.y) * invScale;
      input[i * 3 + 2] = (l.z - origin.z) * invScale;
    }

    _startInference(input);
    return _latestResult;
  }

  void _startInference(List<double> input) {
    if (_inferring) return;
    _inferring = true;
    unawaited(_runInference(input));
  }

  Future<void> _runInference(List<double> input) async {
    try {
      final output = [List<double>.filled(_labels.length, 0.0)];
      await _isolateInterpreter.run([input], output);

      final scores = output[0];
      var bestIdx = -1;
      var allowedSum = 0.0;
      for (var i = 0; i < scores.length; i++) {
        if (!_allowedMask[i]) continue;
        allowedSum += scores[i];
        if (bestIdx == -1 || scores[i] > scores[bestIdx]) bestIdx = i;
      }
      // Renormalized within the allowed category, not the raw softmax --
      // the same fix TfliteSignClassifier needed (see its own comment):
      // masking restricts which class can win but not, on its own, what
      // share of the total probability its score reflects.
      final confidence = bestIdx == -1 || allowedSum <= 0 ? 0.0 : scores[bestIdx] / allowedSum;
      _latestResult = bestIdx == -1 ? null : ClassificationResult(_labels[bestIdx], confidence);
    } finally {
      _inferring = false;
    }
  }

  void dispose() {
    _isolateInterpreter.close();
    _interpreter.close();
  }
}
