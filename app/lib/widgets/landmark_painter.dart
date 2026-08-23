import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

import '../app_theme.dart';

/// Connections between MediaPipe's 21 hand landmarks, grouped by finger.
class HandLandmarkConnections {
  static const List<List<int>> connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [5, 9], [9, 10], [10, 11], [11, 12], // Middle
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20], // Pinky + palm
  ];
}

/// Draws the live hand skeleton over the camera preview.
///
/// The transform (rotate by sensor orientation, mirror for the front
/// camera, then map normalized landmark coords to the preview's logical
/// size) mirrors the plugin's own example so the overlay lines up with
/// what the sensor actually saw.
class LandmarkPainter extends CustomPainter {
  LandmarkPainter({
    required this.hands,
    required this.previewSize,
    required this.lensDirection,
    required this.sensorOrientation,
  });

  final List<Hand> hands;
  final Size previewSize;
  final CameraLensDirection lensDirection;
  final int sensorOrientation;

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty) return;
    final scale = size.width / previewSize.height;

    final dotPaint = Paint()
      ..color = AppColors.amber
      ..strokeWidth = 8 / scale
      ..strokeCap = StrokeCap.round;

    final linePaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.85)
      ..strokeWidth = 3 / scale;

    canvas.save();
    final center = Offset(size.width / 2, size.height / 2);
    canvas.translate(center.dx, center.dy);
    canvas.rotate(sensorOrientation * math.pi / 180);

    if (lensDirection == CameraLensDirection.front) {
      canvas.scale(-1, 1);
      canvas.rotate(math.pi);
    }
    canvas.scale(scale);

    final logicalWidth = previewSize.width;
    final logicalHeight = previewSize.height;

    for (final hand in hands) {
      for (final connection in HandLandmarkConnections.connections) {
        final start = hand.landmarks[connection[0]];
        final end = hand.landmarks[connection[1]];
        canvas.drawLine(
          Offset((start.x - 0.5) * logicalWidth, (start.y - 0.5) * logicalHeight),
          Offset((end.x - 0.5) * logicalWidth, (end.y - 0.5) * logicalHeight),
          linePaint,
        );
      }
      for (final landmark in hand.landmarks) {
        canvas.drawCircle(
          Offset((landmark.x - 0.5) * logicalWidth, (landmark.y - 0.5) * logicalHeight),
          6 / scale,
          dotPaint,
        );
      }
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant LandmarkPainter oldDelegate) => true;
}
