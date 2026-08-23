import 'package:flutter/material.dart';

/// App bar title with the hand mark beside the screen name, so every
/// screen still reads as SignBridge even when its title is just "Listen"
/// or "About this build".
class BrandTitle extends StatelessWidget {
  const BrandTitle(this.title, {super.key});
  final String title;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Image.asset('assets/branding/mark.png', height: 26, width: 26),
        const SizedBox(width: 10),
        Text(title),
      ],
    );
  }
}
