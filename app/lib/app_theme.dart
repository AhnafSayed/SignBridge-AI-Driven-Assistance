import 'package:flutter/material.dart';

/// SignBridge brand palette.
///
/// A friendly, calm blue carries the app's chrome (app bar, nav, buttons);
/// warm amber is reserved for the one thing that should pop on every
/// screen: the recognized sign.
class AppColors {
  static const primary = Color(0xFF2E6BB3);
  static const primaryDark = Color(0xFF1E4A80);
  static const amber = Color(0xFFF2A93B);
  static const ink = Color(0xFF16232E);
  static const inkSoft = Color(0xFF4B5A68);
  static const paper = Color(0xFFF6F4EE);
  // A warm off-white rather than pure white, so cards and surfaces don't
  // read as stark/clinical next to the paper background.
  static const surfaceWarm = Color(0xFFFCFAF6);
  // A touch softer than a pure alarm-red, so delete/clear actions read as
  // "careful" rather than "danger" -- this app's most destructive action
  // is clearing a sentence, not something catastrophic.
  static const coral = Color(0xFFBF5A42);
  // Frames the camera preview when reading BdSL, so the active language is
  // never something you have to double-check further up the screen for --
  // deliberately not blue (already `primary`, ASL's own accent) and not a
  // literal flag color, just a clearly distinct third hue.
  static const bdslAccent = Color(0xFF2F7D5C);
}

class AppTheme {
  static ThemeData light() {
    final scheme = ColorScheme.fromSeed(
      seedColor: AppColors.primary,
      brightness: Brightness.light,
      primary: AppColors.primary,
      secondary: AppColors.amber,
      surface: AppColors.surfaceWarm,
    );
    return _build(scheme, AppColors.paper);
  }

  static ThemeData dark() {
    final scheme = ColorScheme.fromSeed(
      seedColor: AppColors.primary,
      brightness: Brightness.dark,
      primary: AppColors.amber,
      secondary: AppColors.primary,
      surface: const Color(0xFF15202B),
    );
    return _build(scheme, const Color(0xFF0C141C));
  }

  static ThemeData _build(ColorScheme scheme, Color background) {
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: background,
      // Solid brand blue, not the theme background, in both light and dark
      // mode: the app bar and nav bar are the one piece of "always the same
      // furniture" a user orients by, so they carry the brand color while
      // page content underneath can vary.
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.primary,
        foregroundColor: Colors.white,
        elevation: 2,
        shadowColor: Colors.black26,
        centerTitle: false,
        titleTextStyle: TextStyle(
          fontWeight: FontWeight.w700,
          fontSize: 20,
          letterSpacing: 0.2,
          color: Colors.white,
        ),
        iconTheme: IconThemeData(color: Colors.white),
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: scheme.surface,
        elevation: 3,
        indicatorColor: AppColors.amber.withValues(alpha: 0.22),
        iconTheme: WidgetStateProperty.resolveWith(
          (states) => IconThemeData(
            color: states.contains(WidgetState.selected) ? AppColors.primaryDark : AppColors.inkSoft,
          ),
        ),
        labelTextStyle: WidgetStateProperty.resolveWith(
          (states) => TextStyle(
            fontSize: 12,
            fontWeight: states.contains(WidgetState.selected) ? FontWeight.w700 : FontWeight.w500,
            color: states.contains(WidgetState.selected) ? AppColors.primaryDark : AppColors.inkSoft,
          ),
        ),
      ),
      textTheme: const TextTheme(
        headlineMedium: TextStyle(fontWeight: FontWeight.w700, height: 1.15),
        titleMedium: TextStyle(fontWeight: FontWeight.w600),
        labelLarge: TextStyle(
          fontWeight: FontWeight.w600,
          letterSpacing: 0.6,
        ),
      ),
      chipTheme: ChipThemeData(
        selectedColor: scheme.primary,
        labelStyle: TextStyle(color: scheme.onSurface, fontSize: 13),
        side: BorderSide(color: scheme.outlineVariant),
        shape: const StadiumBorder(),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: AppColors.primary,
          foregroundColor: Colors.white,
          shape: const StadiumBorder(),
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 14),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.primaryDark,
          side: const BorderSide(color: AppColors.primary),
          shape: const StadiumBorder(),
        ),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected) ? AppColors.primary : null,
        ),
        trackColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected) ? AppColors.primary.withValues(alpha: 0.4) : null,
        ),
      ),
    );
  }
}
