import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app_theme.dart';
import '../services/ui_strings.dart';
import '../state/app_state.dart';
import '../widgets/brand_title.dart';
import 'about_screen.dart';

/// A dedicated settings screen, replacing the overflow popup menu that
/// used to hold these five items. The menu made sense while there were
/// two or three settings; once Theme, App language, Auto mode,
/// Hand-shape gestures, and About had all accumulated in it, a blind
/// "tap to cycle" popup was the wrong interaction for things a user might
/// genuinely want to see all the options for at once (Theme and App
/// language) or just flip on/off with a normal switch (Auto mode,
/// Hand-shape gestures), not guess at through repeated taps.
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();
    final bangla = appState.uiBangla;
    return Scaffold(
      appBar: AppBar(title: BrandTitle(uiText(bangla, 'settings', 'Settings'))),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _SectionLabel(uiText(bangla, 'appearance', 'Appearance')),
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(
              uiText(bangla, 'theme', 'Theme'),
              style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
            ),
          ),
          Padding(
            padding: const EdgeInsets.only(bottom: 20),
            child: SegmentedButton<ThemeMode>(
              segments: [
                ButtonSegment(
                  value: ThemeMode.system,
                  icon: const Icon(Icons.brightness_auto),
                  label: Text(uiText(bangla, 'theme_system', 'Match device')),
                ),
                ButtonSegment(
                  value: ThemeMode.light,
                  icon: const Icon(Icons.light_mode),
                  label: Text(uiText(bangla, 'theme_light', 'Light')),
                ),
                ButtonSegment(
                  value: ThemeMode.dark,
                  icon: const Icon(Icons.dark_mode),
                  label: Text(uiText(bangla, 'theme_dark', 'Dark')),
                ),
              ],
              selected: {appState.themeMode},
              onSelectionChanged: (s) => appState.setThemeMode(s.first),
            ),
          ),
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(
              uiText(bangla, 'app_language', 'App language'),
              style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
            ),
          ),
          Text(
            uiText(bangla, 'app_language_note',
                'Independent of which sign language the camera reads -- this is just the '
                'language of the app\'s own buttons and menus.'),
            style: const TextStyle(color: AppColors.inkSoft, fontSize: 12.5, height: 1.4),
          ),
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.only(bottom: 24),
            child: SegmentedButton<bool>(
              segments: const [
                ButtonSegment(value: false, label: Text('English')),
                ButtonSegment(value: true, label: Text('বাংলা')),
              ],
              selected: {bangla},
              onSelectionChanged: (s) => appState.toggleUiLanguage(),
            ),
          ),
          const Divider(),
          _SectionLabel(uiText(bangla, 'recognition', 'Recognition')),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            secondary: Icon(
              appState.autoAssist ? Icons.auto_awesome : Icons.auto_awesome_outlined,
              color: appState.autoAssist ? AppColors.amber : null,
            ),
            title: Text(uiText(bangla, 'auto_mode', 'Auto mode')),
            subtitle: Text(
              appState.autoAssist
                  ? uiText(bangla, 'auto_mode_on', 'Hold signs to add, pause to speak')
                  : uiText(bangla, 'auto_mode_off', 'Use the buttons below instead'),
            ),
            value: appState.autoAssist,
            onChanged: (_) => appState.toggleAutoAssist(),
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            secondary: Icon(
              appState.controlGesturesEnabled ? Icons.back_hand : Icons.back_hand_outlined,
              color: appState.controlGesturesEnabled ? AppColors.amber : null,
            ),
            title: Text(uiText(bangla, 'hand_gestures', 'Hand-shape gestures')),
            subtitle: Text(
              appState.controlGesturesEnabled
                  ? uiText(bangla, 'hand_gestures_on',
                      'Fist (thumb out) / thumbs-up / thumbs-down control the sentence')
                  : uiText(bangla, 'hand_gestures_off', 'Use the Delete / Clear / Speak buttons instead'),
            ),
            value: appState.controlGesturesEnabled,
            onChanged: (_) => appState.toggleControlGestures(),
          ),
          const Divider(),
          const SizedBox(height: 8),
          ListTile(
            contentPadding: EdgeInsets.zero,
            leading: const Icon(Icons.info_outline),
            title: Text(uiText(bangla, 'about_this_build', 'About this build')),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const AboutScreen()),
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Text(
        text.toUpperCase(),
        style: const TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.8,
          color: AppColors.primary,
        ),
      ),
    );
  }
}
