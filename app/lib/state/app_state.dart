import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart' show ThemeMode;

import '../services/sign_category.dart';

enum SignLanguage { asl, bdsl }

extension SignLanguageLabel on SignLanguage {
  String get label => this == SignLanguage.asl ? 'ASL' : 'BdSL';

  String get fullName =>
      this == SignLanguage.asl ? 'American Sign Language' : 'Bangla Sign Language';

  /// Which [SignCategory] modes this language's model actually has classes
  /// for. Both languages now ship all three (words, digits, letters).
  List<SignCategory> get availableCategories =>
      const [SignCategory.word, SignCategory.digit, SignCategory.letter];
}

/// User-facing settings and the sentence being built.
///
/// Deliberately does NOT hold the per-frame hand-tracking result -- that
/// changes many times a second and lives as local widget state in
/// HomeScreen, so we aren't calling notifyListeners() on every camera frame.
class AppState extends ChangeNotifier {
  SignLanguage _language = SignLanguage.asl;
  // Auto mode: hold a sign steady to add it, pause signing to have the
  // sentence spoken -- no taps needed. Off falls back to manual +/Speak
  // buttons. Defaults on since the whole point of this app is that the
  // user's hands, not a touchscreen, do the talking.
  bool _autoAssist = true;
  // Separate from autoAssist: whether the closed-fist/thumbs-up/thumbs-
  // down hand *shapes* (backspace/clear/speak) are checked at all. These
  // rely on a geometric read of the hand that -- unlike the trained
  // word/letter/digit classifiers -- isn't learned from data, so its
  // reliability varies more with lighting, hand shape, and camera angle
  // than actual sign recognition does. Rather than chase a perfect
  // one-size-fits-all heuristic, this is a real off switch: turning it off
  // keeps hold-a-sign-to-add-it and pause-to-speak (both classifier-driven,
  // consistently reliable) working exactly as before, and falls back to
  // the manual Delete/Clear/Speak buttons for those three specific
  // actions instead of trying to read them from a hand shape at all.
  bool _controlGesturesEnabled = true;
  // Defaults to following the OS setting, same as before this existed --
  // MaterialApp always had both a light and a dark ThemeData wired in
  // main.dart, so the app already switched with system dark mode; this
  // just adds an explicit in-app override for a user who wants light or
  // dark regardless of what the phone's OS is set to (e.g. high-contrast
  // light for a low-vision user on a phone whose system default is dark).
  ThemeMode _themeMode = ThemeMode.system;
  // Independent of [_language]: that's which sign language the camera is
  // reading (ASL vs BdSL), not what script the app's own buttons/menus/
  // labels are written in. A BdSL user's family member might read English
  // chrome text fine while signing is in BdSL, or vice versa -- tying the
  // two together would force a choice neither setting actually implies.
  bool _uiBangla = false;
  final List<String> _sentence = [];

  // Which class of sign the classifier should recognize right now -- see
  // sign_category.dart. Restricting to one category at a time is what
  // stops a held-steady letter from losing to a superficially similar
  // word (or vice versa) purely because both live in the same model's
  // output; defaults to Words since that's the primary use case, with an
  // explicit switch to Letters for fingerspelling.
  SignCategory _activeCategory = SignCategory.word;

  SignLanguage get language => _language;
  SignCategory get activeCategory => _activeCategory;
  bool get autoAssist => _autoAssist;
  bool get controlGesturesEnabled => _controlGesturesEnabled;
  ThemeMode get themeMode => _themeMode;
  bool get uiBangla => _uiBangla;
  List<String> get sentence => List.unmodifiable(_sentence);
  String get sentenceText => _sentence.join(' ');

  void setLanguage(SignLanguage value) {
    if (_language == value) return;
    _language = value;
    // e.g. leaving BdSL's Digits mode for ASL, which has no digit_ classes
    // at all -- falling back to Words rather than a mode that would
    // recognize nothing.
    if (!value.availableCategories.contains(_activeCategory)) {
      _activeCategory = SignCategory.word;
    }
    notifyListeners();
  }

  void setActiveCategory(SignCategory value) {
    if (_activeCategory == value) return;
    _activeCategory = value;
    notifyListeners();
  }

  void toggleAutoAssist() {
    _autoAssist = !_autoAssist;
    notifyListeners();
  }

  void toggleControlGestures() {
    _controlGesturesEnabled = !_controlGesturesEnabled;
    notifyListeners();
  }

  void setThemeMode(ThemeMode value) {
    if (_themeMode == value) return;
    _themeMode = value;
    notifyListeners();
  }

  void toggleUiLanguage() {
    _uiBangla = !_uiBangla;
    notifyListeners();
  }

  void appendWord(String word) {
    if (word.isEmpty) return;
    _sentence.add(word);
    notifyListeners();
  }

  void backspace() {
    if (_sentence.isEmpty) return;
    _sentence.removeLast();
    notifyListeners();
  }

  void clearSentence() {
    if (_sentence.isEmpty) return;
    _sentence.clear();
    notifyListeners();
  }

  /// Replaces the whole sentence with a stored phrase (see PhraseStore).
  /// Split on whitespace so the result is a normal word list -- backspace
  /// then removes one word at a time, exactly as it does for a sentence
  /// built by signing, rather than deleting the entire recalled phrase.
  void replaceSentence(String phrase) {
    final words = phrase.trim().split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();
    if (words.isEmpty && _sentence.isEmpty) return;
    _sentence
      ..clear()
      ..addAll(words);
    notifyListeners();
  }
}
