import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../state/app_state.dart';

/// Device-local storage for phrases the user wants back: ones they
/// deliberately saved, and ones they recently spoke.
///
/// Why this exists at all. Building a sentence sign-by-sign is slow, and
/// the sentences people actually need in daily life repeat constantly --
/// "thank you", "where is the bathroom", "please call my family". Forcing
/// a user to re-sign an identical phrase every time is the single most
/// obvious inefficiency in the app.
///
/// Why it is local-only, and why that is not a compromise. The project
/// deliberately ships with no accounts and no server (see
/// docs/PROJECT_REPORT.md §3.6): the app's input is continuous video of
/// the user's hands and face, and the strongest privacy guarantee
/// available is that such data is never transmitted at all. Saved phrases
/// are, if anything, *more* sensitive than the video -- they are the
/// things this person most often needs to say, which for a disabled user
/// can include medical and personal details. Syncing them to a server
/// would trade a real privacy guarantee for convenience nobody asked
/// for. [SharedPreferences] keeps them in the app's private storage,
/// removed with the app.
///
/// Phrases are stored *per language*: a Bangla phrase is not a useful
/// suggestion while signing ASL, and mixing the two would produce a list
/// that is half unreadable whichever mode you are in.
class PhraseStore extends ChangeNotifier {
  /// Constructs empty and not yet loaded, so it can be handed to a
  /// provider synchronously at app start; [init] fills it in from disk.
  /// The alternative -- an async factory -- forces every consumer to cope
  /// with a nullable store, which is a lot of nullability to spread
  /// through the UI for one convenience feature.
  PhraseStore() {
    for (final lang in SignLanguage.values) {
      _saved[lang] = <String>[];
      _recent[lang] = <String>[];
    }
  }

  static const int maxRecent = 20;
  static const int maxSaved = 50;

  SharedPreferences? _prefs;
  final Map<SignLanguage, List<String>> _saved = {};
  final Map<SignLanguage, List<String>> _recent = {};

  /// False until [init] completes. The UI disables phrase controls while
  /// false rather than showing an empty list, so "still loading" is never
  /// mistaken for "you have no saved phrases".
  bool get isReady => _prefs != null;

  static String _savedKey(SignLanguage l) => 'phrases_saved_${l.name}';
  static String _recentKey(SignLanguage l) => 'phrases_recent_${l.name}';

  /// Loads persisted phrases. Never throws: a storage failure degrades to
  /// "no saved phrases" rather than taking down the recognition screen,
  /// which is the app's actual purpose.
  Future<void> init() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      for (final lang in SignLanguage.values) {
        _saved[lang] = prefs.getStringList(_savedKey(lang)) ?? <String>[];
        _recent[lang] = prefs.getStringList(_recentKey(lang)) ?? <String>[];
      }
      _prefs = prefs;
      notifyListeners();
    } catch (e) {
      debugPrint('PhraseStore: storage unavailable, phrases disabled: $e');
    }
  }

  List<String> savedFor(SignLanguage language) => List.unmodifiable(_saved[language] ?? const []);
  List<String> recentFor(SignLanguage language) => List.unmodifiable(_recent[language] ?? const []);

  bool isSaved(SignLanguage language, String phrase) =>
      (_saved[language] ?? const []).contains(phrase.trim());

  /// Pins a phrase. Most-recently-saved first, since a phrase just saved
  /// is the one most likely to be wanted next.
  Future<void> savePhrase(SignLanguage language, String phrase) async {
    final text = phrase.trim();
    if (text.isEmpty) return;
    final list = _saved[language]!;
    list.remove(text); // re-saving moves it to the top rather than duplicating
    list.insert(0, text);
    if (list.length > maxSaved) list.removeRange(maxSaved, list.length);
    await _persist(_savedKey(language), list);
  }

  Future<void> removeSaved(SignLanguage language, String phrase) async {
    if (!_saved[language]!.remove(phrase.trim())) return;
    await _persist(_savedKey(language), _saved[language]!);
  }

  /// Records a spoken sentence in history. Called on speak rather than on
  /// every committed word, so history holds complete utterances rather
  /// than every partial state the sentence passed through.
  Future<void> recordSpoken(SignLanguage language, String phrase) async {
    final text = phrase.trim();
    if (text.isEmpty) return;
    final list = _recent[language]!;
    list.remove(text); // repeating a phrase moves it up, never duplicates it
    list.insert(0, text);
    if (list.length > maxRecent) list.removeRange(maxRecent, list.length);
    await _persist(_recentKey(language), list);
  }

  Future<void> clearRecent(SignLanguage language) async {
    if (_recent[language]!.isEmpty) return;
    _recent[language]!.clear();
    await _persist(_recentKey(language), _recent[language]!);
  }

  /// Erases everything for every language -- the user-facing "forget all my
  /// phrases" action. Present because a tool that remembers what you say
  /// must also let you make it stop.
  Future<void> clearAll() async {
    for (final lang in SignLanguage.values) {
      _saved[lang]!.clear();
      _recent[lang]!.clear();
      try {
        await _prefs?.remove(_savedKey(lang));
        await _prefs?.remove(_recentKey(lang));
      } catch (e) {
        debugPrint('PhraseStore: failed to clear $lang: $e');
      }
    }
    notifyListeners();
  }

  Future<void> _persist(String key, List<String> value) async {
    try {
      await _prefs?.setStringList(key, value);
    } catch (e) {
      // The in-memory list is already updated, so the phrase still works
      // for this session; only durability is lost. Failing loudly here
      // would be worse than continuing.
      debugPrint('PhraseStore: failed to persist $key: $e');
    }
    notifyListeners();
  }
}
