import 'package:flutter/services.dart' show rootBundle;

/// Gboard-style offline prefix completion for fingerspelling. As letters
/// are spelled one at a time, this suggests likely full-word completions
/// so the user doesn't have to spell every letter of a long word --
/// tapping a suggestion commits the whole word at once.
///
/// One bundled, frequency-ordered word list per language, both offline,
/// no network call ever:
///  - English (ASL): 10,000 most common words, MIT-licensed, from
///    first20hours/google-10000-english.
///  - Bangla (BdSL): 10,000 most common words, derived from
///    hermitdave/FrequencyWords' `bn` (Bengali) OpenSubtitles-based list
///    (code MIT, word-list content CC BY-SA 4.0 -- attribution required
///    if this app is ever distributed publicly, same as BdSLW60's
///    CC BY-NC-ND flag elsewhere in this project). Filtered to
///    Bengali-script tokens only -- the raw source (subtitle transcripts)
///    has real English-word contamination (movie credits, character
///    names) mixed in, stripped out here rather than shipped as-is.
///
/// Both lists are real, verifiable, independently-sourced word data, not
/// guessed-at entries -- the same discipline this project applies to every
/// other dataset (see ml/README.md).
///
/// Because BdSL fingerspelling signs only cover Bangla's *independent*
/// letter forms (see bdsl_glossary.dart's `_bdslLetterGlyphs`), not the
/// dependent vowel signs (matra) Bangla orthography normally uses when a
/// vowel follows a consonant mid-word, a spelled-out prefix won't always
/// character-match a real word's actual spelling. Suggestions still only
/// ever show real dictionary words (never fabricated), so this shows up
/// as "fewer matches for some words," not as wrong ones.
class WordCompleter {
  WordCompleter._(this._words);

  // Frequency-ordered already (the source list's own order), so the first
  // N prefix matches are already the most likely candidates -- no need to
  // re-rank.
  final List<String> _words;

  static Future<WordCompleter> load(String assetPath) async {
    final raw = await rootBundle.loadString(assetPath);
    final words = raw.split('\n').map((w) => w.trim()).where((w) => w.isNotEmpty).toList();
    return WordCompleter._(words);
  }

  static Future<WordCompleter> loadEnglish() => load('assets/dictionary/common_words.txt');
  static Future<WordCompleter> loadBangla() => load('assets/dictionary/common_words_bn.txt');

  // Raised from 3: the suggestion row (HomeScreen's fixed-height,
  // horizontally-scrollable ActionChip strip) isn't a fixed 3-slot bar
  // like Gboard's, so there was no layout reason for the tight cap --
  // it only meant fewer real completions were ever offered per keystroke.
  // Matches are still frequency-ranked for free: `_words` is already
  // frequency-ordered from the source list, and a linear prefix scan over
  // an already-sorted list yields its matches in that same order, so a
  // higher cap surfaces more common words rather than less common ones.
  List<String> suggest(String prefix, {int maxResults = 6}) {
    if (prefix.isEmpty) return const [];
    final lower = prefix.toLowerCase();
    final matches = <String>[];
    for (final word in _words) {
      if (word.startsWith(lower)) {
        matches.add(word);
        if (matches.length >= maxResults) break;
      }
    }
    return matches;
  }
}
