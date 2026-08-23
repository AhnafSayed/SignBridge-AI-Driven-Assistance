import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:sign_bridge/services/word_completer.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('English completer loads and suggests', () async {
    final c = await WordCompleter.loadEnglish();
    final s = c.suggest('hel');
    expect(s, isNotEmpty);
    expect(s.first.startsWith('hel'), isTrue);
  });

  test('Bangla completer loads and suggests', () async {
    final c = await WordCompleter.loadBangla();
    // "আ" (a common Bangla prefix -- আমি, আমার, আছে, etc. are all near the
    // top of the frequency list) should have real matches.
    final s = c.suggest('আ');
    expect(s, isNotEmpty, reason: 'expected real Bangla suggestions for prefix আ');
    for (final word in s) {
      expect(word.startsWith('আ'), isTrue, reason: 'suggestion "$word" does not start with আ');
    }
  });

  test('Bangla completer: exact known word prefix', () async {
    final c = await WordCompleter.loadBangla();
    final s = c.suggest('আমি');
    expect(s, contains('আমি'));
  });

  test('default suggestion count is 6, not the old 3', () async {
    // Regression guard for the "show more suggestions" fix: the row is a
    // horizontally-scrollable strip with no fixed-slot layout constraint,
    // so there was no reason to cap this tightly, and a common short
    // prefix like "th" has far more than 6 real English matches to draw
    // from.
    final c = await WordCompleter.loadEnglish();
    final s = c.suggest('th');
    expect(s.length, 6);
  });

  test('suggestions stay frequency-ordered as the cap is raised', () async {
    // The ranking guarantee the user actually asked for: raising the cap
    // must surface *more* common words, not reorder them. Since `_words`
    // is frequency-ordered from the source list and suggest() is a linear
    // prefix scan, each match's index in the full word list must be
    // non-decreasing across the returned suggestions.
    final c = await WordCompleter.loadEnglish();
    final all = await rootBundle.loadString('assets/dictionary/common_words.txt');
    final ranked = all.split('\n').map((w) => w.trim()).where((w) => w.isNotEmpty).toList();
    final s = c.suggest('th', maxResults: 6);
    final ranks = s.map(ranked.indexOf).toList();
    for (var i = 1; i < ranks.length; i++) {
      expect(ranks[i], greaterThan(ranks[i - 1]),
          reason: 'suggestion "${s[i]}" is not less common than "${s[i - 1]}"');
    }
  });
}
