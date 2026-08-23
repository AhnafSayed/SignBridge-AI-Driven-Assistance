import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:provider/provider.dart';

import '../app_theme.dart';
import '../services/asl_vocabulary.dart';
import '../services/bdsl_glossary.dart';
import '../services/sign_category.dart';
import '../state/app_state.dart';

class _DictionaryEntry {
  const _DictionaryEntry({
    required this.term,
    required this.speech,
    required this.category,
    this.unreliable = false,
  });

  final String term;
  final String speech;
  final SignCategory category;
  final bool unreliable;
}

List<_DictionaryEntry> _entriesFor(SignLanguage language) {
  if (language == SignLanguage.asl) {
    return [
      for (final word in aslVocabulary) _DictionaryEntry(term: word, speech: word, category: SignCategory.word),
      for (final digit in aslDigitNames) _DictionaryEntry(term: digit, speech: digit, category: SignCategory.digit),
      for (final letter in aslLetterNames) _DictionaryEntry(term: letter, speech: letter, category: SignCategory.letter),
    ];
  }
  return [
    for (final e in bdslBangla.entries)
      _DictionaryEntry(term: '${e.value}  ${e.key}', speech: e.value, category: SignCategory.word),
    for (var i = 0; i < 10; i++)
      _DictionaryEntry(term: '${bengaliFor('digit_$i')}  $i', speech: bengaliFor('digit_$i'), category: SignCategory.digit),
    for (final name in bdslLetterNames)
      () {
        final glyph = bengaliFor('letter_$name');
        return _DictionaryEntry(
          term: glyph == name ? name : '$glyph  $name',
          speech: glyph,
          category: SignCategory.letter,
          unreliable: bdslUnreliableLetters.contains(name),
        );
      }(),
  ];
}

/// A searchable, tap-to-pronounce reference for every sign this app
/// recognizes -- separate from the live Sign tab, which is about
/// *translating*, not *looking up*. Read-only and stateless beyond its own
/// search/filter/language selection, so it never touches [AppState]'s
/// live recognition language -- browsing the dictionary while signing in
/// the other language should not silently switch what the camera is
/// configured to detect.
class DictionaryScreen extends StatefulWidget {
  const DictionaryScreen({super.key});

  @override
  State<DictionaryScreen> createState() => DictionaryScreenState();
}

class DictionaryScreenState extends State<DictionaryScreen> {
  final _tts = FlutterTts();
  final _searchController = TextEditingController();
  late SignLanguage _language;
  SignCategory? _categoryFilter;
  String _query = '';

  @override
  void initState() {
    super.initState();
    _language = context.read<AppState>().language;
    _tts.setErrorHandler((msg) => debugPrint('Dictionary TTS error: $msg'));
  }

  @override
  void dispose() {
    _tts.stop();
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _speak(String text) async {
    if (text.trim().isEmpty) return;
    final requested = _language == SignLanguage.bdsl ? 'bn-BD' : 'en-US';
    var localeId = requested;
    try {
      final available = await _tts.isLanguageAvailable(requested);
      if (available != true) localeId = 'en-US';
    } catch (_) {
      // Fall through and try the requested locale anyway.
    }
    await _tts.setLanguage(localeId);
    await _tts.setSpeechRate(0.45);
    await _tts.speak(text);
  }

  @override
  Widget build(BuildContext context) {
    final all = _entriesFor(_language);
    final available = _language.availableCategories;
    final query = _query.trim().toLowerCase();
    final filtered = all.where((e) {
      if (_categoryFilter != null && e.category != _categoryFilter) return false;
      if (query.isEmpty) return true;
      return e.term.toLowerCase().contains(query);
    }).toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Dictionary'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(56),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 10),
            child: SegmentedButton<SignLanguage>(
              segments: const [
                ButtonSegment(value: SignLanguage.asl, label: Text('ASL · English')),
                ButtonSegment(value: SignLanguage.bdsl, label: Text('BdSL · বাংলা')),
              ],
              selected: {_language},
              onSelectionChanged: (s) => setState(() {
                _language = s.first;
                _categoryFilter = null;
              }),
            ),
          ),
        ),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: TextField(
              controller: _searchController,
              onChanged: (v) => setState(() => _query = v),
              decoration: InputDecoration(
                hintText: 'Search the ${_language == SignLanguage.asl ? 'ASL' : 'BdSL'} dictionary',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: _query.isEmpty
                    ? null
                    : IconButton(
                        icon: const Icon(Icons.close),
                        onPressed: () => setState(() {
                          _query = '';
                          _searchController.clear();
                        }),
                      ),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                isDense: true,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _FilterChip(
                    label: 'All (${all.length})',
                    selected: _categoryFilter == null,
                    onTap: () => setState(() => _categoryFilter = null),
                  ),
                  for (final category in available)
                    Padding(
                      padding: const EdgeInsets.only(left: 8),
                      child: _FilterChip(
                        label: category.displayName,
                        selected: _categoryFilter == category,
                        onTap: () => setState(() => _categoryFilter = category),
                      ),
                    ),
                ],
              ),
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: filtered.isEmpty
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24),
                      child: Text(
                        'No signs match "$query"',
                        style: const TextStyle(color: AppColors.inkSoft),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  )
                : GridView.builder(
                    padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
                    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      mainAxisExtent: 64,
                      crossAxisSpacing: 8,
                      mainAxisSpacing: 8,
                    ),
                    itemCount: filtered.length,
                    itemBuilder: (context, i) => _DictionaryCard(
                      entry: filtered[i],
                      onSpeak: () => _speak(filtered[i].speech),
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}

/// A dictionary entry as a compact card, two per row -- denser and easier
/// to scan than one row per entry, especially for the alphabet/digit
/// lists where each term is a single glyph. Letters and digits get a
/// round accent badge showing the glyph itself, since that glyph *is*
/// the memorable part; words get a generic translate icon instead, since
/// a multi-word gloss has no single character worth pulling out.
class _DictionaryCard extends StatelessWidget {
  const _DictionaryCard({required this.entry, required this.onSpeak});
  final _DictionaryEntry entry;
  final VoidCallback onSpeak;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final badgeText = entry.category == SignCategory.word ? null : entry.term.characters.first;
    return Material(
      color: scheme.surfaceContainerHighest,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onSpeak,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
          child: Row(
            children: [
              CircleAvatar(
                radius: 18,
                backgroundColor: entry.unreliable
                    ? AppColors.inkSoft.withValues(alpha: 0.25)
                    : AppColors.primary.withValues(alpha: 0.16),
                child: badgeText != null
                    ? Text(badgeText, style: const TextStyle(fontWeight: FontWeight.w700))
                    : const Icon(Icons.translate, size: 18, color: AppColors.primary),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  entry.term,
                  style: const TextStyle(fontSize: 13.5),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              IconButton(
                icon: const Icon(Icons.volume_up_outlined, size: 20),
                tooltip: 'Pronounce',
                onPressed: onSpeak,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FilterChip extends StatelessWidget {
  const _FilterChip({required this.label, required this.selected, required this.onTap});
  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return ChoiceChip(label: Text(label), selected: selected, onSelected: (_) => onTap());
  }
}
