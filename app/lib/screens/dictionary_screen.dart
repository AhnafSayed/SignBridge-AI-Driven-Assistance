import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
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
    required this.demoKey,
    this.unreliable = false,
  });

  final String term;
  final String speech;
  final SignCategory category;
  final bool unreliable;

  /// Matches the exact filename stem (no extension) used when the demo
  /// images and captions.json were generated -- e.g. `asl_letter_A`,
  /// `bdsl_word_father`. Not every entry has a real file behind this key
  /// (only 14 of ASL's 250 words do, since the full training corpus for
  /// the rest isn't available on the machine that generated these -- see
  /// `app/assets/dictionary_demos/captions.json`'s own coverage), so
  /// lookups against it must degrade gracefully rather than assume a hit.
  final String demoKey;
}

List<_DictionaryEntry> _entriesFor(SignLanguage language) {
  if (language == SignLanguage.asl) {
    return [
      for (final word in aslVocabulary)
        _DictionaryEntry(term: word, speech: word, category: SignCategory.word, demoKey: 'asl_word_$word'),
      for (final digit in aslDigitNames)
        _DictionaryEntry(term: digit, speech: digit, category: SignCategory.digit, demoKey: 'asl_digit_$digit'),
      for (final letter in aslLetterNames)
        _DictionaryEntry(term: letter, speech: letter, category: SignCategory.letter, demoKey: 'asl_letter_$letter'),
    ];
  }
  return [
    for (final e in bdslBangla.entries)
      _DictionaryEntry(
        term: '${e.value}  ${e.key}',
        speech: e.value,
        category: SignCategory.word,
        demoKey: 'bdsl_word_${e.key}',
      ),
    for (var i = 0; i < 10; i++)
      _DictionaryEntry(
        term: '${bengaliFor('digit_$i')}  $i',
        speech: bengaliFor('digit_$i'),
        category: SignCategory.digit,
        demoKey: 'bdsl_digit_$i',
      ),
    for (final name in bdslLetterNames)
      () {
        final glyph = bengaliFor('letter_$name');
        return _DictionaryEntry(
          term: glyph == name ? name : '$glyph  $name',
          speech: glyph,
          category: SignCategory.letter,
          demoKey: 'bdsl_letter_$name',
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

  /// Loaded once, not per card build -- 157 entries is small enough that
  /// keeping the whole caption map in memory is cheaper than re-parsing
  /// JSON on every scroll frame. Empty until the async load in initState
  /// finishes, which is fine: cards just show no caption for a moment
  /// rather than blocking the screen on a file read.
  Map<String, String> _captions = const {};

  @override
  void initState() {
    super.initState();
    _language = context.read<AppState>().language;
    _tts.setErrorHandler((msg) => debugPrint('Dictionary TTS error: $msg'));
    _loadCaptions();
  }

  Future<void> _loadCaptions() async {
    try {
      final raw = await rootBundle.loadString('assets/dictionary_demos/captions.json');
      final decoded = jsonDecode(raw) as Map<String, dynamic>;
      if (mounted) setState(() => _captions = decoded.map((k, v) => MapEntry(k, v as String)));
    } catch (e) {
      // No captions is a degraded-but-fine state (plain entries, no demo
      // image/caption) -- not worth surfacing to the user as an error.
      debugPrint('Dictionary: captions.json unavailable: $e');
    }
  }

  @override
  void dispose() {
    _tts.stop();
    _searchController.dispose();
    super.dispose();
  }

  /// Every card shows a description, even the ~236 ASL words this build
  /// has no rendered demo image for (see [_DictionaryEntry.demoKey]) --
  /// a card with a picture and no caption next to a card with neither
  /// reads as broken, not as "nothing to show here yet."
  String _captionFor(_DictionaryEntry entry) {
    final generated = _captions[entry.demoKey];
    if (generated != null) return generated;
    return switch (entry.category) {
      SignCategory.word => 'No demo image available yet for this word sign.',
      SignCategory.digit => 'No demo image available yet for this digit sign.',
      SignCategory.letter => 'No demo image available yet for this letter sign.',
    };
  }

  void _openPreview(_DictionaryEntry entry, String caption) {
    showDialog<void>(
      context: context,
      builder: (context) => _DemoImagePreview(entry: entry, caption: caption),
    );
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
                : ListView.separated(
                    padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
                    itemCount: filtered.length,
                    separatorBuilder: (_, _) => const SizedBox(height: 8),
                    itemBuilder: (context, i) {
                      final entry = filtered[i];
                      final caption = _captionFor(entry);
                      return _DictionaryCard(
                        entry: entry,
                        caption: caption,
                        onSpeak: () => _speak(entry.speech),
                        onOpenPreview: () => _openPreview(entry, caption),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

/// The full-size demo image + caption shown on tapping a Dictionary row's
/// thumbnail -- a centered dialog card, sized to the image's own aspect
/// ratio (BoxFit.contain, not cover) so the whole image is always visible
/// with no cropping, and pinch-zoomable via InteractiveViewer for a closer
/// look at any part of it.
class _DemoImagePreview extends StatelessWidget {
  const _DemoImagePreview({required this.entry, required this.caption});

  final _DictionaryEntry entry;
  final String caption;

  @override
  Widget build(BuildContext context) {
    return Dialog(
      insetPadding: const EdgeInsets.all(20),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(entry.term, style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
                ),
                IconButton(icon: const Icon(Icons.close), onPressed: () => Navigator.of(context).pop()),
              ],
            ),
            Flexible(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: InteractiveViewer(
                  maxScale: 4,
                  child: Image.asset(
                    'assets/dictionary_demos/${entry.demoKey}.png',
                    fit: BoxFit.contain,
                    filterQuality: FilterQuality.high,
                    errorBuilder: (context, error, stack) => const Padding(
                      padding: EdgeInsets.symmetric(vertical: 32),
                      child: Icon(Icons.image_not_supported_outlined, size: 48, color: AppColors.inkSoft),
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Text(caption, style: const TextStyle(height: 1.4)),
          ],
        ),
      ),
    );
  }
}

/// A dictionary entry as a row: a demo thumbnail on the left (a rendered
/// hand-skeleton diagram where one exists -- see captions.json's own
/// coverage notes for which entries do), the term and a short handshape
/// caption in the middle, and a speak button on the right. Letters/digits
/// fall back to a round glyph badge (matching the pre-image design) when
/// no demo image is bundled for that specific entry; words fall back to a
/// generic icon. Either way, a missing image is a quieter version of the
/// same card, never a broken image icon or an empty gap.
class _DictionaryCard extends StatelessWidget {
  const _DictionaryCard({
    required this.entry,
    required this.caption,
    required this.onSpeak,
    required this.onOpenPreview,
  });
  final _DictionaryEntry entry;
  final String caption;
  final VoidCallback onSpeak;
  final VoidCallback onOpenPreview;

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
          padding: const EdgeInsets.all(10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Material(
                color: Colors.transparent,
                borderRadius: BorderRadius.circular(12),
                clipBehavior: Clip.antiAlias,
                child: InkWell(
                  onTap: onOpenPreview,
                  child: SizedBox(
                    width: 84,
                    height: 84,
                    child: Image.asset(
                      'assets/dictionary_demos/${entry.demoKey}.png',
                      fit: BoxFit.cover,
                      filterQuality: FilterQuality.high,
                      errorBuilder: (context, error, stack) => Center(
                        child: CircleAvatar(
                          radius: 20,
                          backgroundColor: entry.unreliable
                              ? AppColors.inkSoft.withValues(alpha: 0.25)
                              : AppColors.primary.withValues(alpha: 0.16),
                          child: badgeText != null
                              ? Text(badgeText, style: const TextStyle(fontWeight: FontWeight.w700))
                              : const Icon(Icons.translate, size: 18, color: AppColors.primary),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      entry.term,
                      style: const TextStyle(fontSize: 14.5, fontWeight: FontWeight.w600),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    if (entry.unreliable)
                      const Padding(
                        padding: EdgeInsets.only(top: 2),
                        child: Text(
                          'Not yet reliably recognized',
                          style: TextStyle(fontSize: 11, color: AppColors.coral),
                        ),
                      ),
                    Padding(
                      padding: const EdgeInsets.only(top: 3),
                      child: Text(
                        caption,
                        style: const TextStyle(fontSize: 12, color: AppColors.inkSoft, height: 1.3),
                        maxLines: 3,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
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
