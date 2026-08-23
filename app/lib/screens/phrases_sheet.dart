import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app_theme.dart';
import '../services/phrase_store.dart';
import '../services/quick_phrases.dart';
import '../state/app_state.dart';

/// Bottom sheet listing the user's saved phrases and recent utterances,
/// with tap-to-reuse.
///
/// Presented as a sheet rather than a separate route because it is used
/// *mid-conversation*: the user should be able to reach a stock phrase,
/// send it, and get back to the camera without losing their place. A
/// full-screen route would tear down the camera preview (see RootShell's
/// pause/resume handling) for what is meant to be a two-second detour.
class PhrasesSheet extends StatelessWidget {
  const PhrasesSheet({
    super.key,
    required this.currentSentence,
    required this.onUsePhrase,
    required this.onSpeakPhrase,
  });

  /// The sentence currently under construction, offered for saving. Empty
  /// when there is nothing to save.
  final String currentSentence;

  /// Load a phrase into the sentence area so the user can edit or extend
  /// it before speaking.
  final ValueChanged<String> onUsePhrase;

  /// Speak a phrase immediately -- the fast path, and the reason this
  /// feature exists: a stock phrase should be one tap from being said.
  final ValueChanged<String> onSpeakPhrase;

  static Future<void> show(
    BuildContext context, {
    required String currentSentence,
    required ValueChanged<String> onUsePhrase,
    required ValueChanged<String> onSpeakPhrase,
  }) {
    return showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => PhrasesSheet(
        currentSentence: currentSentence,
        onUsePhrase: onUsePhrase,
        onSpeakPhrase: onSpeakPhrase,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();
    final store = context.watch<PhraseStore>();
    final lang = appState.language;
    final saved = store.savedFor(lang);
    final recent = store.recentFor(lang);
    final scheme = Theme.of(context).colorScheme;

    return SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(maxHeight: MediaQuery.of(context).size.height * 0.75),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 12, 4),
              child: Row(
                children: [
                  Expanded(
                    child: Text('My phrases',
                        style: Theme.of(context).textTheme.titleLarge),
                  ),
                  if (saved.isNotEmpty || recent.isNotEmpty)
                    TextButton(
                      onPressed: () => _confirmClearAll(context, store),
                      child: const Text('Clear all'),
                    ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 10),
              child: Text(
                'Saved on this device only — never uploaded.',
                style: TextStyle(fontSize: 12.5, color: AppColors.inkSoft),
              ),
            ),
            if (!store.isReady)
              const Padding(
                padding: EdgeInsets.all(28),
                child: Center(child: CircularProgressIndicator()),
              )
            else
              Flexible(
                child: ListView(
                  shrinkWrap: true,
                  padding: const EdgeInsets.only(bottom: 12),
                  children: [
                    if (currentSentence.trim().isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                        child: FilledButton.tonalIcon(
                          onPressed: store.isSaved(lang, currentSentence)
                              ? null
                              : () async {
                                  await store.savePhrase(lang, currentSentence);
                                  if (context.mounted) {
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      const SnackBar(content: Text('Phrase saved')),
                                    );
                                  }
                                },
                          icon: const Icon(Icons.bookmark_add_outlined, size: 20),
                          label: Text(
                            store.isSaved(lang, currentSentence)
                                ? 'Already saved'
                                : 'Save: "$currentSentence"',
                            overflow: TextOverflow.ellipsis,
                          ),
                          style: FilledButton.styleFrom(
                            minimumSize: const Size(double.infinity, 48),
                            alignment: Alignment.centerLeft,
                          ),
                        ),
                      ),
                    if (saved.isEmpty && recent.isEmpty)
                      Padding(
                        padding: const EdgeInsets.fromLTRB(20, 24, 20, 32),
                        child: Text(
                          'Nothing saved yet.\n\nBuild a sentence, then open this panel to '
                          'save it. Phrases you speak also appear here automatically, so a '
                          'sentence you use often only has to be signed once.\n\n'
                          'In the meantime, "Quick phrases" below has common phrases ready '
                          'to use right away.',
                          style: TextStyle(color: AppColors.inkSoft, height: 1.5),
                        ),
                      ),
                    if (saved.isNotEmpty) ...[
                      _SectionLabel('Saved (${saved.length})'),
                      for (final phrase in saved)
                        _PhraseTile(
                          phrase: phrase,
                          leading: Icon(Icons.bookmark, color: scheme.primary, size: 20),
                          onUse: () {
                            Navigator.of(context).pop();
                            onUsePhrase(phrase);
                          },
                          onSpeak: () => onSpeakPhrase(phrase),
                          onRemove: () => store.removeSaved(lang, phrase),
                        ),
                    ],
                    if (recent.isNotEmpty) ...[
                      Row(
                        children: [
                          Expanded(child: _SectionLabel('Recently spoken (${recent.length})')),
                          Padding(
                            padding: const EdgeInsets.only(right: 12),
                            child: TextButton(
                              onPressed: () => store.clearRecent(lang),
                              child: const Text('Clear'),
                            ),
                          ),
                        ],
                      ),
                      for (final phrase in recent)
                        _PhraseTile(
                          phrase: phrase,
                          leading: Icon(Icons.history, color: AppColors.inkSoft, size: 20),
                          onUse: () {
                            Navigator.of(context).pop();
                            onUsePhrase(phrase);
                          },
                          onSpeak: () => onSpeakPhrase(phrase),
                          onSave: store.isSaved(lang, phrase)
                              ? null
                              : () => store.savePhrase(lang, phrase),
                        ),
                    ],
                    _SectionLabel('Quick phrases'),
                    for (final (i, category) in quickPhrasesFor(lang).indexed)
                      ExpansionTile(
                        // A distinct color + icon per category (by
                        // position, not name -- the Bangla category names
                        // are different strings than the English ones at
                        // the same index) so each group reads as its own
                        // identity at a glance, the same idea the
                        // reference app's colorful category chips use,
                        // built from a separate, muted palette rather
                        // than copying its saturated pink/green/amber/red.
                        leading: CircleAvatar(
                          radius: 15,
                          backgroundColor: _quickPhraseColors[i % _quickPhraseColors.length]
                              .withValues(alpha: 0.18),
                          child: Icon(
                            _quickPhraseIcons[i % _quickPhraseIcons.length],
                            size: 16,
                            color: _quickPhraseColors[i % _quickPhraseColors.length],
                          ),
                        ),
                        title: Text(
                          category.name,
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            color: _quickPhraseColors[i % _quickPhraseColors.length],
                          ),
                        ),
                        tilePadding: const EdgeInsets.only(left: 12, right: 16),
                        childrenPadding: EdgeInsets.zero,
                        children: [
                          for (final phrase in category.phrases)
                            _PhraseTile(
                              phrase: phrase,
                              leading: Icon(
                                Icons.flash_on,
                                color: _quickPhraseColors[i % _quickPhraseColors.length],
                                size: 20,
                              ),
                              onUse: () {
                                Navigator.of(context).pop();
                                onUsePhrase(phrase);
                              },
                              onSpeak: () => onSpeakPhrase(phrase),
                              onSave: store.isSaved(lang, phrase)
                                  ? null
                                  : () => store.savePhrase(lang, phrase),
                            ),
                        ],
                      ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  // A muted, earthy quartet -- deliberately not the reference app's
  // saturated pink/green/amber/red -- indexed by category position so it
  // lines up the same way across English and Bangla lists (their names
  // differ, their order doesn't). Skips AppColors.amber (already means
  // "recognized sign" elsewhere) and AppColors.primary (already the
  // app's own chrome color); Medical reuses AppColors.coral, this app's
  // own established "this matters" color from the SOS feature, rather
  // than introducing a second, competing red.
  static const _quickPhraseColors = [
    Color(0xFFC97B4A), // Greetings -- warm terracotta
    Color(0xFF4A8B82), // Needs -- muted teal
    AppColors.coral, // Medical
    Color(0xFF8C7AA6), // Feelings -- dusty plum
  ];
  static const _quickPhraseIcons = [
    Icons.waving_hand_outlined,
    Icons.help_outline,
    Icons.medical_services_outlined,
    Icons.mood_outlined,
  ];

  Future<void> _confirmClearAll(BuildContext context, PhraseStore store) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete all phrases?'),
        content: const Text(
          'This removes every saved phrase and your spoken history for both '
          'languages. It cannot be undone.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(ctx).pop(false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.of(ctx).pop(true), child: const Text('Delete all')),
        ],
      ),
    );
    if (ok == true) await store.clearAll();
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 14, 20, 6),
      child: Text(
        text.toUpperCase(),
        style: const TextStyle(
          fontSize: 11.5,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.7,
          color: AppColors.inkSoft,
        ),
      ),
    );
  }
}

class _PhraseTile extends StatelessWidget {
  const _PhraseTile({
    required this.phrase,
    required this.leading,
    required this.onUse,
    required this.onSpeak,
    this.onRemove,
    this.onSave,
  });

  final String phrase;
  final Widget leading;
  final VoidCallback onUse;
  final VoidCallback onSpeak;
  final VoidCallback? onRemove;
  final VoidCallback? onSave;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: leading,
      title: Text(phrase, style: const TextStyle(fontSize: 16, height: 1.3)),
      // Tapping the row loads the phrase; speaking is a separate explicit
      // button. Making the whole row speak would fire audio on an
      // accidental brush, which is a bad failure for a tool used
      // mid-conversation.
      onTap: onUse,
      contentPadding: const EdgeInsets.only(left: 20, right: 6),
      minVerticalPadding: 12,
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (onSave != null)
            IconButton(
              icon: const Icon(Icons.bookmark_add_outlined),
              tooltip: 'Save this phrase',
              onPressed: onSave,
            ),
          if (onRemove != null)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Remove',
              onPressed: onRemove,
            ),
          IconButton.filledTonal(
            icon: const Icon(Icons.volume_up_rounded),
            tooltip: 'Speak now',
            onPressed: onSpeak,
          ),
        ],
      ),
    );
  }
}
