import 'package:flutter/material.dart';

import '../app_theme.dart';
import '../widgets/brand_title.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const BrandTitle('About')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Text(
            'SignBridge',
            style: Theme.of(context).textTheme.headlineMedium,
          ),
          const SizedBox(height: 4),
          const Text(
            'AI-Driven Assistance for Hearing & Speech Impairments',
            style: TextStyle(color: AppColors.inkSoft),
          ),
          const SizedBox(height: 20),
          const Padding(
            padding: EdgeInsets.only(bottom: 6),
            child: Text(
              'PROJECT',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.8,
                color: AppColors.primary,
              ),
            ),
          ),
          const Center(
            child: Padding(
              padding: EdgeInsets.only(bottom: 14),
              child: Image(
                image: AssetImage('assets/branding/bup_logo.webp'),
                height: 96,
              ),
            ),
          ),
          const _ProjectTable(),
          const _Section(
            title: 'How it works',
            body:
                'Every camera frame is passed to MediaPipe\'s hand-landmark '
                'detector (via the hand_landmarker plugin), which returns 21 '
                'hand keypoints in real time, entirely on-device with no '
                'network needed. Those keypoints are handed to a sign '
                'classifier, and the result is shown as text and, '
                'optionally, spoken aloud.',
          ),
          const _Section(
            title: 'Hands-free controls',
            body:
                'With Auto mode on (the switch above the camera), no '
                'touchscreen taps are needed at all. These work the same '
                'way in both ASL and BdSL mode:',
          ),
          const Padding(
            padding: EdgeInsets.only(bottom: 12),
            child: Column(
              children: [
                _GestureRow(emoji: '🤟', label: 'Hold any sign steady', result: 'Adds it to the sentence'),
                _GestureRow(emoji: '🤙', label: 'Hold a fist, thumb out to the side', result: 'Deletes the last word'),
                _GestureRow(emoji: '👎', label: 'Hold thumbs-down', result: 'Clears the whole sentence'),
                _GestureRow(emoji: '👍', label: 'Hold thumbs-up', result: 'Speaks the sentence now'),
                _GestureRow(emoji: '⏸', label: 'Lower your hand and pause', result: 'Sentence is spoken automatically'),
              ],
            ),
          ),
          const _Section(
            title: 'Saved phrases',
            body:
                'Sentences you speak are remembered automatically, and you '
                'can pin the ones you use often -- open the bookmark icon '
                'in the top bar. Tap a phrase to load it into the sentence, '
                'or press its speaker button to say it immediately, so a '
                'sentence you need every day only has to be signed once.\n\n'
                'These stay on this device. They are never uploaded, and '
                'there is no account and no server to upload them to. That '
                'matters more here than for most apps: the phrases you save '
                'are the things you most often need to say, which can be '
                'personal or medical. "Clear all" in that panel erases '
                'everything, both languages, permanently.',
          ),
          const _Section(
            title: 'If hand-shape gestures feel unreliable',
            body:
                'The fist/thumbs-up/thumbs-down gestures above are read '
                'from hand geometry, not from the trained classifier, so '
                'their reliability can vary a little more with lighting or '
                'camera angle than sign recognition itself does. The '
                '"Hand-shape gestures" switch in Settings turns just '
                'those three gestures off -- holding a sign steady to add it and '
                'pausing to auto-speak keep working exactly the same, and '
                'Delete/Clear/Speak fall back to their manual buttons.',
          ),
          const _Section(
            title: 'Current recognition engine',
            body:
                'Both languages run a real trained model -- a compact '
                'on-device neural network over MediaPipe hand-landmark '
                'sequences, tracking both hands so two-handed signs aren\'t '
                'starved of half their signal.\n\n'
                'BdSL recognizes 60 words, 10 digits, and 37 alphabet '
                'letters, at about 83% accuracy on signers it never saw '
                'during training. ASL recognizes 250 words, 10 digits, and '
                'the fingerspelling alphabet; word accuracy on unseen '
                'signers is lower, around 58%, since 250 fine-grained '
                'words from real video is a harder problem than a smaller '
                'vocabulary of clearly distinct handshapes -- digits and '
                'letters in both languages are recognized above 98% on '
                'their own. For context, this is comparable to or better '
                'than published results at this vocabulary size that use '
                'far more compute than a phone has to work with.',
          ),
          const _Section(
            title: 'Recognition modes',
            body:
                'The mode switch above the camera restricts recognition to '
                'just Words, Digits, or the fingerspelling Alphabet, so the '
                'model only ever competes against the vocabulary you\'re '
                'actually signing -- this is what stops a held-steady '
                'letter from losing to a similar-looking word, or vice '
                'versa.\n\n'
                'In Letters mode, spelled letters build one word (C-A-T '
                'stays "CAT", not three separate entries) instead of each '
                'auto-spacing like a whole sign would. An offline '
                'word-completion list (no network, ever) suggests likely '
                'full-word completions as you spell, Gboard-style, for '
                'both languages -- tap one to commit the whole word without '
                'spelling every letter.',
          ),
          const _Section(
            title: 'Credits',
            body:
                'The ASL fingerspelling-letter photos in the Dictionary are '
                'from the "ASL Hand Alphabet Dataset" by shreyagodhani645 on '
                'Kaggle, licensed CC BY 4.0. Every other Dictionary image is '
                'a hand-skeleton diagram this app renders itself from its '
                'own recognition data, not a photo, so it carries no '
                'separate credit.',
          ),
        ],
      ),
    );
  }
}

class _ProjectTable extends StatelessWidget {
  const _ProjectTable();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Table(
        border: TableBorder.all(color: scheme.outlineVariant, borderRadius: BorderRadius.circular(6)),
        columnWidths: const {0: FlexColumnWidth(1.1), 1: FlexColumnWidth(2)},
        children: [
          _ProjectRow('University', 'Bangladesh University of Professionals (BUP)'),
          _ProjectRow('Faculty', 'Faculty of Science and Technology (FST)'),
          _ProjectRow('Department', 'Department of ICT'),
          _ProjectRow('Program', 'MICT-2023'),
          _ProjectRow('Supervisor', 'Dr. Ahmedul Kabir (University of Dhaka)'),
          _ProjectRow('Team member', 'Samiul Islam\nRoll 23549908006 · Reg 109901230006'),
          _ProjectRow('Team member', 'Ahnaf Sayed\nRoll 23549908020 · Reg 109901230020'),
          _ProjectRow('Team member', 'Abu Saeed Sabuj\nRoll 23549908023 · Reg 109901230023'),
        ],
      ),
    );
  }
}

class _ProjectRow extends TableRow {
  _ProjectRow(String label, String value)
      : super(children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            child: Text(label, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13)),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            child: Text(value, style: const TextStyle(fontSize: 13)),
          ),
        ]);
}

class _GestureRow extends StatelessWidget {
  const _GestureRow({required this.emoji, required this.label, required this.result});
  final String emoji;
  final String label;
  final String result;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Text(emoji, style: const TextStyle(fontSize: 24)),
          const SizedBox(width: 14),
          Expanded(
            child: Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
          ),
          const Icon(Icons.arrow_forward, size: 16, color: AppColors.inkSoft),
          const SizedBox(width: 10),
          Expanded(
            flex: 2,
            child: Text(result, style: const TextStyle(color: AppColors.inkSoft)),
          ),
        ],
      ),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.body});
  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title.toUpperCase(),
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.8,
              color: AppColors.primary,
            ),
          ),
          const SizedBox(height: 6),
          Text(body, style: const TextStyle(height: 1.45)),
        ],
      ),
    );
  }
}
