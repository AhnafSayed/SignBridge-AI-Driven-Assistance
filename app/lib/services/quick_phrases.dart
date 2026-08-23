import '../state/app_state.dart';

/// Pre-built starter phrases, grouped by category -- distinct from
/// [PhraseStore]'s saved/recent lists, which only ever hold what a user
/// has actually said. These exist for the opposite moment: someone who
/// has never used the app yet, or who needs a common phrase *right now*
/// and hasn't built up any history to draw on. A phrase tapped here can
/// still be saved into the user's own list via the same [PhraseStore]
/// used everywhere else -- this is a starting set, not a separate system.
class QuickPhraseCategory {
  const QuickPhraseCategory(this.name, this.phrases);
  final String name;
  final List<String> phrases;
}

const List<QuickPhraseCategory> _englishQuickPhrases = [
  QuickPhraseCategory('Greetings', [
    'Hello, nice to meet you',
    'Thank you very much',
    'Please',
    'Sorry, I need a moment',
    'Goodbye',
  ]),
  QuickPhraseCategory('Needs', [
    'I need help',
    'Can you repeat that please',
    'I do not understand',
    'Where is the bathroom',
    'I am hungry',
    'I am thirsty',
  ]),
  QuickPhraseCategory('Medical', [
    'I need a doctor',
    'I am in pain',
    'I feel dizzy',
    'I have an allergy',
    'Please call my family',
  ]),
  QuickPhraseCategory('Feelings', [
    'I am happy',
    'I am tired',
    'I am worried',
    'I am fine, thank you',
  ]),
];

const List<QuickPhraseCategory> _banglaQuickPhrases = [
  QuickPhraseCategory('সালাম ও শুভেচ্ছা', [
    'হ্যালো, আপনার সাথে দেখা করে ভালো লাগলো',
    'ধন্যবাদ',
    'দয়া করে',
    'দুঃখিত, একটু সময় দিন',
    'বিদায়',
  ]),
  QuickPhraseCategory('প্রয়োজন', [
    'আমার সাহায্য প্রয়োজন',
    'আবার বলবেন দয়া করে',
    'আমি বুঝতে পারছি না',
    'বাথরুম কোথায়',
    'আমার ক্ষুধা লেগেছে',
    'আমার পিপাসা লেগেছে',
  ]),
  QuickPhraseCategory('চিকিৎসা', [
    'আমার ডাক্তার প্রয়োজন',
    'আমার ব্যথা হচ্ছে',
    'আমার মাথা ঘুরছে',
    'আমার এলার্জি আছে',
    'দয়া করে আমার পরিবারকে ফোন করুন',
  ]),
  QuickPhraseCategory('অনুভূতি', [
    'আমি খুশি',
    'আমি ক্লান্ত',
    'আমি চিন্তিত',
    'আমি ভালো আছি, ধন্যবাদ',
  ]),
];

List<QuickPhraseCategory> quickPhrasesFor(SignLanguage language) =>
    language == SignLanguage.bdsl ? _banglaQuickPhrases : _englishQuickPhrases;
