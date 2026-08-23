/// Which kind of sign a trained label represents. Both the ASL (words +
/// A-Z fingerspelling) and BdSL (words + digits + alphabet) models mix
/// several of these into one classifier's output, which is exactly what
/// caused them to compete against each other at inference time -- e.g. a
/// held-steady letter losing to a superficially-similar word, or vice
/// versa. [HomeScreenState] restricts the classifier to one active
/// category at a time (see the mode toggle in home_screen.dart) so the
/// two vocabularies stop stepping on each other.
enum SignCategory { word, digit, letter }

/// Every label in this project's `labels.json` files follows the same
/// convention: a bare gloss for a word (e.g. "cat", "brother_in_law"), or
/// a `digit_`/`letter_` prefix for the static-pose classes (see
/// ml/src/dataset_bdsl_static.py, ml/train_asl_combined.py).
SignCategory categoryOf(String label) {
  if (label.startsWith('digit_')) return SignCategory.digit;
  if (label.startsWith('letter_')) return SignCategory.letter;
  return SignCategory.word;
}

extension SignCategoryLabel on SignCategory {
  String get displayName => switch (this) {
        SignCategory.word => 'Words',
        SignCategory.digit => 'Digits',
        SignCategory.letter => 'Letters',
      };
}
