/// English gloss (the model's actual output label, fixed by training order)
/// -> native Bangla script, transcribed from the BdSLW60 dataset's own
/// documentation (ml/data/bdslw60_docs/BdSLW60.pdf). The model's label
/// order must not change without retraining, so this is a pure display/
/// speech lookup layered on top -- never used for classification.
const Map<String, String> bdslBangla = {
  'father': 'বাবা/আব্বা',
  'relatives': 'আত্মীয়',
  'brother': 'ভাই',
  'sister': 'বোন',
  'wife': 'বৌ',
  'paternal_uncle': 'চাচা',
  'paternal_aunt': 'চাচী',
  'grandfather': 'দাদা/নানা',
  'grandmother': 'দাদী/নানী',
  'responsibility': 'দায়িত্ব',
  'brother_in_law': 'দেবর',
  'sisters_husband': 'দুলাভাই',
  'daughter': 'কন্যা',
  'mother': 'মা',
  'mango': 'আম',
  'potato': 'আলু',
  'pineapple': 'আনারস',
  'grapes': 'আঙ্গুর',
  'apple': 'আপেল',
  'biscuits': 'বিস্কুট',
  'jujube': 'বরই',
  'cake': 'কেক',
  'tea': 'চা',
  'rice': 'চাল',
  'sugar': 'চিনি',
  'chips': 'চিপস',
  'chocolate': 'চকলেট',
  'lentils': 'ডাল',
  'button': 'বোতাম',
  'cap': 'টুপি',
  'shawl': 'চাদর',
  'comb': 'চিরুনি',
  'spectacles': 'চশমা',
  'bangles': 'চুড়ি',
  'clip': 'ক্লিপ',
  'cream': 'ক্রিম',
  'data': 'তথ্য',
  'indebted': 'দেনাদার',
  'twin_baby': 'জমজ',
  'shoe': 'জুতা',
  'toothpaste': 'টুথপেস্ট',
  'tshirt': 'টিশার্ট',
  'tubelight': 'টিউবলাইট',
  'television': 'টিভি',
  'air_conditioner': 'এসি',
  'apartment': 'অ্যাপার্টমেন্ট',
  'audio_cassette': 'অডিও ক্যাসেট',
  'mirror': 'আয়না',
  'water_bucket': 'বালতি',
  'sand': 'বালু',
  'aids': 'এইডস',
  'arthritis': 'বাত',
  'bandage': 'ব্যান্ডেজ',
  'capsule': 'ক্যাপসুল',
  'treatment': 'চিকিৎসা',
  'conjunctivitis': 'চোখ ওঠা',
  'dengue': 'ডেঙ্গু',
  'doctor': 'ডক্টর',
  'bite': 'দংশন',
  'weak': 'দূর্বল',
};

/// Bangla digit glyphs (০-৯), from the combined model's `digit_0`..`digit_9`
/// labels (ml/src/dataset_bdsl_static.py). Unambiguous -- standard Unicode
/// Bengali digits, no transcription risk.
String _bengaliDigit(String digit) =>
    String.fromCharCode(0x09E6 + int.parse(digit));

/// Bangla script glyph for every one of the 37 trained `letter_*` classes.
/// The Romanized names themselves (LETTER_NAMES in
/// ml/src/dataset_bdsl_static.py) come straight from the dataset's own
/// embedded file paths, and they follow the standard Bangla alphabet order
/// exactly -- vowels in vowel order, then consonants in consonant order,
/// with a doubled-letter romanization (Tta/Dda/Ddha) marking the retroflex
/// row apart from the single-letter dental row (Taa/Daa/Dha) right after
/// it. That positional match is what fixes each name to one glyph. A few
/// standard letters have no separate class here (dontyo "ন", "ভ", "য", "র",
/// "ষ", "স") -- the dataset consolidates each into the one visually/
/// acoustically closest sign it does have (e.g. no separate ণ/ন classes,
/// just this one "Naa"), which is a modeling choice made upstream in the
/// dataset, not a gap in this mapping.
const Map<String, String> _bdslLetterGlyphs = {
  'Awe': 'অ',
  'Aaa': 'আ',
  'Eee': 'ই',
  'Uuu': 'উ',
  'Ree': 'ঋ',
  'Aye': 'এ',
  'Oii': 'ঐ',
  'Ooo': 'ও',
  'Oww': 'ঔ',
  'Kaa': 'ক',
  'Kha': 'খ',
  'Gaa': 'গ',
  'Gha': 'ঘ',
  'Uyo': 'ঙ',
  'Cha': 'চ',
  'Chha': 'ছ',
  'Jaa': 'জ',
  'Jha': 'ঝ',
  'Eyo': 'ঞ',
  'Tta': 'ট',
  'Ttha': 'ঠ',
  'Dda': 'ড',
  'Ddha': 'ঢ',
  'Naa': 'ণ',
  'Taa': 'ত',
  'Tha': 'থ',
  'Daa': 'দ',
  'Dha': 'ধ',
  'Paa': 'প',
  'Pha': 'ফ',
  'Baa': 'ব',
  'Maa': 'ম',
  'Laa': 'ল',
  'Haa': 'হ',
  'Sha': 'শ',
  'Anushar': 'ং',
  'Chandra': 'ঁ',
};

/// Romanized names for the 37 trained `letter_*` classes, same order as
/// LETTER_NAMES in ml/src/dataset_bdsl_static.py -- display-only, for the
/// About screen's vocabulary list.
const List<String> bdslLetterNames = [
  'Awe', 'Aaa', 'Eee', 'Uuu', 'Ree', 'Aye', 'Oii', 'Ooo', 'Oww',
  'Kaa', 'Kha', 'Gaa', 'Gha', 'Uyo', 'Cha', 'Chha', 'Jaa', 'Jha', 'Eyo',
  'Tta', 'Ttha', 'Dda', 'Ddha', 'Naa', 'Taa', 'Tha', 'Daa', 'Dha',
  'Paa', 'Pha', 'Baa', 'Maa', 'Laa', 'Haa', 'Sha', 'Anushar', 'Chandra',
];

/// The BdSL letters the shipped model cannot actually recognize, measured
/// rather than guessed: these five score *zero* recall on held-out signers
/// even when the argmax is restricted to Letters mode, so no amount of
/// careful signing will produce them. See `unrecognizable_in_app` in
/// ml/eval_full_metrics.py and docs/PROJECT_REPORT.md 5.1.
///
/// Measuring this the way the app actually scores changed the answer, so
/// it was worth doing properly. Against the full 107-class output six
/// letters look dead -- but `Baa` is merely outscored by *word* classes it
/// never competes with in Letters mode, where it does recognize. Only
/// these five fail inside their own mode. Two the earlier single-hand
/// model could not recognize at all, `Haa` and `Naa`, recovered once the
/// position anchor was added.
///
/// Naming them is the honest alternative to presenting all 37 letters as
/// though they worked equally well. More training data, not an interface
/// change, is the real fix; that is recorded as future work rather than
/// quietly papered over.
const Set<String> bdslUnreliableLetters = {
  'Dda',
  'Kaa',
  'Laa',
  'Ooo',
  'Uuu',
};

/// Falls back to the English gloss itself if it's ever missing --
/// deliberately never throws, since a display lookup shouldn't crash the
/// live camera loop.
String bengaliFor(String englishGloss) {
  if (englishGloss.startsWith('digit_')) {
    return _bengaliDigit(englishGloss.substring('digit_'.length));
  }
  if (englishGloss.startsWith('letter_')) {
    final name = englishGloss.substring('letter_'.length);
    return _bdslLetterGlyphs[name] ?? name;
  }
  return bdslBangla[englishGloss] ?? englishGloss;
}
