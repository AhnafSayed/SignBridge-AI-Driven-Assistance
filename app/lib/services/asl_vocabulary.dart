/// The 250 trained ASL sign labels, in the same order the model's
/// `labels.json` lists them (alphabetical). Display-only, for the About
/// screen's vocabulary list -- classification itself reads labels.json
/// directly (see TfliteSignClassifier), never this file.
const List<String> aslVocabulary = [
  'TV', 'after', 'airplane', 'all', 'alligator', 'animal',
  'another', 'any', 'apple', 'arm', 'aunt', 'awake',
  'backyard', 'bad', 'balloon', 'bath', 'because', 'bed',
  'bedroom', 'bee', 'before', 'beside', 'better', 'bird',
  'black', 'blow', 'blue', 'boat', 'book', 'boy',
  'brother', 'brown', 'bug', 'bye', 'callonphone', 'can',
  'car', 'carrot', 'cat', 'cereal', 'chair', 'cheek',
  'child', 'chin', 'chocolate', 'clean', 'close', 'closet',
  'cloud', 'clown', 'cow', 'cowboy', 'cry', 'cut',
  'cute', 'dad', 'dance', 'dirty', 'dog', 'doll',
  'donkey', 'down', 'drawer', 'drink', 'drop', 'dry',
  'dryer', 'duck', 'ear', 'elephant', 'empty', 'every',
  'eye', 'face', 'fall', 'farm', 'fast', 'feet',
  'find', 'fine', 'finger', 'finish', 'fireman', 'first',
  'fish', 'flag', 'flower', 'food', 'for', 'frenchfries',
  'frog', 'garbage', 'gift', 'giraffe', 'girl', 'give',
  'glasswindow', 'go', 'goose', 'grandma', 'grandpa', 'grass',
  'green', 'gum', 'hair', 'happy', 'hat', 'hate',
  'have', 'haveto', 'head', 'hear', 'helicopter', 'hello',
  'hen', 'hesheit', 'hide', 'high', 'home', 'horse',
  'hot', 'hungry', 'icecream', 'if', 'into', 'jacket',
  'jeans', 'jump', 'kiss', 'kitty', 'lamp', 'later',
  'like', 'lion', 'lips', 'listen', 'look', 'loud',
  'mad', 'make', 'man', 'many', 'milk', 'minemy',
  'mitten', 'mom', 'moon', 'morning', 'mouse', 'mouth',
  'nap', 'napkin', 'night', 'no', 'noisy', 'nose',
  'not', 'now', 'nuts', 'old', 'on', 'open',
  'orange', 'outside', 'owie', 'owl', 'pajamas', 'pen',
  'pencil', 'penny', 'person', 'pig', 'pizza', 'please',
  'police', 'pool', 'potty', 'pretend', 'pretty', 'puppy',
  'puzzle', 'quiet', 'radio', 'rain', 'read', 'red',
  'refrigerator', 'ride', 'room', 'sad', 'same', 'say',
  'scissors', 'see', 'shhh', 'shirt', 'shoe', 'shower',
  'sick', 'sleep', 'sleepy', 'smile', 'snack', 'snow',
  'stairs', 'stay', 'sticky', 'store', 'story', 'stuck',
  'sun', 'table', 'talk', 'taste', 'thankyou', 'that',
  'there', 'think', 'thirsty', 'tiger', 'time', 'tomorrow',
  'tongue', 'tooth', 'toothbrush', 'touch', 'toy', 'tree',
  'uncle', 'underwear', 'up', 'vacuum', 'wait', 'wake',
  'water', 'wet', 'weus', 'where', 'white', 'who',
  'why', 'will', 'wolf', 'yellow', 'yes', 'yesterday',
  'yourself', 'yucky', 'zebra', 'zipper',
];

/// The 26 trained fingerspelling-alphabet classes, labeled `letter_A`
/// .. `letter_Z` by the model (see ml/train_asl_combined.py) to keep them
/// unambiguous alongside the word labels above.
const List<String> aslLetterNames = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
];

/// The 10 trained digit classes, labeled `digit_0` .. `digit_9` by the
/// model, same reasoning as the letters above.
const List<String> aslDigitNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

/// Strips the model's `letter_`/`digit_` prefix for display/speech -- e.g.
/// `letter_A` reads as just "A", `digit_3` reads as just "3". Falls back
/// to the raw label for anything else (a plain word gloss), same
/// never-throws contract as [bdsl_glossary.bengaliFor].
String aslDisplayFor(String label) {
  if (label.startsWith('letter_')) return label.substring('letter_'.length);
  if (label.startsWith('digit_')) return label.substring('digit_'.length);
  return label;
}
