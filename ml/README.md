# SignBridge — ML training pipeline

Trains a landmark-sequence sign classifier (1D-CNN or small Transformer)
and exports it to `.tflite` for the Flutter app in `../app`. See
`../docs/PROJECT_DOCUMENTATION.md` for full results and context.

## Setup

```bash
cd ml
pip install -r requirements.txt
```

## BdSL — done, shipped

The shipped model is trained on **three combined sources** into one
107-class model (60 words + 10 digits + 37 alphabet letters), via
`train_bdsl_combined.py`:

1. **Words** — Kaggle `kaziredwanislam/bdslw60-landmark-data-trials`
   (pre-extracted MediaPipe landmarks, 60 Bangla words, 9,229 sequences, 18
   signers, real video sequences). The English-gloss-to-Bangla-script
   mapping in `../app/lib/services/bdsl_glossary.dart` was transcribed from
   that dataset's own documentation, `data/bdslw60_docs/BdSLW60.pdf`.
2. **Digits (0-9)** and **3. Alphabet letters (37 classes)** — Kaggle
   `rayeed045/bangla-sign-language-dataset` (`CSV Files/Sign Digits - CSV
   Files/` and `CSV Files/Sign Letters - CSV Files/`, 10 signers each).
   Each sample is a single photographed hand pose, not a video — verified
   via the dataset's own embedded file-path column (see
   `src/dataset_bdsl_static.py` docstring for how the labels were
   confirmed). Each single frame is expanded into the model's fixed
   40-frame window by repeating it (`landmarks.prepare_sequence`'s
   existing t==1 branch), so the same architecture and app-side inference
   code handles all three sources with zero changes.

```bash
mkdir -p data/bdslw60 && cd data/bdslw60
kaggle datasets download -d kaziredwanislam/bdslw60-landmark-data-trials -p .
unzip -q *.zip && rm *.zip && cd ../..

mkdir -p data/bdsl_digits && cd data/bdsl_digits
kaggle datasets download -d rayeed045/bangla-sign-language-dataset -p .
unzip -q *.zip && rm *.zip && cd ../..

python train_bdsl_combined.py \
    --words-dir data/bdslw60 \
    --digits-dir "data/bdsl_digits/CSV Files/Sign Digits - CSV Files" \
    --letters-dir "data/bdsl_digits/CSV Files/Sign Letters - CSV Files" \
    --epochs 100
```

Result: **82.8%** accuracy on signers never seen during training, across
107 classes. That number reflects three upgrades stacked on top of each
other, in the order they were actually applied:

1. **71.9% → 77.6%**: landmark augmentation (mirroring, rotation, scale,
   noise -- see "Augmentation" below) and slightly higher dropout in
   `src/model.py`.
2. **77.6% → 83.4%**: both hands, not just one (see "Two hands, not one"
   below). BdSLW60 genuinely has both hands in frame for ~43% of frames
   across a random sample -- real signal an earlier version of this
   pipeline discarded by keeping only "whichever hand tracked better" for
   the whole sequence.
3. **83.4% → 82.8%**: the position anchor (see "The position anchor"
   below). Overall this is *flat to marginally down*, and it is listed
   anyway because the category breakdown is where it earned its place:
   letters went 54.1% → 59.5% and the zero-recall letter count went 7 → 5.
   For BdSL the anchor mostly helps the alphabet, because the digit and
   letter samples are single photographs with no meaningful in-frame
   position and are given a constant neutral anchor at training time. On
   ASL, where every class is real video, the same change is worth +4.5
   points on words.

Reported the honest way round: the headline number did not improve, so
the headline number is not what is claimed for this change.

```bash
python train_bdsl_combined.py \
    --words-dir data/bdslw60 \
    --digits-dir "data/bdsl_digits/CSV Files/Sign Digits - CSV Files" \
    --letters-dir "data/bdsl_digits/CSV Files/Sign Letters - CSV Files" \
    --epochs 100 --augment-copies 3
```

Output copied into `../app/assets/models/bdsl/` and wired up
(`../app/lib/services/bdsl_glossary.dart` maps every `digit_*`/`letter_*`
label to a real Bangla numeral or letter glyph for display and speech —
see that file and `dataset_bdsl_static.py` for how each Romanized name was
pinned to its glyph via standard Bangla alphabet order).

The older words-only run (`train.py --language bdsl --mode full`, 64.7%,
60 classes, single-hand) is superseded by this combined model but the
script still works if you want to reproduce it.

## ASL — done, shipped

The full `asl-signs` dataset (Kaggle competition "Google - Isolated Sign
Language Recognition": 250 signs, 94,477 sequences, 21 signers,
pre-extracted MediaPipe landmarks) was originally blocked by an
undocumented per-file download quota on the competition API (see
`probe_asl_availability.py` for how that was diagnosed: 0/210 fresh
signs servable in one probing pass). It unblocked via a full manual
download through the Kaggle website's "Download All" button — a
different, bulk endpoint not subject to that quota.

```bash
mkdir -p data/asl-signs && cd data/asl-signs
# unzip the manually-downloaded asl-signs.zip here
cd ../..

python train.py --language asl --data-dir data/asl-signs --mode full \
    --arch cnn --epochs 40 --batch-size 128 --augment-copies 2
```

At that scale (94,477 parquet files), two real bugs surfaced and were
fixed in `src/dataset_asl.py`:

- **Performance**: the original per-row `iterrows()` loop scaled fine at
  the 535-sequence quota-limited scale this was first tested at, but was
  utterly impractical at 94,477 files. Replaced with a vectorized
  numpy scatter.
- **Correctness**: MediaPipe briefly losing hand tracking leaves scattered
  per-landmark NaNs; left in, they propagate through
  `landmarks.resample_sequence`'s `np.interp` and poison the whole sample
  with NaN. Now interpolated over time before use.
- Loading is parallelized across a **thread pool**, not a process pool —
  a process pool was tried first and crashed the whole run: on Windows,
  each spawned worker process re-imports this script's module-level
  imports from scratch, including TensorFlow, and a dozen-plus
  simultaneous TensorFlow imports exhausted the system paging file.
  Threads share the parent process's memory and avoid this entirely,
  while still parallelizing the actual bottleneck (pandas/pyarrow parquet
  parsing, which releases the GIL).

Result: **52.0%** accuracy across all 250 classes (random chance: 0.4%)
single-hand, rising only to **53.4%** after adding both hands (see "Two
hands, not one" below), and to **57.9%** with the position anchor (see
"The position anchor" below) -- a much smaller gain than BdSL's from the same
upgrade. The reason is a genuine dataset property, not a shortcoming of
the two-hand pipeline: a random sample of `asl-signs` frames has both
hands present just **0.1% of the time**, against BdSL's 43%. This specific
corpus's recordings almost never have a usable second hand to give the
model in the first place. Landmark augmentation alone (before two hands)
plateaued fast too, tried up to 4x copies for +2.5 points and no further
-- consistent with the same story: with only 21 signers and this little
exploitable structure left in the hand data itself, there isn't much more
to buy without a genuinely new data source, which is why the
fingerspelling alphabet (and later, digits) were added instead, not more
augmentation or more architecture tuning.

### ASL fingerspelling alphabet (A-Z) and digits (0-9)

Unlike BdSL's digit/letter classes, no pre-extracted landmark dataset
exists for ASL fingerspelling that was found trustworthy (see "Considered,
not used" below). Rather than guess at a third-party CSV's labels, this
project generates its own: `extract_asl_alphabet_landmarks.py` runs the
real MediaPipe `HandLandmarker` (the same Tasks-API model family, and
literally the same `hand_landmarker.task` bundle, that the Flutter app's
plugin uses) over `grassknoted/asl-alphabet` (87,000 raw JPEGs, A-Z +
space/del/nothing, capped here at 400 images/letter) to produce real
(21, 3) landmarks per image. Nothing here is trusted or reverse-engineered
from someone else's export -- a folder named "A" produces landmarks for
whatever hand shape the detector found in that folder's images.

ASL originally shipped with no digit vocabulary at all (only BdSL had
one) until real-device testing asked for one directly.
`extract_asl_digit_landmarks.py` does the identical thing over
`rayeed045/american-sign-language-digit-dataset` (5,000 raw JPEGs, 0-9,
capped here at 400 images/digit, CC0-1.0, derived from the older Massey
University 2D static ASL gesture dataset).

```bash
mkdir -p data/asl-alphabet && cd data/asl-alphabet
kaggle datasets download -d grassknoted/asl-alphabet -p .
unzip -q *.zip && cd ../..

kaggle datasets download -d rayeed045/american-sign-language-digit-dataset \
    -p data/asl-digits --unzip

curl -sL -o .models/hand_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

python extract_asl_alphabet_landmarks.py \
    --images-dir data/asl-alphabet/asl_alphabet_train/asl_alphabet_train \
    --max-per-class 400
python extract_asl_digit_landmarks.py \
    --images-dir "data/asl-digits/American Sign Language Digits Dataset" \
    --max-per-class 400

python train_asl_combined.py --epochs 40 --augment-copies 1
```

Neither source has participant/signer metadata (independent photos, not a
multi-signer video corpus), so their portion of the train/val split in
`train_asl_combined.py` is an ordinary random split, not a held-out-signer
split like every other model this project ships. Called out here and in
the app's About screen, not glossed over.

Result: **286 classes** (250 words + 26 letters + 10 digits). Extraction:
9,364 of a possible 10,400 alphabet samples (1,036 images had no
detectable hand, mostly on visually ambiguous letters like M/N); 3,599 of
a possible 4,000 digit samples (401 skipped, mostly concentrated on digit
"0" -- an ASL closed-fist/O handshape MediaPipe's detector reads far less
reliably than an open hand: only 150/400 of that digit's images had a
detectable hand at all, versus 340-399/400 for every other digit).
Combined held-out accuracy: **61.7%** overall, but that single blended
number hides very different results per category -- **57.9%** words,
**99.0%** letters, **98.9%** digits. Those per-category numbers came from
`eval_confidence.py` (see below), not just the training run's own blended
figure, precisely so a strong digit/letter score can't quietly stand in
for the harder word score. Not apples-to-apples with each other or with
BdSL's 82.8%: digits and letters are small, visually distinct vocabularies
of clean, single, deliberately-posed photos, evaluated on the easier
random split described above -- a much easier problem than 250 words
performed as real video by 21 different people. Shipped anyway, because
each is a real, honestly-labeled addition on top of the real word model,
not a way to make the headline number look better -- the word-only number
above is the one to compare against BdSL's.

## Two hands, not one

Every model up to this point trained on one hand per frame -- whichever
one a loader decided was "the" hand for a whole sequence, discarding the
other entirely. Both `asl-signs` and BdSLW60 actually record both hands
per frame; that second hand was real signal being thrown away for every
two-handed sign in either vocabulary.

**The slot-assignment problem.** The natural fix is a 42-landmark frame
(two 21-landmark slots) instead of 21. The obstacle: the app's
`hand_landmarker` plugin reports each detected hand's 21 points with *no
handedness field at all* -- there's no "this is the left hand" signal at
inference time to match against a dataset's own left_hand/right_hand
column labels. Training on semantic handedness and inferring on... nothing,
would be a real train/inference mismatch.

The fix is geometric, not semantic: sort whichever hands are detected by
**wrist x-coordinate** (smaller x → slot 0) at both training-data
reconstruction time (from the datasets' raw per-frame coordinates,
ignoring their own left_hand/right_hand labels) and live inference (from
the camera frame's detected hands) -- see `landmarks.py`'s module
docstring and `sort_hands_by_x` / `combine_two_hands_by_x`, mirrored
exactly in the app's `TfliteSignClassifier._buildTwoHandFrame`. A lone
detected hand always lands in slot 0, matching how the single-hand static
datasets (BdSL digits/letters, the ASL alphabet) are embedded
(`landmarks.embed_single_hand`). Normalization uses one shared
origin/scale (from whichever slot is present, preferring slot 1) so the
two hands' *relative* position to each other -- often the entire signal
for a two-handed sign -- survives instead of each hand being centered
independently.

**Three real bugs surfaced building this, in order of discovery:**

1. **Windows process-pool memory exhaustion** (already covered above,
   applies here too since the loaders changed again).
2. **Mirror-augmentation swapped the wrong frames.** The first version
   swapped slot 0 ↔ slot 1 for *every frame* of a sample chosen for
   mirroring, unconditionally. For a frame where only one hand is
   present -- effectively all of them for ASL, ~57% for BdSL -- that
   moved the lone hand into slot 1 for roughly half of all mirrored
   samples, directly contradicting the "lone hand always in slot 0" rule
   used everywhere else. This alone dropped ASL's words-only accuracy
   *below* its old single-hand baseline (52.0% → 47.2%) before it was
   caught and fixed to swap per-*frame*, gated on whether that specific
   frame actually has a second hand present (`augment.py`).
3. **A dropped data-quality filter.** The original single-hand loader
   rejected a hand if more than half its landmarks were NaN on average --
   unreliably tracked data that hurt more than it helped. The first
   two-hand rewrite of `dataset_asl.py` dropped this check entirely (more
   inclusive: every one of 94,477 sequences produced *some* data, versus
   69% before), which measured *worse* again (52.0% → 47.2%, same
   benchmark, even after fixing bug #2) -- reinstated per-hand in
   `_extract_hand_aligned`, restoring the original ~69% inclusion rate and
   the accuracy that went with it.

The lesson worth stating plainly: none of these were caught by code
review or by the numbers "looking plausible" -- each was caught by
comparing a new result against the previous honest benchmark and refusing
to ship a regression, then tracing it to a specific line. `dataset_cache`'s
`LANDMARK_VERSION` constant exists so a stale cache from before a fix like
this can never be silently reloaded as if it still matched current code.

## Confidence calibration (`eval_confidence.py`)

The app gates two decisions behind a confidence threshold on the model's
own softmax output: whether a recognition is even shown as a live guess,
and whether it's allowed to auto-commit into the sentence (see
`app/lib/screens/home_screen.dart`'s `_minCommitConfidenceFor`). An
earlier build used one flat number for every category, picked by feel,
not measurement -- and after real-device testing reported recognition
"hardly detect[ing] anything," that number turned out to be exactly the
kind of thing that should never have been guessed at when the real model
was sitting right there to check against.

`eval_confidence.py` runs the *shipped* `.tflite` model (not the
pre-quantization Keras model -- what's actually on-device) over its own
real held-out validation split, and reports, per category: of predictions
that were *actually correct*, what fraction would still clear a given
confidence bar (recall -- a low number means the gate is discarding real,
correct recognitions); and of predictions clearing a given bar, what
fraction were actually correct (precision -- a low number means lowering
the bar lets a lot of wrong guesses through as if they were real). Usage:

```bash
python eval_confidence.py --language asl --data-dir /path/to/asl-signs
python eval_confidence.py --language bdsl
```

What it found: BdSL words were essentially unaffected by the threshold
(97% of correct words still clear 0.60), but BdSL digits and letters were
hit hard (only 78% and 85% respectively still cleared 0.55) for barely
any precision gain over a lower bar. ASL's letters and digits behaved like
BdSL's -- always near-100% confident when correct, so the threshold barely
matters there -- but ASL's words are the one category where the threshold
is a real precision/recall trade, not a free win: recall climbs from 73%
to 87% dropping the bar from 0.55 to 0.40, but precision falls from 78% to
69% at the same time, meaning more of what gets committed at the lower bar
is simply wrong. The app now uses 0.55 for words and 0.40 for digits/
letters -- a real, measured split instead of one more guess. Re-run this
whenever a model is retrained or a threshold is reconsidered; don't
re-guess.

## Two accuracies, and a metric bug of my own making

`eval_full_metrics.py` reports per-category accuracy two ways, because for
a while it reported only the first and the first was misleading:

- **unmasked** -- argmax over all 107 (or 286) classes. Measures the model.
- **in-app** -- argmax restricted to the category the user has selected,
  which is exactly what `setAllowedCategories` does in the app. Measures
  the product.

For BdSL digits these differ by **29 points**: 70.56% unmasked, **99.53%**
in-app. The giveaway was digit macro-precision of exactly 1.0000 -- when
the model says a digit it is *never* wrong, so its "errors" were not
digit/digit confusions at all, they were losses to **word** classes that a
user in Digits mode never competes against. Reporting only the unmasked
number documented a weakness the shipped app does not have.

It cuts the other way too, which is why both are kept: the in-app number
alone would flatter the model by hiding that those classes are genuinely
outranked whenever the mode is not restricted. ASL barely moves either way
(≤0.8 points) -- its static classes are already separated confidently
enough that nothing outranks them.

The same distinction changes which BdSL letters count as broken. Six have
zero unmasked recall, but `Baa` recovers to 16.7% in Letters mode -- it was
losing to words, not to other letters. **Five** are genuinely
unrecognizable: `Dda`, `Kaa`, `Laa`, `Ooo`, `Uuu`. Those five are what the
app marks in its About screen (`bdslUnreliableLetters`), rather than
offering all 37 as if they worked equally. Two that the pre-anchor model
also failed on, `Haa` and `Naa`, recovered.

Two operational gotchas in `eval_full_metrics.py`, both of which cost me a
wrong measurement before I noticed:

- **`--model` defaults to the copy in `../app/assets/models/`, not to
  `models/`.** That is the right default -- the point of the script is to
  measure what actually ships -- but it means that after training a new
  variant you are still measuring the *old* one until you copy it across.
  I compared two models this way and got byte-identical numbers, which is
  the symptom to watch for. Pass `--model models/<dir>/model.tflite`
  explicitly when comparing variants.
- **The dataset cache is keyed by argument values, including the absolute
  data path**, so two scripts that load "the same" data with any
  difference in those arguments will miss each other's cache and silently
  re-parse 94,477 parquet files. If only a subset of the raw corpus is on
  disk that re-parse *fails* rather than being merely slow. `.cache/`
  contains one entry per distinct key; check shapes with
  `np.load(...)['X'].shape` before assuming an entry is stale (43 rows =
  with anchor, 42 = without).

## Class weighting: tried, measured, dropped

The zero-recall letters are also among the rarest classes, so imbalance was
the obvious suspect. `train_bdsl_combined.py --class-weight` applies
inverse-frequency weights so every class contributes equally to the loss.

It didn't work:

| | anchor only (shipped) | anchor + class weight |
|---|---|---|
| Top-1 | 82.84% | 82.87% |
| **Macro-F1** | **0.7488** | 0.7425 |
| Digits (in-app) | **99.53%** | 99.07% |
| Letters (in-app) | **63.36%** | 61.86% |
| Zero-recall | 5 letters | 6 letters + `digit_8` |

Overall accuracy moved 0.03 points -- noise -- while macro-F1, the metric
class weighting exists to improve, went *down*, and it broke a digit class
that the unweighted model gets right 99.5% of the time. The flag stays in
the code because the experiment is worth being able to reproduce; the
unweighted model is what ships.

Method note, since it is the more useful lesson: I first ran this with the
anchor *and* class weighting introduced at the same time, got a mess of
mixed gains and regressions, and could not attribute any of it. That run
was thrown away and both variables re-measured separately against the same
split. The table above is the second, controlled pair.

## Training details

Each run writes to `models/<language>_<mode>_<arch>/`:

- `model.tflite` — quantized model ready to bundle into the app
- `labels.json` — index -> label name, in the order the model outputs them
- `history.json` — per-epoch accuracy/loss
- `saved_model/` — the pre-conversion Keras SavedModel

The train/validation split is by **participant**, not by sequence (see
`GroupShuffleSplit` in `train.py`) — the model is evaluated on signers it
never saw during training.

## Wiring a new model into the app

Copy `model.tflite` + `labels.json` into `../app/assets/models/<language>/`,
add the path to `pubspec.yaml`'s `assets:` list, then load it via
`TfliteSignClassifier.create(...)` (see
`../app/lib/services/tflite_sign_classifier.dart`) and select it in
`HomeScreen._classifierFor()`.

## Why hands-only, no pose/face

The deployed app's `hand_landmarker` plugin reports up to two hands' 21
keypoints each per frame (see "Two hands, not one" above), not the
pose/face landmarks MediaPipe Holistic also provides. Training on the
richer representation would create a train/inference mismatch the app
could never satisfy at inference time, so every loader in `src/` extracts
hands only, discarding pose/face columns even where the source dataset
has them (asl-signs does).

**This section previously claimed face landmarks wouldn't help at all.
That claim was wrong and is retracted.** The argument was: every class
here is a word/letter/digit gloss distinguished by hand shape and motion,
not by facial *expression*. That much is still true -- expression carries
grammar (questions, negation, clause boundaries), which this vocabulary
doesn't depend on. But it silently conflated expression with the face's
other job: being a **spatial anchor**. A large number of ASL signs are
defined by *where on the face* the hand is placed, and
`normalize_two_hand_frame` puts the origin at the hand's own wrist --
which deletes precisely that relationship. Two signs with identical
handshape and motion at the chin and at the cheek normalize to nearly
identical inputs.

`analyze_face_hypothesis.py` tests this instead of assuming it. A list of
47 face-located glosses was written from the vocabulary by meaning
*before* any per-class score was looked at, then compared against the
other 203 word classes:

| Group | Classes | Mean per-class recall | Median |
|---|---|---|---|
| Face-located words | 47 | **0.4421** | 0.4262 |
| All other words | 203 | **0.5414** | 0.5538 |

Mann-Whitney U = 3561.5, one-sided **p = 0.0034**. Worst affected: `chin`
0.057, `lips` 0.092, `awake` 0.094, `yucky` 0.107, `tongue` 0.109,
`cheek` 0.113.

## The position anchor

The face-location result above said *what* was missing: where the hand is,
relative to the signer. The cheapest thing that supplies most of it turned
out to already be sitting in the code, computed and discarded.

`normalize_two_hand_frame` derives an origin (the reference wrist) and a
scale (wrist → middle-finger MCP), divides them out, and drops them. Those
two quantities *are* the hand's in-frame position and apparent size. The
**position anchor** keeps them: one extra row per frame holding
`(origin_x, origin_y, scale)`, so the tensor goes (T, 42, 3) → (T, 43, 3).

It costs nothing at inference. The `hand_landmarker` plugin already
reports the wrist's frame position, so the app computes the same three
numbers it was already computing -- no second model, no new failure mode,
no re-extraction of either corpus. `tflite_sign_classifier.dart` reads the
row count off the model's own input tensor and appends the anchor only if
the model expects 43 rows, so a 42-row and a 43-row model both run
correctly without a matching code change.

| Metric | Landmarks only | + anchor | Δ |
|---|---|---|---|
| ASL combined (286) | 57.70% | **61.74%** | +4.04 |
| ASL words (250) | 53.42% | **57.90%** | +4.48 |
| ASL top-5 | 82.69% | **84.99%** | +2.30 |
| ASL macro-F1 | 0.5723 | **0.6107** | +0.0384 |
| BdSL combined (107) | 83.4% | 82.8% | −0.6 |
| BdSL letters | 54.05% | **59.46%** | +5.41 |

What makes this a confirmed explanation rather than a lucky number: the
face-hypothesis test was re-run *unchanged*, against the same
pre-committed 47-gloss list, and the gain landed where the hypothesis said
it would.

| Group | Before | After | Gain |
|---|---|---|---|
| Face-located words | 44.21% | **50.16%** | **+5.95** |
| All other words | 54.14% | **58.12%** | +3.98 |

Gap 9.93 → **7.96** points, U = 3751.5, one-sided p 0.0034 → **0.0113**.
`lips`, `tongue` and `cheek` all dropped out of the worst-ten list.

Two things this is *not*. It is not a face landmark: it gives position
relative to the **camera frame**, not to the face, so it holds only while
the signer stays roughly still relative to the camera -- which is why 7.96
points of the gap survive. And the p-value going *up* is not a failure;
shrinking a real effect is exactly what makes it harder to distinguish
from chance.

Two traps in the augmentation code, both worth knowing if you touch
`src/augment.py`. The anchor is not a landmark and must not be transformed
like one: under mirroring, landmark x negates (`x → −x`, wrist-relative)
but the anchor's x is an absolute frame coordinate and must reflect about
the centre (`x → 1 − x`); rotation must not touch the anchor at all. And
the two-hand slot-swap used to detect a two-hand frame with
`shape[-2] == 42`, which silently became false at 43 rows -- the swap just
stopped running, no error, only quietly worse training data. That check
now counts landmarks explicitly, and `test_anchor.py` guards it.

A real face reference point (nose tip or face centre) is still the next
step and would close the remaining gap properly:
`mediapipe_face_mesh` has Flutter plugin support so the app could supply
it at inference time, and asl-signs already ships the face columns the
loaders throw away, so it is evaluable with no new data. See "Investigated,
not pursued" below for the separate (and still open) facial-*expression*
question.

## Rotation robustness: ASL "V"/"W" reading as "Q"/"R" on real devices (measured, NOT shipped)

Real-device testing found a specific, reproducible failure: signing ASL
"V" or "W" would frequently be recognized as "Q" or "R" instead, and
rotating the hand toward a more upright angle sometimes fixed it. That
last detail was the actual clue -- it pointed at rotation specifically,
not a general letter-confusion problem.

**The standard offline benchmark cannot see this at all.** It evaluates
on real photos in their original, canonical orientation -- there is no
rotation in that test set for a rotation-robustness gap to show up in.
Comparing two models only on that benchmark would answer "did the
aggregate number regress", not "does this actually fix what was reported".
So a dedicated test was built instead (`eval_rotation_robustness.py`):
take the real ASL alphabet static poses, apply the *same* 2D in-plane
rotation transform training's own augmentation uses (`src/augment.py`,
applied to the hand-relative normalized landmark block, never to the
frame-absolute anchor row), across a sweep of angles, and measure each
model's accuracy at each one. This directly tests the thing that was
actually reported, rather than a proxy for it.

**Training only augments rotation by +/-15 degrees by default**
(`augment.py`'s `rotation_deg`). A person signing one-handed while
holding a phone naturally rolls the signing hand more than a posed
training photo does, and that's a very plausible amount to exceed 15
degrees. Widening it to +/-30 for a first test confirmed the mechanism
directly:

| Rotation | Baseline (15°) letter acc | Baseline "V" recall | Wider (30°) letter acc | Wider "V" recall |
|---|---|---|---|---|
| +0° | 99.6% | 0.99 | 99.5% | 0.99 |
| +30° | 88.5% | 0.95 | 96.1% | 0.98 |
| +45° | 56.9% | **0.15** | 85.6% | **0.92** |
| +60° | 31.1% | **0.00** | 69.4% | **0.78** |

At 0 degrees -- the offline benchmark's entire operating range -- the two
models are statistically identical (-0.07 points). The gap only opens as
rotation increases, exactly where the standard benchmark has nothing to
say, and "V" is precisely the letter that collapses first and hardest in
the baseline (99% -> 15% -> 0% recall by 60 degrees).

**Widening rotation for every category cost something, though**: word
accuracy (a full sequence-based recognition problem, not a static pose)
dropped about 0.9 points (57.90% -> 56.98% in-app) when the wider range
was applied uniformly. Plausibly, a dynamic sign's motion pattern is a
harder thing to hold recognizable under added spatial jitter than a
single static pose is -- augmenting a temporal sequence's every frame with
independent rotation jitter is a bigger perturbation to the *shape of the
motion* than the same jitter is to one photographed handshape.

**The shipped fix is category-specific, not a blanket change.**
`augment.py`'s `rotation_deg` parameter now accepts a per-sample array
(`train_asl_combined.py` builds one from a tracked `is_static` mask,
carried alongside `X`/`y`/`signers` through the same concatenation and
train/val split): word sequences keep the original +/-15 degrees exactly,
while static letter/digit samples widen to +/-30. Result: word accuracy
recovered to 57.35% in-app (indistinguishable from the 57.90% baseline --
the residual difference is ordinary run-to-run training variance, since
word augmentation is now byte-identical to before), digits and letters
are flat to slightly improved, and "V" recall still recovers dramatically
under rotation (0.00 -> 0.47 at 60 degrees, 0.15 -> 0.93 at 45 degrees) --
a little less extreme than the blanket-rotation experiment at the most
severe angles, in exchange for not costing anything on words at all.

**This model is NOT what the app currently ships, by explicit choice.**
Deployed and live-tested briefly, it surfaced a new, different real-device
report -- "P" and "M" becoming a common misread -- that had not been
measured or root-caused before the decision had to be made. Rather than
ship an improvement on one axis (V/W) while an unverified regression on
another (P/M) was still an open question, the app was reverted to the
pre-fix anchor-only model, the same one described in the position-anchor
section of this document. That is what ships today.

The trained artifact is kept, not discarded: `models/
asl_combined_cnn_rot30_static_only/` holds the category-specific-rotation
model in full, and everything measured above about it is real and
reproducible. Restoring it to `app/assets/models/asl/` -- a two-file copy,
no retraining -- is what shipping it later would take, *after* the P/M
report gets the same treatment the original V/W report got: reproduced
with real held-out data, root-caused, and weighed against a control
before being accepted or rejected. `models/asl_combined_cnn_rot15_
baseline/` is the currently-shipping model, kept under this name for the
comparison's own sake now that the two have swapped roles.

The intermediate blanket-rotation experiment (rotation widened for
*every* category, the one measured to cost 0.9 points of word accuracy
above) was evaluated directly rather than saved to its own directory, and
was overwritten on disk by a later training run before being copied
elsewhere -- its measurements above are reproducible by rerunning
`train_asl_combined.py --rotation-deg 30` without this session's
category-specific change, but the exact weights from that run no longer
exist as a separate artifact.

**Follow-up needed before this can ship**: reproduce the "P"/"M" report
against real held-out data the same way the original "V"/"W" report was
reproduced in `eval_rotation_robustness.py` -- confusion-matrix check
across both the anchor-only and rotation-widened models, at multiple
rotation angles, specifically for "P" and "M". If it's a real regression
introduced by widening rotation on static samples, the category-specific
model should not ship until that's addressed too; if it turns out to be
unrelated (present in the anchor-only model already, or a rotation- or
lighting-specific artifact of one test session), that should be
established with the same rigor before re-deciding to ship.

## Skeleton-image merge experiment for ASL letters (investigated, not shipped)

A third-party project (Devansh-47/Sign-Language-To-Text-and-Speech-Conversion)
ships `AtoZ_3.1`, ~4,681 real photos of a 21-point hand skeleton drawn on a
white canvas -- one webcam capture per frame, 26 ASL letters, an
independent signer pool from ours. The question worth testing: does
merging that real, independently-captured corpus into training produce a
better letter recognizer than the one already shipped?

**The representations had to be made to match, not just resized.**
`AtoZ_3.1`'s own landmark coordinates aren't recoverable from its images --
the 21 red joint dots are 2px circles drawn under 3px green bones, and
JPEG compression destroys most of them (a spot check recovered 9 of 21).
So the merge works the other way: our own alphabet/digit landmarks
(`ml/data/asl_alphabet_landmarks.npz`) are *rendered* into their exact
drawing convention (`src/skeleton_render.py`) -- same canvas size, same
bone/joint colors and thickness, same centring rule, matched hand-size
distribution (validated by comparing rendered bounding-box statistics
against theirs, not assumed). Every rendered sample is also independently
mirrored ~50% of the time (`render_skeleton_dataset.py`), because neither
corpus attempted to balance which hand its signers used, and a
fingerspelling letter's label doesn't depend on which hand performs it --
the same reasoning `augment.py`'s `mirror_prob` already applies to the
landmark-sequence model.

**A real defect was found and fixed in the renderer before any training
happened**: `render_two_hand_from_landmarks` (the BdSL-oriented two-hand
variant, built in case this approach proved worth extending) transformed
landmark points but never converted them to pixel space before drawing --
every point collapsed to a single dot near canvas centre. Caught by
looking at the actual rendered image, not by reasoning about the code;
`test_skeleton_render.py` now asserts on pixel spread and bounding-box
size specifically so this class of bug can't silently return.

**A second real defect surfaced during training, not rendering**: a small
CNN (three conv+BatchNorm+pool blocks) trained on these images was
numerically unstable -- val_loss spiking into the tens or hundreds while
training accuracy climbed smoothly, val_accuracy gyrating near chance
rather than trending. Lowering the learning rate removed the spikes but
made convergence too slow to be usable; gradient clipping alone reduced
but didn't eliminate them; lowering BatchNorm's momentum made it *worse*
(spike to 42.7). Removing BatchNorm entirely fixed it outright -- the
likely mechanism is that with only ~100-800 steps per epoch on these
corpora, BatchNorm's running statistics (an exponential moving average)
hadn't converged, so train-mode and eval-mode behaviour diverged sharply.
`diagnose_lr.py` documents the full investigation and is kept as a
reusable diagnostic for the next time a small-data CNN in this project
shows the same healthy-train/exploding-val signature.

**The result, once training was actually stable:**

| Trained on | eval: theirs_held_out | eval: ours_held_out |
|---|---|---|
| theirs_only (3,746) | 67.6% | 16.6% |
| ours_only (29,964) | 39.7% | **94.7%** |
| **merged (33,710)** | **81.1%** | **95.2%** |

The merge genuinely helps generalization, not just accuracy on more of the
same data: the merged model beats *both* single-source models on the eval
set neither of them was trained toward reaching -- 81.1% on an
independent real capture setup theirs_only itself only reached 67.6% on,
and 95.2% on our own held-out set, a hair above ours_only's 94.7%. That is
a genuine, positive, reportable finding about cross-source robustness.

**It does not clear the bar to ship.** The shipped landmark-sequence
model's in-app ASL letter accuracy is 99.08% (`ml/eval_full_metrics.py`).
The best skeleton-image result, 95.2%, falls short by about 3.9 points --
image rendering is a strictly lossier re-encoding of the same landmarks
(z is dropped, coordinates are quantized to pixels) for the "ours" half of
the merge, and that loss isn't recovered by the smaller amount of added
"theirs" diversity. The merged model, its training artefacts, and this
write-up are kept; the shipped model is unchanged.

## Investigated, not pursued

**ASL alphabet landmark dataset** (`srisahithis/american-sign-language-a-z-dataset-hand-landmarks`)
— found while looking for a pre-extracted alternative to running MediaPipe
ourselves. Despite the name, its "landmarks" are 21 keypoints drawn as red
dots directly onto the JPEGs, not exported numeric coordinates -- visual
clutter that would confuse a detector run over these images, and nothing
to train a numeric landmark classifier on regardless. Ran our own
extraction over `grassknoted/asl-alphabet` (clean images, no overlay)
instead -- see the ASL fingerspelling section above.

**Facial expression / pose** — the main project proposal's Research
Objective #2 mentions facial expression alongside hand gesture
recognition. Flutter plugin support exists (`mediapipe_face_mesh`,
`flutter_mp_pose_landmarker` on pub.dev), so on-device integration is
technically possible. What's missing is training data: no dataset was
found that pairs facial expression or body pose with ASL or BdSL
*grammar* labels compatible with this project's word/letter/digit gloss
vocabulary. The one facial-expression dataset found on Kaggle
(`prashant111/grammatical-facial-expressions`) is annotated for Brazilian
Sign Language (Libras) non-manual markers — a different sign language,
a different annotation scheme (per-frame yes/no judgments on specific
grammatical constructions), and no hand landmarks at all, so it can't be
merged into this project's hands-only pipeline or its gloss vocabulary.
Building a face/pose classifier without a dataset that actually ties
facial expression to sign meaning would mean guessing labels — the same
mislabeling risk already rejected for the alphabet/BdSLW401 candidates
below. Revisit if a labeled ASL/BdSL facial-grammar dataset surfaces.

**BdSL fingerspelling/digit signs** (individual Bangla characters, not
whole words): an *earlier* candidate dataset was found
(`rayeed045/bangla-sign-alphabet-dataset` — note: a different dataset from
`rayeed045/bangla-sign-language-dataset` used above) but rejected — only
20 classes from a single signer, and its label column is a bare integer
with no verified mapping to actual Bangla characters. The digits/letters
now shipped come from the sibling dataset, `bangla-sign-language-dataset`,
whose labels were independently verified via embedded file paths.

**BdSLW401** (401 words instead of 60): two landmark candidates were
checked. `sahebaakter/bdslw401-1st-processed` turned out to be a tiny,
incomplete upload (2.3MB total, nowhere near the full corpus).
`adibrafid/final-bdslw-401-keypoints` is real (401 classes, 18 signers,
config-verified) but uses a 75-landmark pose+hands feature layout, not
this project's hands-only 21-landmark contract, and its labels are bare
codes ("W001"-"W401") with no documentation found mapping them to actual
Bangla words. Reverse-engineering both the column layout and the label
mapping without a verified source risked the same mislabeling problem as
the alphabet dataset above.

Both were set aside rather than pursued — word-level BdSLW60 (already
accurate and correctly labeled) was kept instead. Revisit if a properly
documented version of either surfaces.
