# SignBridge

**AI-Driven Assistance for Hearing & Speech Impairments**
A bilingual (American Sign Language + Bangla Sign Language) sign-language recognition Android app with a two-way conversation mode.

MICT-2023, Department of Information and Communication Technology, Bangladesh University of Professionals.
Supervisor: Dr. Ahmedul Kabir (University of Dhaka).
Team: Samiul Islam, Ahnaf Sayed, Abu Saeed Sabuj.

---

## 1. Problem and Motivation

Hearing and speech impairments affect communication in both directions: the impaired person needs a way to express signed language to people who do not know sign language, and they need a way to understand what those people say back. Sign language interpreters are not always available, and text-only communication (typing back and forth) is slow and does not scale to natural conversation.

The original project proposal (30 Jun 2024) framed this as a one-directional problem: translate the user's hand gestures and facial expressions into text and speech. Over the course of development this was refined into a fuller, two-directional communication tool, and the technical approach was substantially revised from the original proposal, described in §3.

## 2. What Was Actually Built

A Flutter Android app, **SignBridge**, with two tabs:

- **Sign tab** — the camera watches the impaired user's hands (up to two at once), MediaPipe extracts 21 hand-landmark keypoints per hand per frame, a trained classifier recognizes the sign, and the recognized word or letter is added to a growing sentence, which is read aloud via text-to-speech.
- **Listen tab** — the reverse direction. The other person speaks into the microphone; on-device speech recognition transcribes it live as large on-screen text the impaired user can read. This completes the two-way conversation loop (comparable to Google Translate's face-to-face conversation mode, but signs-in/speech-out on one side and speech-in/text-out on the other, matched to what each participant can actually do).

Both directions support English and Bangla, and the whole pipeline runs on-device — no network connection required for recognition.

### 2.1 Hands-free operation

Because the target user may not be able to operate a touchscreen easily mid-conversation (or may simply prefer not to break eye contact to tap buttons), the Sign tab supports fully hands-free control once "Auto mode" is on:

| Action | How | Why this shape |
|---|---|---|
| Add a word | Hold any sign steady (~0.7s) | The core action |
| Delete last word | Hold a closed fist (~0.9s) | Rare shape among trained signs, easy to detect |
| Clear whole sentence | Hold thumbs-down (~1.4s, longer since destructive) | Intuitive "cancel" |
| Speak sentence now | Hold thumbs-up (~0.9s) | Intuitive "go/confirm" |
| *(implicit)* word separation | Nothing needed | Each committed word is already its own sentence entry, auto-spaced — mirroring how sign-typing keyboards (e.g. Gboard's sign-to-text mode) auto-space between recognized signs rather than requiring an explicit space gesture |
| Speak automatically | Lower your hand and pause (~1.8s) | Catches users who do not remember to use the gesture |

Manual buttons (+, Delete word, Clear, Speak) remain available as a fallback for anyone who prefers tapping, and Auto mode can be switched off entirely.

**A known, deliberately handled risk:** control handshapes can coincide with real signs (a closed fist happens to be exactly what the ASL prototype's finger-counter reads as digit "0"). This is resolved with *confidence-gated arbitration* — for a language with its real trained model loaded, a control gesture is only acted on when the classifier is not confidently recognizing an actual word in that same shape; for the finger-count prototype (a fallback shown only if a model fails to load, whose "confidence" is not a meaningful signal since it is a simple rule rather than a trained model), control gestures take priority and the digit can still be added manually via the + button.

**A second, real instance of this same risk surfaced during real-device testing**, this time against the trained models themselves rather than the prototype: several real Letters/Digits classes (ASL A/S/E/M/N/T, for instance) *are* closed-fist shapes. The first fix tried was to scope the automatic control handshapes to Words mode only, on the theory that confidence-gated arbitration alone wasn't a strong enough tiebreaker to stop backspace from winning most of the time. That went too far: it silently disabled backspace, clear, *and* speak together the moment the user was in Letters or Digits mode — read back as "the gestures don't work," since fingerspelling happens almost entirely in those two modes. The actual fix keeps all three gestures checked in every category, and relies on the confidence-gated arbitration alone (per-category thresholds, §4.4) — workable now that letters/digits are measured to be extremely confident when correctly formed (~99% recall), so a real, deliberate letter/digit attempt reliably outscores a merely fist-shaped "I meant to delete" hand.

**A third, more fundamental bug was found while chasing the same "gestures don't work" report**: the geometric test for "is the thumb held out" (`ControlGestureDetector._thumbExtended`, the difference between backspace and clear/speak) compared thumb-tip-to-*wrist* distance against palm *width*. Checked directly against this project's own real extracted training landmarks (`ml/data/asl_alphabet_landmarks.npz`), that comparison came out **true for every single closed-fist letter tested, including the canonical ASL "A"** — the wrist is structurally far from the thumb whether or not the thumb is actually held out, so the check was measuring hand size, not thumb curl. The practical effect: the "thumb is extended" branch was true almost unconditionally, which made the *other* branch — backspace — effectively unreachable for any real hand. Holding a closed fist could never register as delete; it would always land on clear or speak instead, depending on incidental thumb position. Fixed by measuring the thumb's direction from its own base joint instead of the wrist, and requiring the resulting vector to be both large and close to vertical — re-checked against the same real data, this now behaves correctly for the large majority of real closed-fist letters, with a specific, small, honestly-documented residual case (ASL "M", whose real handshape tucks the thumb down between two fingers) that the same confidence-based arbitration above already covers. Regression-tested in `app/test/control_gesture_test.dart` using these exact real landmark values, not synthetic stand-ins, for the closed-fist cases.

### 2.2 Two languages, one pipeline

ASL and BdSL share the same landmark-based architecture (see §4), so adding a language means training a new classifier, not rebuilding the app.

### 2.3 Recognition modes: Words / Digits / Letters

Real-device testing surfaced a specific failure mode: with the full trained vocabulary (words, digits, and fingerspelling letters) all competing for the same "best guess," a held-steady letter would sometimes lose to a similar-looking word and vice versa, since nothing told the classifier which vocabulary the user actually meant to sign. A mode bar above the camera (`SignCategory`: Words / Digits / Letters, filtered per-language since ASL has no digit vocabulary) restricts the classifier's output to the active category without retraining — the trained model is unchanged; the app simply ignores any class outside the active category before deciding what was recognized. See §4.8.

In Letters mode, spelled letters accumulate into one in-progress word (spelling C-A-T shows "CAT" building live) instead of each letter auto-spacing into the sentence the way a whole-word sign does — the earlier behavior that produced "C A T" as three separate sentence entries. Delete-last acts on the letter buffer first before falling back to deleting the last committed word. See §4.8 for the offline autocomplete built on top of this buffer.

### 2.4 Tab lifecycle (Sign ↔ Listen)

Both tabs hold expensive native resources — a live camera + hand-landmark pipeline on Sign, a microphone + speech recognizer on Listen. An early build destroyed and recreated each screen's widget on every tab switch, which raced native teardown against the next screen's init and could crash on a fast switch. The fix: `RootShell` keeps both screens permanently mounted in a `PageView` (via `AutomaticKeepAliveClientMixin` on each), and instead of destroy/recreate, calls public `pause()`/`resume()` methods on whichever screen is leaving/entering view (through a `GlobalKey`). The Listen tab additionally defers requesting microphone permission until the user actually swipes to it for the first time, rather than at app launch. `PageView` also means the two tabs are already swipe-navigable directly (not just via the bottom nav bar).

**Back-navigation and exiting.** From the Listen tab, back (button or the gesture-nav edge-swipe — Flutter's `PopScope` doesn't distinguish between the two) returns to the Sign tab, matching standard "back goes up one level" tab-navigation conventions. From the Sign tab — the root, with nowhere further "up" to go — an earlier build let a single back gesture exit the app immediately, which is one accidental edge-swipe away from losing the whole session (a real risk one-handed, mid-conversation, holding the phone with the same hand used to swipe). Real-device feedback asked for the standard Android "back again to exit" pattern instead: the first back attempt from the Sign tab shows an "Exit SignBridge?" confirmation dialog; a second attempt within 3 seconds — whether the dialog is still open or was already dismissed — exits directly without needing to tap anything, matching "swipe, see the prompt, swipe again to confirm" rather than requiring a button tap on the second attempt.

## 3. Architecture: What Changed From the Original Proposal, and Why

| | Original proposal (30 Jun 2024) | What was built |
|---|---|---|
| Input | Raw camera images | MediaPipe hand-landmark coordinates, up to two hands × 21 keypoints × x,y,z |
| Model | Native Android (Kotlin) + TensorFlow, CNN + RNN combo, hand gestures **and facial expressions** | TensorFlow-trained 1D-CNN over landmark sequences, exported to TensorFlow Lite; **both hands, no facial-expression model** |
| Platform | Native Android/Kotlin | Flutter (Android-first) |
| Direction | One-way (sign → text/speech) | Two-way (sign → speech, and speech → text) |

The shift to landmark-based recognition (rather than classifying raw pixels) is a deliberate, well-established pattern in sign-language recognition research: landmarks are far more robust to lighting, background, and skin tone than raw images, and the resulting model is dramatically smaller (the BdSL model is 196KB for 107 classes).

**Facial expression, re-investigated:** the original proposal's Research Objective #2 names facial expression as a target alongside hand gesture, so this was revisited directly rather than dropped by default. Flutter plugin support exists for both MediaPipe face mesh and pose landmarks, confirming that on-device integration is technically feasible. The blocker is training data: no dataset was found pairing facial expression or body pose with ASL/BdSL grammar labels compatible with this project's word/letter/digit vocabulary. The one facial-expression dataset located (`prashant111/grammatical-facial-expressions`, Kaggle) is annotated for Brazilian Sign Language (Libras) non-manual markers — a different sign language, with a different per-frame annotation scheme and no hand landmarks — so it cannot be merged into this project's hands-only pipeline or gloss vocabulary without guessing at labels, the same risk already rejected for the alphabet and BdSLW401 candidates in §5. This is a data-availability constraint, documented and deliberate, not an unexamined gap — see `ml/README.md`, "Investigated, not pursued," for the full trail.

An earlier prototype (the "Devansh" reference scripts, `test.py` / `train.py` in the repo root) used the original proposal's raw-image approach for ASL fingerspelling (A–Z letters via a hand-coded rule engine layered on a CNN). It was fixed and made to run during development, but is not part of the shipped app — it answers a different question (letters, not words) with a different, superseded technique, and is kept only as a reference/demo of the problem space.

## 4. Machine Learning Pipeline

### 4.1 Landmark preprocessing (`ml/src/landmarks.py`)

Every training sample and every live camera frame goes through the same steps, which is what makes training and on-device inference consistent:

1. **Combine both hands into fixed slots.** The app's `hand_landmarker` plugin (and every source dataset used here) can report zero, one, or two hands per frame, with no reliable "this is the left hand" signal on the Flutter side (the plugin exposes no handedness field). Rather than invent a semantic left/right assignment the app could never actually verify at inference time, both hands are sorted **geometrically** — by wrist x-coordinate — into slot 0 (whichever hand is further left in frame) and slot 1. A frame with only one hand fills slot 0 and zero-fills slot 1; this exact rule is applied identically during training (`landmarks.py`) and live inference (`tflite_sign_classifier.dart`), which is what makes the two consistent. See §4.5a for why this was worth doing and what it cost to get right.
2. **Resample** the sequence to a fixed 40 frames (linear interpolation over time), since isolated signs vary in duration but the model needs a fixed-size input.
3. **Normalize** each frame: translate so the reference hand's wrist landmark is the origin (slot 1 if present, otherwise slot 0), then scale by that hand's wrist-to-middle-finger-knuckle distance, making the input invariant to how far the hands are from the camera and where in frame they sit.

### 4.2 Model (`ml/src/model.py`)

A small 1D-CNN treats the (40 frames × 42 landmarks × 3 coordinates) input as a 1D signal over time: three Conv1D+BatchNorm+MaxPool blocks, global average pooling, dropout, and a dense softmax head. A Transformer-encoder alternative is also implemented (`build_transformer`) for future comparison but was not the one trained for this build. Total size for the shipped BdSL model (107 classes): under 250KB as `.tflite`.

### 4.3 Training results

| Language | Vocabulary | Sequences | Signers | Held-out accuracy* | Status |
|---|---|---|---|---|---|
| **BdSL** | 107 classes — 60 words + 10 digits + 37 alphabet letters | 9,229 word sequences + ~1,000 digit/letter frames (expanded to sequences) | 18 (words) + 10 (digits/letters) | **82.8%** (both hands + anchor; 77.6% single-hand) | Real, trained, shipped |
| **ASL** | 250 words | 65,220 (post quality-filter) | 21 | **57.9%** (both hands + anchor; 53.4% without anchor; 52.0% single-hand) | Real, trained, shipped |
| **ASL + alphabet + digits** | 286 classes — 250 words + 26 fingerspelling letters + 10 digits | 65,220 word + 9,364 letter + 3,599 digit | 21 (words) + none (letters/digits, see §4.7) | **61.7%** (blended, see §4.7) | Real, trained, shipped |

Random chance for ASL's 250-way classification is 0.4%, so ~58% is a strong signal, not a weak one — but it is meaningfully below BdSL's, and that gap is discussed honestly in §4.6 rather than hidden. All three numbers above are from the corrected two-hand pipeline (§4.5); see that section for the real bugs found and fixed along the way, and why ASL's two-hand gain is so much smaller than BdSL's.

The digit/letter portion of the BdSL model, and the ASL fingerspelling alphabet and digits (A-Z, 0-9, see §4.7), are trained on single photographed hand poses (not video), each expanded to the model's 40-frame window by repetition (see `ml/train_bdsl_combined.py` / `ml/train_asl_combined.py`), so they share one architecture and one on-device inference path with the word models — the app's Dart code needed zero changes to consume the larger vocabularies, only a bigger `labels.json`.

*Held out by **participant**, not by sequence — the reported accuracy is on signers the model never saw during training, which is the honest measure of whether it generalizes to a new person, not just memorizes. (Exception: the ASL alphabet/digit portions have no participant metadata — see §4.7.)

### 4.4 On-device inference

Both models run via `tflite_flutter`'s `IsolateInterpreter`, on a background isolate rather than the UI thread — an early build ran inference synchronously on the main thread, which visibly janked the camera preview; moving it off-thread fixed that. Training resamples each *whole gesture* to 40 frames regardless of duration. Live inference now matches that: a motion-energy segmenter (`app/lib/services/gesture_segmenter.dart`) accumulates frames while the hand moves and emits a complete, resampled gesture once motion settles, so the model receives input in the same form it was trained on. The old rolling window is kept only as a fallback for continuous signing that never settles, since a segmenter that waited forever for a pause would make the app stop responding. On-device accuracy may still differ from the offline benchmark, because live segmentation can itself pick the wrong boundary — a smaller and better-understood gap than before, but not zero.

The `HandLandmarker` plugin is initialized with `numHands: 2` and the **CPU** delegate, not GPU. It was originally GPU (faster), but real-device testing on some phones produced total silence — no spoken output, no gesture recognition at all — with no exception the app could catch and report; the most plausible cause is the GPU delegate failing to initialize on that hardware and failing silently rather than throwing. For an accessibility tool, a slower recognizer that works everywhere beats a faster one that is unpredictably dead on some devices, so CPU is now the default.

A committed recognition also has to clear a minimum-confidence gate before it is added to the sentence — added because real-device testing showed the same word being committed repeatedly in quick succession even as the sign changed, i.e. a stale/low-confidence classification being accepted as if it were a fresh, real one.

**That gate then itself caused a regression.** The next round of real-device testing reported the opposite failure: recognition "hardly detects anything." Two real, distinct causes, both traced with evidence rather than guessed at a second time:

- The same build had also raised the hand-landmarker's own `minHandDetectionConfidence` from the plugin's default (0.5) to 0.6, on the theory that stricter detection would help — shipped without any way to check that against a real device. Reverted to the default once real-device testing showed tracking itself getting worse, not better; there was never actual evidence 0.6 helped.
- The commit gate's threshold (0.55, flat across every category) was picked by feel rather than measured. `ml/eval_confidence.py` was written to check it directly against real held-out data: it runs the shipped `.tflite` model over its own validation split and reports, per category, what fraction of *actually correct* predictions would survive a given confidence bar. The result: BdSL words were basically unaffected (97% of correct words still clear 0.60), but BdSL digits and letters were hit hard (only 78% and 85% respectively cleared 0.55) for barely any gain in precision at that threshold versus a lower one. The gate now uses 0.55 for words and 0.4 for digits/letters — a real, measured split, not two more guesses. Separately, the app had also been hiding the live-guess label entirely below the gate, so a real but not-yet-confident detection looked identical to no detection at all; the label now always shows the raw top guess, and only the auto-commit-to-sentence step is gated.

**Speak not working traced to a third, unrelated cause**: Android 11+ hides installed text-to-speech and speech-recognition engine packages from every app by default (package visibility) unless the app explicitly declares it wants to query them, via a `<queries>` block in `AndroidManifest.xml`. This app's manifest never had that declaration for either `flutter_tts` or `speech_to_text` — confirmed against both plugins' own example apps, which do include it — so `isLanguageAvailable`/`speak()` could silently find no engine and do nothing, with no exception to catch. Added the two missing `<queries>` entries (`android.intent.action.TTS_SERVICE`, `android.speech.RecognitionService`); this plausibly also explains any earlier Listen-tab speech-recognition flakiness on Android 11+ devices, not just Speak.

### 4.5 Two hands, not one

Real-device testing raised a direct question: many signs in both ASL and BdSL genuinely use two hands, so wasn't the original single-hand model always going to be blind to half the signal? The original design tracked only one hand because the very first version of the `hand_landmarker` Flutter plugin integration only requested one, not because the datasets lacked a second hand — so this was worth fixing rather than a fundamental limit of the approach.

**Design problem: no handedness signal.** The Flutter plugin's `Hand` result has no left/right field at all — unlike the source *datasets* (asl-signs labels `left_hand`/`right_hand` explicitly; BdSLW60's raw export does too), the app has no way to ask the live camera feed "which hand is this?" Assigning training data to a `left`/`right` slot pair the app could never reproduce at inference time would be a train/inference mismatch baked in from the start. The fix used throughout this project is **geometric, not semantic** slot assignment: at every frame, whichever hand's wrist has the smaller x-coordinate (further left in the raw camera frame) goes in slot 0, the other in slot 1. A frame with only one hand always fills slot 0. This rule is implemented once in `ml/src/landmarks.py` for training and mirrored exactly in `tflite_sign_classifier.dart` for inference, so both sides agree by construction rather than by convention.

**Checking it was worth doing before building it.** Before touching any code, both datasets were sampled directly to check they actually contain two-hand frames worth training on: BdSLW60 has a real second hand present in roughly 43% of frames across a 300-sample check; asl-signs has a real second hand in only about 0.1% of frames across a 500-sequence check. This is an honest, load-bearing difference in the two datasets, not a bug — and it correctly predicted the outcome below.

**Results:** BdSL rose from 77.6% (single-hand) to **83.4%** (two-hand) — a real gain, because BdSLW60's signs frequently use both hands. ASL rose from 52.0% to only **52.9%** — barely above baseline, because this specific dataset's recordings almost never have a usable second hand to begin with, so there was little second-hand signal to gain from regardless of how correctly it was extracted. Both numbers are honestly reported rather than only publishing the flattering one.

**Three real bugs found and fixed while building this** (each caught by comparing against the previously-measured baseline rather than trusting a new number in isolation):

1. **Windows multiprocessing memory exhaustion.** `dataset_asl.py`'s loader originally used a process pool for parallel loading; on Windows, each spawned worker process re-imports the whole training script — including TensorFlow — from scratch, and at a dozen-plus workers this exhausted the system paging file. Fixed by switching to a thread pool (pandas/pyarrow and numpy release the GIL for their C-level work, so threads still parallelize the parts that matter here, without paying a process pool's Windows import cost).
2. **Mirror-augmentation swapped hands unconditionally.** The augmentation step that mirrors a training sample left-right also needs to swap which slot each hand occupies (a mirrored left-hand-in-slot-0 sign is now geometrically a right-hand-in-slot-0 sign). The first version of this swap applied to slot 1 on *every* mirrored frame, even frames where slot 1 was genuinely empty (zero-filled, no second hand) — silently violating the "a lone hand is always in slot 0" rule that training data, live inference, and the model itself all otherwise depend on. Because ASL is almost entirely single-hand frames, this corrupted roughly half of all mirrored ASL training data and drove held-out accuracy down to 46.5% — *below* the 52.0% single-hand baseline. Caught immediately because the discipline throughout this project is to always compare a new number against the last honest benchmark, never trust it in isolation. Fixed by making the swap conditional on slot 1 actually being present, verified with a synthetic test (a single-hand sample's slot 1 stays all-zero after mirroring; a two-hand sample's both slots remain populated).
3. **A data-quality filter was silently dropped during the two-hand rewrite.** The original single-hand loader rejected a hand's tracking data outright if MediaPipe's own confidence in it was too low for more than half the frames it appeared in (a real "this tracking is more noise than signal" gate). The two-hand rewrite of that extraction function omitted this check — measurably more inclusive (100% of sequences produced *some* data, versus 69% before) but measurably worse, because it started admitting unreliably-tracked hands the original loader had correctly thrown out: accuracy fell further, to 47.2%, still below baseline even after fixing bug #2. Fixed by reinstating the identical filter inside the new two-hand extraction code, verified by a synthetic test (a mostly-untracked hand is correctly rejected; a well-tracked hand is correctly kept) and confirmed correct in aggregate — the resulting sequence count matched the original single-hand filtered count exactly. Only after both fixes did ASL correctly land at 52.9%, above baseline as it should be.

See `ml/README.md`'s "Two hands, not one" section for the full technical trail, including the exact numbers at each step.

### 4.6 ASL: why the accuracy gap, and what closed part of it

ASL training uses Kaggle's *Google – Isolated Sign Language Recognition* competition dataset (`asl-signs`): 250 signs, 94,477 sequences, 21 signers, pre-extracted MediaPipe landmarks. Getting the full dataset took two real obstacles:

1. **Download quota.** The competition's per-file download API has an undocumented quota — after roughly 3,000 file-download requests in one session, every subsequent request failed (confirmed by probing 210 different signs with fresh, never-before-requested files: 0/210 succeeded). This initially capped the dataset at 14 of an intended 40 classes (535 sequences). Resolved with a full manual download via the Kaggle website's "Download All" button — a different, bulk endpoint not subject to the per-file quota.
2. **Scale bugs, once the real data landed.** At 94,477 files, a per-row Python loop that was fine at 535 sequences became impractical, and a latent NaN-propagation bug (MediaPipe briefly losing hand tracking leaves scattered per-landmark gaps, which silently poisoned samples through the resampling step) became a real correctness issue. Both fixed in `ml/src/dataset_asl.py` — see `ml/README.md` for detail.

With the real data, first single-hand result: 52.0% held-out accuracy — solid against 0.4% chance, but well below BdSL's. Landmark augmentation (mirroring, rotation, scale jitter, per-frame noise — synthesizing signer-style variation that only 21 real signers can't provide) closed part of the gap: 54.1% with 2 extra augmented copies of the training data, 54.5% with 4, essentially plateauing. Two-hand tracking (§4.5) was the next thing tried, on the reasoning that a single-hand-only representation necessarily drops information for ASL's many two-handed signs — but it only pushed the number to 52.9%, because (unlike BdSL) this specific dataset's recordings almost never actually have a usable second hand in frame (~0.1% of frames), so there was little second-hand signal available to gain from. The remaining gap looks structural rather than something either augmentation or two-hand tracking alone can close: most plausibly the combination of only 21 total signers (vs. BdSL's dataset) and 250 fine-grained classes. A larger or more diverse ASL word corpus, not a further pipeline change, is the most promising next step and is noted as a direction rather than attempted here.

### 4.7 ASL fingerspelling alphabet (A-Z) and digits (0-9)

Extending ASL past word signs was a specific ask: could the alphabet be added the way BdSL's digits/letters were? Unlike BdSL's digit/letter source, no pre-extracted landmark dataset for ASL fingerspelling was found trustworthy — one candidate (`srisahithis/american-sign-language-a-z-dataset-hand-landmarks`) turned out to have its "landmarks" drawn as visible red dots on top of the JPEGs, not exported coordinates, and unusable both as training data and as something to verify.

Rather than guess, this project generates its own landmarks: `ml/extract_asl_alphabet_landmarks.py` runs the actual MediaPipe `HandLandmarker` — the same Tasks-API model family, and literally the same `hand_landmarker.task` bundle, that the Flutter app's own plugin uses — over `grassknoted/asl-alphabet` (87,000 clean JPEGs, A-Z + space/del/nothing, capped at 400 images/letter here). A folder named "A" produces landmarks for whatever hand shape the real detector found in that folder's images; nothing is trusted from a third party's export. ASL had no digit vocabulary at all until real-device feedback asked for one directly (BdSL already had one) — `ml/extract_asl_digit_landmarks.py` does the identical thing over `rayeed045/american-sign-language-digit-dataset` (5,000 clean JPEGs, 0-9, capped at 400 images/digit).

Neither source has participant/signer identity (independent photos, not a multi-signer video corpus), so their portion of the train/val split is an ordinary random split, not the held-out-*signer* split every other model in this project uses — called out here rather than glossed over. Combined with the 250-word two-hand model into one 286-class model via `ml/train_asl_combined.py`: 9,364 of a possible 10,400 alphabet samples were extracted successfully (1,036 images had no detectable hand, concentrated on visually ambiguous letters like M and N), and 3,599 of a possible 4,000 digit samples (401 skipped, concentrated on digit "0" — an ASL closed-fist/O handshape that MediaPipe's detector reads far less reliably than an open hand, only 150/400 of that digit's images had a detectable hand at all). The combined model reached **61.7%** held-out accuracy overall, but that single blended number hides very different results per category: **57.9%** for words, **99.0%** for letters, **98.9%** for digits.

Those digit/letter numbers are genuine, but not directly comparable to the 57.9% word-only figure (or to BdSL's 82.8%): both are small, visually distinct vocabularies of clean, single, deliberately-posed photos — a much easier problem than 250 words performed as real signed video by 21 different people, and evaluated on the easier random split described above rather than a held-out-signer one. Shipped anyway, because each is a real, honestly-labeled addition on top of the real word model, not a way to make the headline number look better — the word-only figure remains the right one to compare against BdSL's 83.4%.

### 4.8 Category filtering, fingerspelling buffer, and offline autocomplete

Two more direct pieces of real-device feedback shaped the app on top of the model itself, neither requiring a retrain:

- **Category filtering** (`app/lib/services/sign_category.dart`, `sign_classifier.dart`'s `setAllowedCategories`): every label is tagged Word, Digit, or Letter by its naming convention (`digit_*`, `letter_*`, else Word), and the mode bar in §2.3 tells the classifier's inference step to zero out any logit outside the active category before taking the argmax. This is pure post-processing on the same trained model — restricting which of its already-learned classes are allowed to win, not changing what it learned.
- **Fingerspelling buffer** (`home_screen.dart`'s `_spellBuffer`): in Letters mode, committed letters append to an in-progress buffer instead of the sentence directly; the buffer flushes into the sentence as one word on mode switch, clear, speak, or accepting a suggestion. This directly fixes the "creating a whole word adds a space per character" complaint from real-device testing.
- **Offline word completion** (`app/lib/services/word_completer.dart`): as the buffer fills, up to 3 likely completions are suggested by prefix-matching against a bundled, frequency-ordered word list — Gboard-style tap-to-complete, but with zero network calls, so it works exactly as well with the phone in airplane mode as connected. One list per language: English (10,000 words, `google-10000-english`, MIT license) for ASL, Bangla (10,000 words, derived from `hermitdave/FrequencyWords`' OpenSubtitles-based Bengali list, code MIT / content CC BY-SA 4.0, filtered to Bengali-script tokens only since the raw subtitle corpus has real English-word contamination mixed in) for BdSL. This was a deliberate choice over a cloud predictive-text API, since the app's core accessibility premise is that it must work for a deaf or hard-of-hearing user regardless of their connectivity. One real limitation, documented rather than hidden: BdSL fingerspelling signs only cover Bangla's independent letter forms, not the dependent vowel signs (matra) real Bangla orthography uses mid-word, so a spelled prefix won't always character-match a word's actual spelling — this shows up as fewer suggestions for some words, never as a wrong or fabricated one. Both the underlying word-list lookup (`test/word_completer_test.dart`) and the display layer were checked directly after a "Bangla suggestions aren't showing" report: the lookup itself tested correctly (a bare consonant prefix alone has hundreds of real matches), but two real, separate issues were found and fixed anyway — the async dictionary load had no error handling at all, so any failure would leave suggestions silently and permanently empty with no indication why; and the suggestion chips rendered in a `Wrap`, which can spill onto a second line for longer text and get clipped by the fixed-height row (§2.1's camera-resize fix), swapped for a horizontally-scrolling single line that can never wrap.
- **Hand-shape control gestures, made independently switchable** (`AppState.controlGesturesEnabled`, an app-bar toggle next to Auto mode): the fist/thumbs-up/thumbs-down gestures are read from raw hand geometry, not the trained classifier, so their reliability doesn't track sign-recognition accuracy the same way — it varies more with lighting, hand shape, and camera angle. Rather than chase a perfect one-size-fits-all heuristic further, this gives users a genuine escape hatch: turning it off leaves hold-a-sign-to-add-it and pause-to-auto-speak (both classifier-driven) working exactly as before, and routes Delete/Clear/Speak to their manual buttons instead of reading them from a hand shape at all.

### 4.9 Saved phrases (device-local)

`app/lib/services/phrase_store.dart` persists two lists per language via `SharedPreferences`: phrases the user explicitly pinned (up to 50) and the last 20 sentences they spoke. A bottom sheet (`screens/phrases_sheet.dart`) lists both — tap to load a phrase into the sentence, or hit its speaker button to say it immediately.

Why this exists: every sentence in this app is built one sign at a time. A sentence the user needs regularly — an address, an allergy, "please call my brother" — costs that full effort each time. Saving it makes the cost one-time, which for the target user is accessibility, not convenience.

It is also the one place the app persists anything, so the privacy reasoning is explicit rather than incidental. Saved phrases skew personal and medical precisely *because* they are the things someone needs to say often. They are stored in app-private storage, never transmitted, and never synced — there is no account and no server to sync to, which is a structural property of the architecture rather than a promise. "Clear all" wipes both lists in both languages, and uninstalling takes them with it. Nothing derived from the camera is stored at any point.

Implemented as a bottom sheet rather than a route deliberately: pushing a route would tear down and re-initialize the camera underneath it, which is both slow and a visible flicker.

## 5. Datasets

| Dataset | Source | Size used | License / access | Role |
|---|---|---|---|---|
| **asl-signs** ("Google – Isolated Sign Language Recognition") | Kaggle competition | 94,477 sequences, all 250 classes, 21 signers | Kaggle competition terms | ASL training data — words (shipped) |
| **ASL Alphabet** | Kaggle dataset `grassknoted/asl-alphabet` (raw JPEGs; landmarks self-extracted via MediaPipe `HandLandmarker` — see §4.7) | 26 letters (A-Z), up to 400 images/letter | GPL-2.0 | ASL training data — fingerspelling alphabet (shipped, merged into the same model as asl-signs) |
| **ASL Digits** | Kaggle dataset `rayeed045/american-sign-language-digit-dataset` (raw JPEGs, derived from the Massey University 2D static ASL gesture dataset; landmarks self-extracted via MediaPipe `HandLandmarker` — see §4.7) | 10 digits (0-9), up to 400 images/digit | CC0-1.0 | ASL training data — digits (shipped, merged into the same model as asl-signs) |
| **BdSLW60** | Kaggle dataset `kaziredwanislam/bdslw60-landmark-data-trials` (pre-extracted MediaPipe landmarks); word gloss list transcribed from `hasaniut/bdslw60`'s own `BdSLW60.pdf` | 9,229 of 9,307 sequences, all 60 words, 18 signers | CC BY-NC-ND 4.0 (non-commercial, no derivatives) — **academic/demo use only**, flagged for before any public or commercial release | BdSL training data — words (shipped) |
| **Bangla Sign Language Dataset** (digits + alphabet) | Kaggle dataset `rayeed045/bangla-sign-language-dataset`, `CSV Files/Sign Digits - CSV Files/` and `CSV Files/Sign Letters - CSV Files/` | 10 digits + 37 alphabet letters, ~100 samples/class/signer, 10 signers each; labels independently verified via each CSV's own embedded image-path column | Per Kaggle dataset terms | BdSL training data — digits + alphabet letters (shipped, merged into the same model as BdSLW60) |
| **google-10000-english** | GitHub `first20hours/google-10000-english` | Top 10,000 words, unfiltered | MIT | App data — offline word-completion dictionary for ASL fingerspelling (shipped, `assets/dictionary/common_words.txt`) |
| **FrequencyWords (bn)** | GitHub `hermitdave/FrequencyWords`, `content/2018/bn/bn_full.txt` (OpenSubtitles-derived) | Top 10,000 Bengali-script words after filtering (raw list of 99,993 entries had real English-word contamination from subtitle credits/dialogue, stripped) | Code MIT, word-list content CC BY-SA 4.0 — attribution required if this app is ever distributed publicly | App data — offline word-completion dictionary for BdSL fingerspelling (shipped, `assets/dictionary/common_words_bn.txt`) |
| Bangla Sign Alphabet dataset | Kaggle dataset `rayeed045/bangla-sign-alphabet-dataset` (**note:** a different, unrelated dataset from the one above despite the similar name) | Investigated, not used | ODbL-1.0 | Considered for BdSL fingerspelling/digit recognition; rejected — only 20 classes from a single signer, and the `Label` column is a bare integer with no verified mapping to actual Bangla characters. See §6. |
| ASL A-Z Dataset + Hand Landmarks | Kaggle dataset `srisahithis/american-sign-language-a-z-dataset-hand-landmarks` | Investigated, not used | Apache 2.0 | Considered as a pre-extracted alternative to self-running MediaPipe for the ASL alphabet; rejected — its "landmarks" are drawn as red dots directly on the JPEGs, not exported coordinates. See §4.7. |
| BdSLW401 (401 words) | Kaggle datasets `sahebaakter/bdslw401-1st-processed` and `adibrafid/final-bdslw-401-keypoints` | Investigated, not used | CC0-1.0 (latter) | Considered as a larger BdSL vocabulary; rejected — the first candidate was an incomplete upload (2.3MB total), the second uses a 75-landmark pose+hands format incompatible with this app's hands-only contract and has unverified word-code labels. See §6. |
| Grammatical Facial Expressions | Kaggle dataset `prashant111/grammatical-facial-expressions` | Investigated, not used | Unknown | Considered for the proposal's facial-expression objective; rejected — annotated for Brazilian Sign Language (Libras), a different sign language with an incompatible per-frame annotation scheme, and has no hand landmarks. See §4.1 and §6. |
| AtoZ_3.1 + `cnn8grps_rad1_model.h5` | GitHub, Devansh-47/Sign-Language-To-Text-and-Speech-Conversion | Used only by the legacy `test.py`/`train.py` reference scripts, not the shipped app | Per that repository | Superseded raw-image ASL fingerspelling prototype (see §3) |

## 6. Known Limitations (Honest Status)

- **ASL word accuracy trails BdSL's** (57.9% word-only, vs. BdSL's 82.8%) — see §4.6 for why: fewer signers (21 vs. BdSL's), more fine-grained classes (250 vs. 60), and — unlike BdSL — a dataset that almost never has a second hand in frame to begin with (~0.1% of frames), so the two-hand upgrade in §4.5 that helped BdSL substantially could not help ASL nearly as much. ASL's digits and fingerspelling letters, by contrast, are both around 99% accurate (§4.7) — small, visually distinct, cleanly-posed vocabularies, a much easier problem than 250 words from real signed video. The finger-count heuristic engine (counts extended fingers, reports "0"–"5") still exists as a fallback, shown only if a trained model fails to load, and is explicitly badged "PROTOTYPE ENGINE" when it is what's active — never presented as the real recognizer.
- **Five BdSL letters do not work at all.** `Dda`, `Kaa`, `Laa`, `Ooo` and `Uuu` score zero recall on unseen signers even in Letters mode, where they compete only against other letters. They are struck through in the app's About screen rather than offered as if they worked — marked, not hidden, since they remain in the trained vocabulary and can still surface in a prediction. Two cheap fixes were tried: the position anchor recovered two other letters (`Haa`, `Naa`), and inverse-frequency class weighting did not help and slightly hurt macro-F1. The remaining constraint is the alphabet corpus itself — single photos from ten signers, a few dozen samples for the rarest letters.
- **BdSL vocabulary is 107 classes** (60 words, 10 digits, 37 alphabet letters); **ASL is 286 classes** (250 words, 26 fingerspelling letters, 10 digits) — neither has continuous-sentence grammar (word order, non-manual markers), and BdSL has no larger 401-word vocabulary. A 401-word expansion was investigated (§5) and set aside: both available candidates were either an incomplete upload or used an incompatible landmark format with unverified labels.
- **Facial expression was investigated and not shipped**, despite being named in the original proposal (§4.1) — no dataset was found pairing facial *expression* with ASL/BdSL grammar in a format compatible with this project. That remains a genuine data-availability gap.
- **Face landmarks as a spatial anchor are a separate question, and the earlier answer to it was wrong.** This documentation previously argued that face landmarks were unnecessary because no class in either vocabulary is distinguished by facial *expression*. That reasoning is correct about expression but conflated it with the face's other role: a *spatial reference point*. Many ASL signs are defined by where on the face the hand is placed, and the preprocessing in §4.1 normalizes each frame to the hand's own wrist — deleting exactly that information. A pre-registered test (`ml/analyze_face_hypothesis.py`) confirms the cost: 47 face-located ASL words average **44.2%** per-class recall against **54.1%** for the other 203 words, a 9.9-point gap, Mann-Whitney U one-sided **p = 0.0034**. The worst-affected classes were `chin` (5.7%), `lips` (9.2%), `tongue` (10.9%), and `cheek` (11.3%).

  **This has since been acted on, and it produced the project's largest single accuracy gain.** Rather than adding a face model, the fix reclaims what normalization already computes and throws away: the hand's in-frame origin and apparent scale, kept as a **position anchor** row appended to every frame (42 rows → 43). It costs nothing at inference, because the hand detector already reports the wrist's position. ASL words rose 53.4% → **57.9%**, and — the part that actually tests the hypothesis — the gain landed where it was predicted to: face-located words gained **+5.95** points against **+3.98** for everything else, narrowing the gap from 9.93 to 7.96 points (p 0.0034 → 0.0113). `lips`, `tongue`, and `cheek` all left the worst-ten list.

  The gap narrowed but did not close, and that is expected: the anchor gives position relative to the *camera frame*, not relative to the *face*, so it works only while the signer stays roughly still with respect to the camera. A true face reference point remains the highest-value next step, and asl-signs already contains the needed landmarks (currently discarded at extraction).
- **Gesture segmentation is a threshold heuristic, not learned.** The app now detects sign boundaries from motion energy — accumulate while the hand moves, emit when it settles — which removed the need to hold every sign deliberately and closed the train/inference mismatch (§4.4). It is still a rule with a fixed threshold, though: continuous signing that never settles falls back to the old rolling window, so live behaviour can still diverge from the offline numbers, which are computed on cleanly pre-segmented sequences.
- **Bangla text-to-speech and speech-to-text depend on the device.** Both require the phone's OS to have a Bangla voice/language model installed; the app detects this at runtime and tells the user plainly (falling back to English) rather than failing silently if it is missing.
- **Hand tracking has no true left/right handedness signal on-device**, only the geometric left-to-right-in-frame convention described in §4.5 — a deliberate, documented workaround for a real plugin limitation, not an oversight, but it means (for example) crossing your hands mid-sign can momentarily relabel which hand is in which slot.
- **The CPU delegate is slower than GPU would be**, chosen deliberately for cross-device reliability over raw speed (§4.4) — a real trade-off, not a free fix.

## 7. App Details

- **Name / package:** SignBridge, `com.bup.signbridge.sign_bridge`
- **Platform:** Android (Flutter), minimum SDK 24 (required by the MediaPipe hand-landmark plugin)
- **Key packages:** `camera`, `hand_landmarker` (MediaPipe hand landmarks via JNI, configured for up to 2 hands on the CPU delegate — see §4.4), `tflite_flutter` (on-device inference), `speech_to_text`, `flutter_tts`, `provider` (state management)
- **Icon and splash:** custom-designed mark — an open, fanned hand inside a speech bubble, on the app's brand blue, representing sign-to-speech translation directly
- **Brand palette:** blue (#2E6BB3 / #1E4A80) for chrome (app bar, navigation), warm amber (#F2A93B) reserved for the one thing that should visually pop on every screen — the recognized sign

## 8. Project Structure

```
SignBridge/
  app/    Flutter Android app
    lib/
      screens/    Sign tab, Listen tab, About screen, tab shell
      services/   sign classifiers (prototype + trained), sign categories,
                   offline word completer, control-gesture detector, Bangla
                   glossary
      state/      shared app state (language, category, sentence, settings)
      widgets/    camera landmark overlay, brand title
    assets/dictionary/  bundled offline word list for autocomplete
  ml/     Python training pipeline
    src/          landmark preprocessing, model architectures, dataset loaders
    train.py      CLI: trains a model, exports .tflite + labels.json
    models/       trained model outputs (per language/mode/architecture)
  design/         icon and splash artwork source (Python/Pillow script)
  docs/           this document, plus the build checklist
```

## 9. How to Build and Run

```bash
# App
cd app
flutter pub get
flutter run                       # debug, or:
flutter build apk --release       # release APK

# ML pipeline (see ml/README.md for full dataset setup)
cd ml
pip install -r requirements.txt

python train_bdsl_combined.py \
    --words-dir data/bdslw60 \
    --digits-dir "data/bdsl_digits/CSV Files/Sign Digits - CSV Files" \
    --letters-dir "data/bdsl_digits/CSV Files/Sign Letters - CSV Files" \
    --epochs 100 --augment-copies 3

python train.py --language asl --data-dir data/asl-signs --mode full \
    --arch cnn --epochs 40 --batch-size 128 --augment-copies 1
python extract_asl_alphabet_landmarks.py \
    --images-dir data/asl-alphabet/asl_alphabet_train/asl_alphabet_train
python train_asl_combined.py --epochs 40 --augment-copies 2
```

## 10. Team & Supervision

| Role | Name |
|---|---|
| Supervisor | Dr. Ahmedul Kabir, University of Dhaka |
| Student | Samiul Islam |
| Student | Ahnaf Sayed |
| Student | Abu Saeed Sabuj |

Program: MICT-2023, Department of ICT, Bangladesh University of Professionals.
