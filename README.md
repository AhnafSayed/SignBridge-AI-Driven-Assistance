# SignBridge

![Flutter](https://img.shields.io/badge/Flutter-Dart-02569B?logo=flutter&logoColor=white)
![Android](https://img.shields.io/badge/Android-minSdk%2024-3DDC84?logo=android&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-00897B)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-on--device-FF6F00?logo=tensorflow&logoColor=white)
![Offline](https://img.shields.io/badge/Network-none%20required-informational)

![Bangladesh University of Professionals](app/assets/branding/bup_logo.webp)

SignBridge is a bilingual sign language recognition app for Android, built as a final year project at the Bangladesh University of Professionals. It reads American Sign Language and Bangla Sign Language through the phone's camera and speaks the recognized words and letters out loud, and it also listens to spoken speech and shows it as text, so a conversation can flow in both directions without needing an interpreter.

Everything runs on the device itself. There is no server, no account, and no camera frame or audio ever leaves the phone. That was a deliberate choice from the start, not just a privacy footnote: the app has to work for someone who may not have reliable internet access, and the things it processes, a person's face, their hands, their speech, are not the kind of data that should need to touch a server at all.

## Project background

This is the MICT-2023 final project for the Department of ICT, Faculty of Science and Technology, Bangladesh University of Professionals.

Supervisor: Dr. Ahmedul Kabir, University of Dhaka.

Team: Samiul Islam, Ahnaf Sayed, Abu Saeed Sabuj.

The full project report, including the methodology, the datasets used, every model's measured accuracy, and an honest account of what did not work along the way, was written as part of the academic submission and is not published in this repository. It is available on request.

## What it does

The Sign tab reads hand shapes through the camera using MediaPipe's hand landmark detector, and a trained model classifies them into words, digits, or fingerspelled letters, in either ASL or BdSL. Recognized signs build up into a sentence, which can be spoken aloud automatically or on demand. Hands-free operation is supported through a small set of hand shape gestures (a closed fist to delete, a thumbs up to speak, and so on), with a toggle to fall back to manual buttons if that proves unreliable for a given hand or lighting condition.

The Listen tab does the opposite direction: it transcribes the other person's speech into large on screen text, in either English or Bangla.

A Dictionary tab lets you search and browse every sign the app knows, with a tap to hear it pronounced. Phrases you use often can be saved for one tap reuse, and a set of starter phrases is included for people who have not built up their own history yet. An emergency SOS option is also available: it sounds a repeating alert and opens a pre filled text message to a chosen contact, though sending that message is always left to the user's own final tap, on purpose.

## Current model status

Both languages run real trained models rather than a placeholder. BdSL recognizes sixty words, ten digits, and thirty seven alphabet letters, at about eighty three percent accuracy on signers the model never saw during training. ASL recognizes two hundred fifty words, ten digits, and the full fingerspelling alphabet. Word accuracy for ASL is lower, around fifty eight percent on unseen signers, since two hundred fifty distinct word signs from limited training video is a genuinely harder problem than a small set of clearly distinct handshapes. Digits and letters in both languages score well above ninety eight percent on their own.

ASL's letters and digits are handled by a separate, direct classifier rather than the same sequence model used for words, since a static handshape does not need to be treated as a moving gesture. That classifier was also trained with a wider range of rotation in its training data, which measurably fixed a real problem found during device testing, where certain letters were misread if the signing hand was not held at close to the same angle as the training photos.

The full numbers, including what did not work and why, are in the project report described above.

## Repository layout

```
SignBridge/
  app/     Flutter Android application
  ml/      Python training and evaluation pipeline
  docs/    Supplementary technical notes
```

The `ml/data` folder, which holds the raw training datasets, is not included in this repository. Some of those datasets carry licenses that restrict redistribution, and even the ones that do not are too large to reasonably keep in version control. The scripts that expect them, and where to get each dataset, are documented in `ml/README.md`.

## Building the app

```
cd app
flutter pub get
flutter build apk --release
```

A signed release APK is also attached to this repository's releases page, for anyone who just wants to install it without building from source.

## Running the training pipeline

```
cd ml
pip install -r requirements.txt
```

From there, `ml/README.md` covers dataset setup and the individual training and evaluation scripts for each language and model.
