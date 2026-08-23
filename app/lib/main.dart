import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
import 'package:provider/provider.dart';

import 'app_theme.dart';
import 'screens/root_shell.dart';
import 'services/emergency_contact_store.dart';
import 'services/phrase_store.dart';
import 'state/app_state.dart';

Future<void> main() async {
  final widgetsBinding = WidgetsFlutterBinding.ensureInitialized();
  // Held past Flutter's first frame -- HomeScreen calls
  // FlutterNativeSplash.remove() itself once the camera and BdSL model
  // are actually ready, so the splash covers the loading time instead of
  // handing off to a blank screen + spinner.
  FlutterNativeSplash.preserve(widgetsBinding: widgetsBinding);

  // Locked to portrait. This is a real-device bug fix, not a stylistic
  // choice: nothing in the app was ever laid out for landscape -- the
  // camera preview's aspect ratio, the landmark overlay's coordinate
  // mapping (built from `sensorOrientation`, which assumes the device is
  // held upright), and the fixed-height suggestion/mode bars all have a
  // portrait frame baked in. Rotating produced visibly broken layout
  // (widgets overlapping, the landmark overlay misaligned with the
  // camera feed it's drawn over) rather than a working alternate layout,
  // which is what real-device testing reported.
  //
  // The fix a full landscape mode would need -- reflowing the camera +
  // controls layout, and re-deriving the overlay transform for a rotated
  // sensor -- is a real UI project on its own, disproportionate to a
  // one-handed, phone-held-up signing interaction that has no obvious
  // benefit from landscape anyway. Locking portrait, the same choice
  // barcode- and ID-scanning apps make for the same reason (a live camera
  // overlay is exactly the kind of UI that breaks under free rotation),
  // prevents the broken state from ever being reachable rather than
  // leaving it reachable and broken.
  await SystemChrome.setPreferredOrientations(
    [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown],
  );

  runApp(const SignBridgeApp());
}

class SignBridgeApp extends StatelessWidget {
  const SignBridgeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AppState()),
        // Constructed empty and filled from device storage in the
        // background (PhraseStore.init). Deliberately not awaited: saved
        // phrases are a convenience, and the camera/recognition path must
        // never be gated on a storage read. Until it loads, `isReady` is
        // false and the phrase controls stay disabled.
        ChangeNotifierProvider(create: (_) => PhraseStore()..init()),
        ChangeNotifierProvider(create: (_) => EmergencyContactStore()..init()),
      ],
      child: Consumer<AppState>(
        builder: (context, appState, _) => MaterialApp(
          title: 'SignBridge',
          debugShowCheckedModeBanner: false,
          theme: AppTheme.light(),
          darkTheme: AppTheme.dark(),
          themeMode: appState.themeMode,
          home: const RootShell(),
        ),
      ),
    );
  }
}
