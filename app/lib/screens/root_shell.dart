import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show SystemNavigator;
import 'package:provider/provider.dart';

import '../services/ui_strings.dart';
import '../state/app_state.dart';
import 'dictionary_screen.dart';
import 'home_screen.dart';
import 'listen_screen.dart';

/// Switches between the app's three destinations:
///  - Sign: camera reads the impaired user's signs, speaks them aloud.
///  - Listen: microphone reads the other person's speech, shows it as text.
///  - Dictionary: a searchable, tap-to-pronounce reference -- browsing it
///    needs neither camera nor mic, so it just pauses both of the other
///    two rather than starting anything of its own.
///
/// Both tabs stay mounted for the app's lifetime (via a [PageView], which
/// also gets swipe-between-tabs navigation) and are individually paused/
/// resumed instead of being destroyed and recreated on every switch --
/// see the class docs on `HomeScreenState` for why that matters (a real
/// crash risk, not just a performance nicety).
class RootShell extends StatefulWidget {
  const RootShell({super.key});

  @override
  State<RootShell> createState() => _RootShellState();
}

class _RootShellState extends State<RootShell> {
  static const _signPage = 0;
  static const _listenPage = 1;
  static const _dictionaryPage = 2;
  // How long a second back gesture/press counts as "the same attempt" as
  // the first, rather than a fresh one that needs its own confirmation.
  static const _exitConfirmWindow = Duration(seconds: 3);

  final _pageController = PageController();
  final _homeKey = GlobalKey<HomeScreenState>();
  final _listenKey = GlobalKey<ListenScreenState>();
  int _index = _signPage;

  DateTime? _lastExitAttempt;
  bool _exitDialogShowing = false;

  void _goToPage(int index) {
    _pageController.animateToPage(
      index,
      duration: const Duration(milliseconds: 260),
      curve: Curves.easeOutCubic,
    );
  }

  void _onPageChanged(int index) {
    setState(() => _index = index);
    switch (index) {
      case _signPage:
        _listenKey.currentState?.pause();
        _homeKey.currentState?.resume();
      case _listenPage:
        _homeKey.currentState?.pause();
        _listenKey.currentState?.ensureStarted();
      case _dictionaryPage:
        // Needs neither the camera nor the mic, so both of the other
        // tabs' hardware just pauses -- nothing to start here.
        _homeKey.currentState?.pause();
        _listenKey.currentState?.pause();
      default:
        _homeKey.currentState?.pause();
        _listenKey.currentState?.pause();
    }
  }

  /// Handles a back gesture/press while already on the Sign tab -- there's
  /// nowhere left "up" to go, so Android's default behavior would be to
  /// exit the app immediately on the very first swipe. That's one swipe
  /// away from an accidental exit (mid-conversation, holding a phone with
  /// one hand while signing with the other, an edge-swipe back gesture is
  /// an easy accidental input) with no way to recover unsaved context.
  ///
  /// This is the standard Android "press back again to exit" pattern,
  /// just triggered by the gesture-nav back-swipe instead of a button --
  /// Flutter's [PopScope] doesn't distinguish between the two, so this one
  /// handler covers both a 3-button nav bar's back button and a gesture-
  /// nav edge swipe identically. First attempt shows a confirmation
  /// dialog; a second attempt within [_exitConfirmWindow] -- whether the
  /// dialog is still showing or was already dismissed -- exits directly,
  /// without needing to tap anything.
  Future<void> _handleBackFromSignTab() async {
    final now = DateTime.now();
    final within = _lastExitAttempt != null && now.difference(_lastExitAttempt!) < _exitConfirmWindow;
    if (_exitDialogShowing || within) {
      SystemNavigator.pop();
      return;
    }
    _lastExitAttempt = now;
    _exitDialogShowing = true;
    final bangla = context.read<AppState>().uiBangla;
    final shouldExit = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(uiText(bangla, 'exit_signbridge_title', 'Exit SignBridge?')),
        content: const Text('Swipe back (or press back) again to exit, or tap Exit below.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: Text(uiText(bangla, 'cancel', 'Cancel')),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: Text(uiText(bangla, 'exit', 'Exit')),
          ),
        ],
      ),
    );
    _exitDialogShowing = false;
    if (shouldExit == true) {
      SystemNavigator.pop();
    }
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bangla = context.watch<AppState>().uiBangla;
    return PopScope(
      // Never let the system pop this route directly -- both the Listen
      // tab (go back to Sign) and the Sign tab (exit-confirmation above)
      // need to intercept and decide for themselves what a back gesture
      // means here, rather than letting Android just close the app.
      canPop: false,
      onPopInvokedWithResult: (didPop, result) {
        if (didPop) return;
        if (_index == _signPage) {
          _handleBackFromSignTab();
        } else {
          _goToPage(_signPage);
        }
      },
      child: Scaffold(
        body: PageView(
          controller: _pageController,
          onPageChanged: _onPageChanged,
          children: [
            HomeScreen(key: _homeKey),
            ListenScreen(key: _listenKey),
            const DictionaryScreen(),
          ],
        ),
        bottomNavigationBar: NavigationBar(
          selectedIndex: _index,
          onDestinationSelected: _goToPage,
          destinations: [
            NavigationDestination(
              icon: const Icon(Icons.front_hand_outlined),
              label: uiText(bangla, 'sign', 'Sign'),
            ),
            NavigationDestination(
              icon: const Icon(Icons.mic_none_outlined),
              label: uiText(bangla, 'listen', 'Listen'),
            ),
            NavigationDestination(
              icon: const Icon(Icons.menu_book_outlined),
              label: uiText(bangla, 'dictionary', 'Dictionary'),
            ),
          ],
        ),
      ),
    );
  }
}
