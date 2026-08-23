import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sign_bridge/services/phrase_store.dart';
import 'package:sign_bridge/state/app_state.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() => SharedPreferences.setMockInitialValues({}));

  Future<PhraseStore> newStore() async {
    final s = PhraseStore();
    await s.init();
    return s;
  }

  test('starts empty and becomes ready after init', () async {
    final s = PhraseStore();
    expect(s.isReady, isFalse);
    await s.init();
    expect(s.isReady, isTrue);
    expect(s.savedFor(SignLanguage.asl), isEmpty);
    expect(s.recentFor(SignLanguage.asl), isEmpty);
  });

  test('saves a phrase and reports it as saved', () async {
    final s = await newStore();
    await s.savePhrase(SignLanguage.asl, 'thank you');
    expect(s.savedFor(SignLanguage.asl), ['thank you']);
    expect(s.isSaved(SignLanguage.asl, 'thank you'), isTrue);
  });

  test('phrases are kept separate per language', () async {
    final s = await newStore();
    await s.savePhrase(SignLanguage.asl, 'thank you');
    await s.savePhrase(SignLanguage.bdsl, 'ধন্যবাদ');
    expect(s.savedFor(SignLanguage.asl), ['thank you']);
    expect(s.savedFor(SignLanguage.bdsl), ['ধন্যবাদ']);
    expect(s.isSaved(SignLanguage.asl, 'ধন্যবাদ'), isFalse);
  });

  test('re-saving moves to top instead of duplicating', () async {
    final s = await newStore();
    await s.savePhrase(SignLanguage.asl, 'one');
    await s.savePhrase(SignLanguage.asl, 'two');
    await s.savePhrase(SignLanguage.asl, 'one');
    expect(s.savedFor(SignLanguage.asl), ['one', 'two']);
  });

  test('empty and whitespace-only phrases are ignored', () async {
    final s = await newStore();
    await s.savePhrase(SignLanguage.asl, '');
    await s.savePhrase(SignLanguage.asl, '   ');
    await s.recordSpoken(SignLanguage.asl, '  ');
    expect(s.savedFor(SignLanguage.asl), isEmpty);
    expect(s.recentFor(SignLanguage.asl), isEmpty);
  });

  test('recent history dedupes and keeps newest first', () async {
    final s = await newStore();
    await s.recordSpoken(SignLanguage.asl, 'hello');
    await s.recordSpoken(SignLanguage.asl, 'goodbye');
    await s.recordSpoken(SignLanguage.asl, 'hello');
    expect(s.recentFor(SignLanguage.asl), ['hello', 'goodbye']);
  });

  test('recent history is capped', () async {
    final s = await newStore();
    for (var i = 0; i < PhraseStore.maxRecent + 10; i++) {
      await s.recordSpoken(SignLanguage.asl, 'phrase $i');
    }
    expect(s.recentFor(SignLanguage.asl).length, PhraseStore.maxRecent);
    // Newest retained, oldest dropped.
    expect(s.recentFor(SignLanguage.asl).first, 'phrase ${PhraseStore.maxRecent + 9}');
    expect(s.recentFor(SignLanguage.asl), isNot(contains('phrase 0')));
  });

  test('saved list is capped', () async {
    final s = await newStore();
    for (var i = 0; i < PhraseStore.maxSaved + 5; i++) {
      await s.savePhrase(SignLanguage.asl, 'p$i');
    }
    expect(s.savedFor(SignLanguage.asl).length, PhraseStore.maxSaved);
  });

  test('removeSaved deletes only the target', () async {
    final s = await newStore();
    await s.savePhrase(SignLanguage.asl, 'keep');
    await s.savePhrase(SignLanguage.asl, 'drop');
    await s.removeSaved(SignLanguage.asl, 'drop');
    expect(s.savedFor(SignLanguage.asl), ['keep']);
  });

  test('clearRecent leaves saved phrases intact', () async {
    final s = await newStore();
    await s.savePhrase(SignLanguage.asl, 'kept');
    await s.recordSpoken(SignLanguage.asl, 'transient');
    await s.clearRecent(SignLanguage.asl);
    expect(s.recentFor(SignLanguage.asl), isEmpty);
    expect(s.savedFor(SignLanguage.asl), ['kept']);
  });

  test('clearAll erases both languages completely', () async {
    final s = await newStore();
    await s.savePhrase(SignLanguage.asl, 'a');
    await s.savePhrase(SignLanguage.bdsl, 'b');
    await s.recordSpoken(SignLanguage.asl, 'c');
    await s.clearAll();
    for (final lang in SignLanguage.values) {
      expect(s.savedFor(lang), isEmpty);
      expect(s.recentFor(lang), isEmpty);
    }
  });

  test('phrases persist across a reload -- the point of the feature', () async {
    final first = await newStore();
    await first.savePhrase(SignLanguage.bdsl, 'আমার নাম');
    await first.recordSpoken(SignLanguage.bdsl, 'ধন্যবাদ');

    final second = PhraseStore();
    await second.init();
    expect(second.savedFor(SignLanguage.bdsl), ['আমার নাম']);
    expect(second.recentFor(SignLanguage.bdsl), ['ধন্যবাদ']);
  });

  test('returned lists are unmodifiable -- callers cannot corrupt the store', () async {
    final s = await newStore();
    await s.savePhrase(SignLanguage.asl, 'x');
    expect(() => s.savedFor(SignLanguage.asl).add('y'), throwsUnsupportedError);
  });
}
