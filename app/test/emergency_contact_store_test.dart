import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sign_bridge/services/emergency_contact_store.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() => SharedPreferences.setMockInitialValues({}));

  Future<EmergencyContactStore> newStore() async {
    final s = EmergencyContactStore();
    await s.init();
    return s;
  }

  test('starts empty and unconfigured, becomes ready after init', () async {
    final s = EmergencyContactStore();
    expect(s.isReady, isFalse);
    await s.init();
    expect(s.isReady, isTrue);
    expect(s.isConfigured, isFalse);
    expect(s.number, isEmpty);
  });

  test('setNumber trims whitespace and reports configured', () async {
    final s = await newStore();
    await s.setNumber('  +1 555 123 4567  ');
    expect(s.number, '+1 555 123 4567');
    expect(s.isConfigured, isTrue);
  });

  test('an all-whitespace number is not configured', () async {
    final s = await newStore();
    await s.setNumber('   ');
    expect(s.isConfigured, isFalse);
  });

  test('persists across a reload -- the point of storing it at all', () async {
    final s1 = await newStore();
    await s1.setNumber('+15551234567');

    final s2 = EmergencyContactStore();
    await s2.init();
    expect(s2.number, '+15551234567');
    expect(s2.isConfigured, isTrue);
  });

  test('notifies listeners on change', () async {
    final s = await newStore();
    var notified = false;
    s.addListener(() => notified = true);
    await s.setNumber('+15551234567');
    expect(notified, isTrue);
  });

  test('setContact stores name and number together', () async {
    final s = await newStore();
    await s.setContact(number: '+15551234567', name: 'Jane Doe');
    expect(s.number, '+15551234567');
    expect(s.name, 'Jane Doe');
  });

  test('setNumber clears any previously stored name -- it would otherwise mislabel a different number', () async {
    final s = await newStore();
    await s.setContact(number: '+15551234567', name: 'Jane Doe');
    await s.setNumber('+19998887777');
    expect(s.name, isEmpty);
  });

  test('name persists across a reload alongside the number', () async {
    final s1 = await newStore();
    await s1.setContact(number: '+15551234567', name: 'Jane Doe');

    final s2 = EmergencyContactStore();
    await s2.init();
    expect(s2.number, '+15551234567');
    expect(s2.name, 'Jane Doe');
  });
}
