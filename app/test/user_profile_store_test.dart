import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sign_bridge/services/user_profile_store.dart';

/// Covers the text-field half of UserProfileStore -- the same
/// SharedPreferences-backed persistence EmergencyContactStore already has
/// tests for. The photo half (setPhoto/clearPhoto) touches the real
/// filesystem via path_provider, which needs a device/platform channel to
/// mock meaningfully, so it isn't unit-tested here -- consistent with how
/// this project already leaves the contact-picker's platform side
/// untested and only covers the pure logic around it.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() => SharedPreferences.setMockInitialValues({}));

  Future<UserProfileStore> newStore() async {
    final s = UserProfileStore();
    await s.init();
    return s;
  }

  test('starts empty, becomes ready after init', () async {
    final s = UserProfileStore();
    expect(s.isReady, isFalse);
    await s.init();
    expect(s.isReady, isTrue);
    expect(s.name, isEmpty);
    expect(s.bloodGroup, isEmpty);
    expect(s.allergies, isEmpty);
    expect(s.conditions, isEmpty);
    expect(s.medications, isEmpty);
    expect(s.hasAnyMedicalInfo, isFalse);
    expect(s.hasPhoto, isFalse);
  });

  test('setProfile trims whitespace on every field', () async {
    final s = await newStore();
    await s.setProfile(
      name: '  Samiul  ',
      bloodGroup: ' O+ ',
      allergies: '  Penicillin  ',
      conditions: ' Type 1 diabetes ',
      medications: ' Insulin ',
    );
    expect(s.name, 'Samiul');
    expect(s.bloodGroup, 'O+');
    expect(s.allergies, 'Penicillin');
    expect(s.conditions, 'Type 1 diabetes');
    expect(s.medications, 'Insulin');
  });

  test('hasAnyMedicalInfo is true if even one medical field is set', () async {
    final s = await newStore();
    await s.setProfile(name: '', bloodGroup: 'O+', allergies: '', conditions: '', medications: '');
    expect(s.hasAnyMedicalInfo, isTrue);
  });

  test('persists across a reload -- the point of storing it at all', () async {
    final s1 = await newStore();
    await s1.setProfile(
      name: 'Samiul',
      bloodGroup: 'O+',
      allergies: 'Penicillin',
      conditions: 'Type 1 diabetes',
      medications: 'Insulin',
    );

    final s2 = UserProfileStore();
    await s2.init();
    expect(s2.name, 'Samiul');
    expect(s2.bloodGroup, 'O+');
    expect(s2.allergies, 'Penicillin');
    expect(s2.conditions, 'Type 1 diabetes');
    expect(s2.medications, 'Insulin');
  });

  test('notifies listeners on change', () async {
    final s = await newStore();
    var notified = false;
    s.addListener(() => notified = true);
    await s.setProfile(name: 'Samiul', bloodGroup: '', allergies: '', conditions: '', medications: '');
    expect(notified, isTrue);
  });

  test('each field can be left empty independently', () async {
    final s = await newStore();
    await s.setProfile(
      name: 'Samiul',
      bloodGroup: '',
      allergies: '',
      conditions: '',
      medications: '',
    );
    expect(s.name, 'Samiul');
    expect(s.bloodGroup, isEmpty);
    expect(s.hasAnyMedicalInfo, isFalse);
  });
}
