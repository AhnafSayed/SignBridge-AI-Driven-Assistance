import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// The user's own name, medical details, and photo -- all optional, all
/// device-local (same reasoning as EmergencyContactStore and PhraseStore:
/// this is more personal than either, so it stays on this device only, no
/// account, no server, ever). Nothing in this app requires filling any of
/// this in; it exists to make the SOS message more useful to whoever
/// receives it ("From: Samiul, blood group O+, allergic to penicillin"
/// instead of an anonymous "Emergency, I need help") and to let someone
/// helping the user in person -- a first responder, a stranger -- see who
/// they are and their medical details at a glance from the profile screen
/// itself.
///
/// Medical information is split into separate fields (blood group,
/// allergies, conditions, medications) rather than one free-text box --
/// each is short enough to scan at a glance in an emergency, which a
/// single paragraph mixing all four is not, and it means the SOS message
/// can render "None" for a field the user left blank instead of just
/// omitting it silently (a first responder reading "Allergies: None" is
/// meaningfully different from not being told anything about allergies
/// at all).
class UserProfileStore extends ChangeNotifier {
  static const _nameKey = 'profile_name';
  static const _bloodGroupKey = 'profile_blood_group';
  static const _allergiesKey = 'profile_allergies';
  static const _conditionsKey = 'profile_conditions';
  static const _medicationsKey = 'profile_medications';
  static const _photoPathKey = 'profile_photo_path';
  static const _photoFileName = 'profile_photo.jpg';

  SharedPreferences? _prefs;
  String _name = '';
  String _bloodGroup = '';
  String _allergies = '';
  String _conditions = '';
  String _medications = '';
  String? _photoPath;

  bool get isReady => _prefs != null;
  String get name => _name;
  String get bloodGroup => _bloodGroup;
  String get allergies => _allergies;
  String get conditions => _conditions;
  String get medications => _medications;
  String? get photoPath => _photoPath;
  bool get hasPhoto => _photoPath != null && File(_photoPath!).existsSync();
  bool get hasAnyMedicalInfo =>
      bloodGroup.isNotEmpty || allergies.isNotEmpty || conditions.isNotEmpty || medications.isNotEmpty;

  Future<void> init() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _name = prefs.getString(_nameKey) ?? '';
      _bloodGroup = prefs.getString(_bloodGroupKey) ?? '';
      _allergies = prefs.getString(_allergiesKey) ?? '';
      _conditions = prefs.getString(_conditionsKey) ?? '';
      _medications = prefs.getString(_medicationsKey) ?? '';
      final storedPath = prefs.getString(_photoPathKey);
      // A stored path only counts if the file it points to still exists --
      // the app's own document directory can be cleared by the OS (low
      // storage, a fresh install over an uninstall) independently of
      // SharedPreferences, and a dangling path would otherwise show a
      // broken image forever.
      _photoPath = storedPath != null && File(storedPath).existsSync() ? storedPath : null;
      _prefs = prefs;
      notifyListeners();
    } catch (e) {
      debugPrint('UserProfileStore: storage unavailable: $e');
    }
  }

  Future<void> setProfile({
    required String name,
    required String bloodGroup,
    required String allergies,
    required String conditions,
    required String medications,
  }) async {
    _name = name.trim();
    _bloodGroup = bloodGroup.trim();
    _allergies = allergies.trim();
    _conditions = conditions.trim();
    _medications = medications.trim();
    try {
      await _prefs?.setString(_nameKey, _name);
      await _prefs?.setString(_bloodGroupKey, _bloodGroup);
      await _prefs?.setString(_allergiesKey, _allergies);
      await _prefs?.setString(_conditionsKey, _conditions);
      await _prefs?.setString(_medicationsKey, _medications);
    } catch (e) {
      debugPrint('UserProfileStore: failed to persist: $e');
    }
    notifyListeners();
  }

  /// Copies [source] into this app's own document directory under a fixed
  /// name, replacing any previous photo, then stores that path -- not the
  /// original picked file's path. `image_picker` hands back a path into a
  /// cache directory the OS can clear at any time (it's meant for
  /// one-shot upload flows, not long-term storage), so a profile photo
  /// that's still supposed to be there next week has to live somewhere
  /// this app actually owns.
  Future<void> setPhoto(File source) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final dest = File('${dir.path}/$_photoFileName');
      await source.copy(dest.path);
      _photoPath = dest.path;
      await _prefs?.setString(_photoPathKey, dest.path);
      notifyListeners();
    } catch (e) {
      debugPrint('UserProfileStore: failed to save photo: $e');
    }
  }

  Future<void> clearPhoto() async {
    final path = _photoPath;
    _photoPath = null;
    try {
      await _prefs?.remove(_photoPathKey);
      if (path != null) {
        final file = File(path);
        if (await file.exists()) await file.delete();
      }
    } catch (e) {
      debugPrint('UserProfileStore: failed to clear photo: $e');
    }
    notifyListeners();
  }
}
