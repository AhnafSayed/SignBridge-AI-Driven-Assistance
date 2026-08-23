import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// The one phone number the SOS feature texts, stored device-local like
/// everything else this app persists (see PhraseStore's own comment on
/// why -- the same reasoning applies here, more so: this is a real
/// person's phone number, not a phrase).
class EmergencyContactStore extends ChangeNotifier {
  static const _key = 'emergency_contact_number';
  static const _nameKey = 'emergency_contact_name';

  SharedPreferences? _prefs;
  String _number = '';
  String _name = '';

  bool get isReady => _prefs != null;
  String get number => _number;
  String get name => _name;
  bool get isConfigured => _number.trim().isNotEmpty;

  Future<void> init() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _number = prefs.getString(_key) ?? '';
      _name = prefs.getString(_nameKey) ?? '';
      _prefs = prefs;
      notifyListeners();
    } catch (e) {
      debugPrint('EmergencyContactStore: storage unavailable: $e');
    }
  }

  /// Manual entry -- no name is known for a hand-typed number, and any
  /// previously stored name almost certainly belongs to a *different*
  /// number now, so it's cleared rather than left silently mismatched.
  Future<void> setNumber(String value) async {
    _number = value.trim();
    _name = '';
    try {
      await _prefs?.setString(_key, _number);
      await _prefs?.remove(_nameKey);
    } catch (e) {
      debugPrint('EmergencyContactStore: failed to persist: $e');
    }
    notifyListeners();
  }

  /// Set together from the contact picker, which supplies both at once.
  Future<void> setContact({required String number, required String name}) async {
    _number = number.trim();
    _name = name.trim();
    try {
      await _prefs?.setString(_key, _number);
      if (_name.isEmpty) {
        await _prefs?.remove(_nameKey);
      } else {
        await _prefs?.setString(_nameKey, _name);
      }
    } catch (e) {
      debugPrint('EmergencyContactStore: failed to persist: $e');
    }
    notifyListeners();
  }
}
