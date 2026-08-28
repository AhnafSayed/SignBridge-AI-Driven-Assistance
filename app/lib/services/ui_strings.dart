/// Short, high-frequency chrome text (nav labels, buttons, menu items) in
/// Bangla, keyed the same plain-lookup-table way bdsl_glossary.dart's maps
/// are -- not a full ARB/intl pipeline this project has no build-system
/// support for. Deliberately does NOT cover long prose (the About
/// screen's explanations, etc.) -- translating those without linguistic
/// review risks shipping something subtly wrong, which is a worse outcome
/// than leaving them in English for now.
const Map<String, String> _bn = {
  'sign': 'সাইন',
  'listen': 'শোনো',
  'dictionary': 'অভিধান',
  'delete': 'মুছুন',
  'clear': 'পরিষ্কার',
  'speak': 'বলুন',
  'more_options': 'আরও অপশন',
  'about_this_build': 'অ্যাপ সম্পর্কে',
  'emergency_sos': 'জরুরি এসওএস (SOS)',
  'saved_phrases': 'সেভ করা বাক্য',
  'search': 'খুঁজুন',
  'no_matches': 'কোনো মিল পাওয়া যায়নি',
  'all': 'সব',
  'words': 'শব্দ',
  'digits': 'সংখ্যা',
  'letters': 'অক্ষর',
  'theme': 'থিম',
  'send_sos_alert': 'এসওএস অ্যালার্ট পাঠান',
  'stop_alert': 'অ্যালার্ট বন্ধ করুন',
  'close': 'বন্ধ করুন',
  'save': 'সেভ করুন',
  'cancel': 'বাতিল',
  'exit': 'বেরিয়ে যান',
  'exit_signbridge_title': 'SignBridge থেকে বেরিয়ে যাবেন?',
  'auto_mode': 'অটো মোড',
  'hand_gestures': 'হ্যান্ড-শেপ জেসচার',
  'app_language': 'অ্যাপের ভাষা',
  'settings': 'সেটিংস',
  'appearance': 'দৃশ্যরূপ',
  'theme_system': 'ডিভাইসের সাথে মিলিয়ে',
  'theme_light': 'হালকা',
  'theme_dark': 'গাঢ়',
  'app_language_note':
      'ক্যামেরা কোন সাইন ভাষা পড়ছে তার থেকে আলাদা -- এটি শুধু অ্যাপের নিজস্ব '
      'বোতাম ও মেনুর ভাষা।',
  'recognition': 'শনাক্তকরণ',
  'auto_mode_on': 'সাইন ধরে রাখলে যোগ হবে, থামলে বলা হবে',
  'auto_mode_off': 'নিচের বোতাম ব্যবহার করুন',
  'hand_gestures_on': 'মুঠি (বুড়ো আঙুল বাইরে) / থাম্বস-আপ / থাম্বস-ডাউন বাক্য নিয়ন্ত্রণ করে',
  'hand_gestures_off': 'ডিলিট / ক্লিয়ার / স্পিক বোতাম ব্যবহার করুন',
};

/// Looks up [key] in Bangla when [bangla] is true (AppState.uiBangla --
/// the app's own chrome-language toggle, independent of which sign
/// language the camera is reading), otherwise returns [english]
/// unchanged. Never throws and never returns an empty string -- a
/// missing translation falls back to English rather than showing
/// nothing, the same never-fail contract [bengaliFor] already uses for
/// individual sign glosses.
String uiText(bool bangla, String key, String english) {
  if (!bangla) return english;
  return _bn[key] ?? english;
}
