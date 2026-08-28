import 'package:flutter/material.dart';
import 'package:flutter_contacts/flutter_contacts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';

import '../app_theme.dart';
import '../services/emergency_contact_store.dart';
import '../services/location_service.dart';
import '../services/sos_service.dart';
import '../services/user_profile_store.dart';

/// Emergency alert: configure one contact number, then trigger a loud
/// repeating spoken alert plus a pre-filled text message the user still
/// taps "send" on themselves -- see sos_service.dart's own comment for
/// why sending is not fully automatic.
///
/// A bottom sheet, like PhrasesSheet, so triggering an alert never tears
/// down the camera preview underneath it -- the two seconds this takes
/// should not cost the user their place in an ongoing conversation, in
/// exactly the moment they may most need it to keep working.
class SosSheet extends StatefulWidget {
  const SosSheet({super.key, this.startInEditing = false});

  /// Opens straight into "set/change contact" instead of the alert
  /// trigger screen -- used by the profile screen's own "Emergency
  /// contact" shortcut, so setting it up doesn't require first opening
  /// the alert screen and then finding the edit option inside it.
  final bool startInEditing;

  static Future<void> show(BuildContext context, {bool startInEditing = false}) {
    return showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      isDismissible: false, // an active alert should not vanish on a stray tap outside it
      enableDrag: false,
      builder: (_) => SosSheet(startInEditing: startInEditing),
    );
  }

  @override
  State<SosSheet> createState() => _SosSheetState();
}

class _SosSheetState extends State<SosSheet> {
  static const _alertMessage = 'Emergency. I need help. This message was sent from SignBridge.';

  final _controller = TextEditingController();
  final _sos = SosService();
  bool _active = false;
  late bool _editing;
  // Set only by the contact picker, and cleared the moment the user
  // types in the number field by hand -- a name saved alongside a number
  // the user then changed manually would silently mislabel it.
  String? _pickedName;

  @override
  void initState() {
    super.initState();
    _editing = widget.startInEditing;
  }

  @override
  void dispose() {
    _sos.dispose();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final contactStore = context.watch<EmergencyContactStore>();
    final configured = contactStore.isConfigured;
    if (_editing || !configured) {
      return _buildSetup(context, contactStore, configured);
    }
    return _buildAlert(context, contactStore);
  }

  Widget _buildSetup(BuildContext context, EmergencyContactStore store, bool configured) {
    if (_controller.text.isEmpty) _controller.text = store.number;
    // The phone-number field opens the keyboard, and this sheet has no
    // Scaffold of its own to get the usual resizeToAvoidBottomInset
    // behavior for free -- without adding the keyboard's own height here,
    // Save sits exactly where it would with no keyboard open at all and
    // ends up hidden behind it instead of pushed above it.
    final keyboardInset = MediaQuery.of(context).viewInsets.bottom;
    return SafeArea(
      child: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(20, 8, 20, 24 + keyboardInset),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // A coral-tinted header, not just plain text -- this is the
            // one setting in the app where getting it wrong (or leaving
            // it unset) has a real consequence in an actual emergency, so
            // it should read with a bit more visual weight than an
            // ordinary settings field.
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: AppColors.coral.withValues(alpha: 0.10),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppColors.coral.withValues(alpha: 0.35)),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.emergency, color: AppColors.coral),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(configured ? 'Change emergency contact' : 'Set an emergency contact',
                            style: Theme.of(context).textTheme.titleMedium
                                ?.copyWith(fontWeight: FontWeight.w700)),
                        const SizedBox(height: 6),
                        const Text(
                          'When you trigger an SOS alert, this app sounds a repeating '
                          'spoken alert, and lets you reach this number by text, '
                          'WhatsApp, or a phone call with an emergency message -- and a '
                          'map link to your current location, if you allow it -- already '
                          'filled in. You still tap send or call yourself, nothing is '
                          'sent automatically, and this number stays on this device only.',
                          style: TextStyle(color: AppColors.inkSoft, height: 1.45, fontSize: 12.5),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            if (_pickedName != null && _pickedName!.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  children: [
                    const Icon(Icons.check_circle, size: 14, color: AppColors.primary),
                    const SizedBox(width: 6),
                    Text('From contacts: $_pickedName',
                        style: const TextStyle(fontSize: 12.5, color: AppColors.primary)),
                  ],
                ),
              ),
            TextField(
              controller: _controller,
              keyboardType: TextInputType.phone,
              onChanged: (_) => setState(() => _pickedName = null),
              decoration: const InputDecoration(
                labelText: 'Phone number',
                border: OutlineInputBorder(),
                hintText: '+1 555 123 4567',
              ),
            ),
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: _pickContact,
              icon: const Icon(Icons.contacts_outlined, size: 18),
              label: const Text('Pick from contacts'),
              style: OutlinedButton.styleFrom(minimumSize: const Size(double.infinity, 44)),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                if (configured)
                  TextButton(
                    onPressed: () => setState(() => _editing = false),
                    child: const Text('Cancel'),
                  ),
                const Spacer(),
                FilledButton(
                  onPressed: _controller.text.trim().isEmpty
                      ? null
                      : () async {
                          await store.setContact(
                            number: _controller.text,
                            name: _pickedName ?? '',
                          );
                          if (mounted) setState(() => _editing = false);
                        },
                  child: const Text('Save'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// Opens the OS's own native contact picker -- not this app's contact
  /// list (it has none), just a one-time "pick a number" handoff to the
  /// Contacts app. Android still requires READ_CONTACTS for this despite
  /// the picker UI living entirely in the Contacts app's own process, so
  /// this can throw on a device where that permission was denied; caught
  /// and surfaced rather than left to crash the sheet.
  Future<void> _pickContact() async {
    try {
      // flutter_contacts' own picker does not request READ_CONTACTS
      // itself in this version -- it just throws if the permission isn't
      // already granted, with nothing prompting the user to grant it.
      // Requesting it explicitly first (the same permission_handler
      // pattern already used for camera/microphone elsewhere in this
      // app) is what actually makes Android's grant dialog appear.
      final status = await Permission.contacts.request();
      if (!status.isGranted) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: const Text('Contacts permission is needed to pick a number'),
              action: SnackBarAction(label: 'Settings', onPressed: openAppSettings),
            ),
          );
        }
        return;
      }
      var contact = await FlutterContacts.native.showPicker(properties: {ContactProperty.phone});
      // Some devices return the picked contact with only id/displayName
      // populated even though phone was requested -- a known rough edge
      // of the picker's platform side, not something this code controls.
      // A contact with a real id but no phones is exactly that case, not
      // proof the contact has no number, so re-fetching it explicitly by
      // id is the difference between "the picker sometimes doesn't attach
      // the number" and "the number silently never appears" from the
      // user's side.
      if ((contact?.phones.isEmpty ?? true) && contact?.id != null) {
        contact = await FlutterContacts.get(contact!.id!, properties: {ContactProperty.phone});
      }
      final phone = contact?.phones.isNotEmpty == true ? contact!.phones.first : null;
      final number = phone?.normalizedNumber?.trim().isNotEmpty == true
          ? phone!.normalizedNumber!.trim()
          : phone?.number.trim();
      if (number != null && number.isNotEmpty) {
        setState(() {
          _controller.text = number;
          _pickedName = contact?.displayName?.trim();
        });
      } else if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('That contact has no phone number saved')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not open contacts: $e')),
        );
      }
    }
  }

  Widget _buildAlert(BuildContext context, EmergencyContactStore store) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.emergency, color: AppColors.coral, size: 28),
                const SizedBox(width: 10),
                Expanded(
                  child: Text('Emergency SOS', style: Theme.of(context).textTheme.titleLarge),
                ),
                PopupMenuButton<String>(
                  tooltip: 'Contact options',
                  onSelected: (value) {
                    if (value == 'change') {
                      setState(() => _editing = true);
                    } else if (value == 'remove') {
                      _confirmAndRemoveContact(store);
                    }
                  },
                  itemBuilder: (context) => const [
                    PopupMenuItem(value: 'change', child: Text('Change contact')),
                    PopupMenuItem(
                      value: 'remove',
                      child: Text('Remove contact', style: TextStyle(color: AppColors.coral)),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              store.name.isNotEmpty ? 'Contact: ${store.name} (${store.number})' : 'Contact: ${store.number}',
              style: TextStyle(color: AppColors.inkSoft),
            ),
            const SizedBox(height: 20),
            if (!_active) ...[
              Text(
                'This will sound a repeating spoken alert on this phone and open '
                'a text message to your contact with an emergency message -- and '
                'your current location, if you allow it -- ready to send. Nothing '
                'is sent until you tap send yourself, and once the alert is '
                'active you can also reach them by WhatsApp or call.',
                style: TextStyle(color: AppColors.inkSoft, height: 1.5),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                    backgroundColor: AppColors.coral,
                    minimumSize: const Size.fromHeight(52),
                  ),
                  onPressed: () => _confirmAndTrigger(store),
                  icon: const Icon(Icons.sos),
                  label: const Text('Send SOS alert'),
                ),
              ),
            ] else ...[
              const Center(
                child: Padding(
                  padding: EdgeInsets.symmetric(vertical: 8),
                  child: Text('🚨 Alert active 🚨', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Reach ${store.name.isNotEmpty ? store.name : 'your contact'} whichever way '
                'works right now -- each just opens that app with the number or message '
                'ready; you still send or call yourself.',
                style: TextStyle(color: AppColors.inkSoft, height: 1.4, fontSize: 12.5),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _sendText(store),
                      icon: const Icon(Icons.sms_outlined, size: 18),
                      label: const Text('Text'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _sendWhatsApp(store),
                      icon: const Icon(Icons.chat_outlined, size: 18),
                      label: const Text('WhatsApp'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => launchCall(store.number),
                      icon: const Icon(Icons.call_outlined, size: 18),
                      label: const Text('Call'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: () async {
                    await _sos.stop();
                    if (mounted) setState(() => _active = false);
                  },
                  icon: const Icon(Icons.stop_circle_outlined),
                  label: const Text('Stop alert'),
                ),
              ),
            ],
            const SizedBox(height: 8),
            Center(
              child: TextButton(
                onPressed: () async {
                  if (_active) await _sos.stop();
                  if (context.mounted) Navigator.of(context).pop();
                },
                child: const Text('Close'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Fully unsets the emergency contact, distinct from "Change contact"
  /// (which keeps one configured, just a different number). Confirmed
  /// first since it's the one action here that leaves SOS unusable until
  /// a new contact is set again.
  Future<void> _confirmAndRemoveContact(EmergencyContactStore store) async {
    if (_active) await _sos.stop();
    if (!mounted) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remove emergency contact?'),
        content: Text(
          'SOS will have no contact to alert until you set a new one. '
          '${store.name.isNotEmpty ? store.name : store.number} will be removed.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(ctx).pop(false), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: AppColors.coral),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Remove'),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await store.clear();
      if (mounted) setState(() => _active = false);
    }
  }

  Future<void> _confirmAndTrigger(EmergencyContactStore store) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Send emergency alert?'),
        content: Text(
          'This sounds a repeating alert on your phone and opens a text message '
          'to ${store.number} with an emergency message ready to send. Once active, '
          'you can also reach them by WhatsApp or a phone call from this screen.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(ctx).pop(false), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: AppColors.coral),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Send alert'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    await _sos.start();
    if (mounted) setState(() => _active = true);
    await _sendText(store);
  }

  /// Builds the emergency message: the base alert text, the user's own
  /// name and medical information if they've filled in a profile (see
  /// ProfileDrawer), and a one-shot Google Maps link if a location fix is
  /// available -- see [currentLocationLinkOrNull]'s own doc for why a
  /// missing fix falls back to the plain message instead of blocking the
  /// send. Location is fetched fresh on every send rather than once when
  /// the alert starts: a location captured when the sheet opened could be
  /// meaningfully stale by the time a contact actually gets reached, and
  /// in a real emergency, stale is worse than a few extra seconds' wait
  /// for a current fix. Name/medical info don't need the same freshness
  /// concern -- they change rarely, so reading whatever's currently saved
  /// is enough.
  Future<String> _composeMessage() async {
    final profile = context.read<UserProfileStore>();
    var message = _alertMessage;
    if (profile.name.isNotEmpty) message = '$message\nFrom: ${profile.name}';
    if (profile.bloodGroup.isNotEmpty) message = '$message\nBlood group: ${profile.bloodGroup}';
    if (profile.allergies.isNotEmpty) message = '$message\nAllergies: ${profile.allergies}';
    if (profile.conditions.isNotEmpty) message = '$message\nMedical conditions: ${profile.conditions}';
    if (profile.medications.isNotEmpty) message = '$message\nMedications: ${profile.medications}';
    final link = await currentLocationLinkOrNull();
    return appendLocationLink(message, link);
  }

  Future<void> _sendText(EmergencyContactStore store) async {
    final message = await _composeMessage();
    await launchPrefilledSms(store.number, message);
  }

  Future<void> _sendWhatsApp(EmergencyContactStore store) async {
    final message = await _composeMessage();
    await launchPrefilledWhatsApp(store.number, message);
  }
}
