import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../app_theme.dart';
import '../services/emergency_contact_store.dart';
import '../services/user_profile_store.dart';
import 'sos_sheet.dart';

/// The user's own profile: name, medical information, and an optional
/// photo -- all optional, all local (see UserProfileStore's own doc for
/// why). Opened as a Drawer from HomeScreen, reachable either by swiping
/// in from the left edge or tapping the menu icon Flutter automatically
/// adds to the AppBar once a Scaffold has a drawer -- see
/// HomeScreenState's own comment on why the swipe stays enabled despite
/// sharing its screen edge with Android's own back gesture.
///
/// The emergency contact shortcut at the bottom is a second entry point
/// to the same setup screen SosSheet already offers, not a separate
/// feature -- someone filling in their medical information is already in
/// exactly the right mindset to also set up who SOS should reach, without
/// needing to separately discover the SOS screen's own edit option.
class ProfileDrawer extends StatefulWidget {
  const ProfileDrawer({super.key});

  @override
  State<ProfileDrawer> createState() => _ProfileDrawerState();
}

const _bloodGroups = ['', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'];

class _ProfileDrawerState extends State<ProfileDrawer> {
  final _nameController = TextEditingController();
  final _allergiesController = TextEditingController();
  final _conditionsController = TextEditingController();
  final _medicationsController = TextEditingController();
  String _bloodGroup = '';
  // Seeded once from the store, not on every rebuild -- otherwise typing
  // would fight with the store's own (unchanged-until-Save) value on
  // every keystroke's rebuild.
  bool _loadedFromStore = false;

  @override
  void dispose() {
    _nameController.dispose();
    _allergiesController.dispose();
    _conditionsController.dispose();
    _medicationsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final profile = context.watch<UserProfileStore>();
    final contactStore = context.watch<EmergencyContactStore>();

    if (!_loadedFromStore && profile.isReady) {
      _nameController.text = profile.name;
      _allergiesController.text = profile.allergies;
      _conditionsController.text = profile.conditions;
      _medicationsController.text = profile.medications;
      _bloodGroup = profile.bloodGroup;
      _loadedFromStore = true;
    }

    return Drawer(
      child: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text('Your profile', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 4),
            const Text(
              'Optional, and kept on this device only -- never uploaded, no '
              'account needed. It makes an SOS alert more useful to whoever '
              'receives it, and lets someone helping you in person see who you '
              'are at a glance.',
              style: TextStyle(color: AppColors.inkSoft, fontSize: 12.5, height: 1.4),
            ),
            const SizedBox(height: 20),
            Center(child: _PhotoPicker(profile: profile)),
            const SizedBox(height: 20),
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
            ),
            const SizedBox(height: 16),
            const Padding(
              padding: EdgeInsets.only(bottom: 8),
              child: Text(
                'MEDICAL INFORMATION',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8,
                  color: AppColors.primary,
                ),
              ),
            ),
            DropdownButtonFormField<String>(
              initialValue: _bloodGroups.contains(_bloodGroup) ? _bloodGroup : '',
              decoration: const InputDecoration(labelText: 'Blood group', border: OutlineInputBorder()),
              items: [
                for (final group in _bloodGroups)
                  DropdownMenuItem(value: group, child: Text(group.isEmpty ? 'Not set' : group)),
              ],
              onChanged: (value) => setState(() => _bloodGroup = value ?? ''),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _allergiesController,
              decoration: const InputDecoration(
                labelText: 'Allergies',
                hintText: 'e.g. Penicillin, peanuts',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _conditionsController,
              decoration: const InputDecoration(
                labelText: 'Medical conditions',
                hintText: 'e.g. Epilepsy, type 1 diabetes',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _medicationsController,
              decoration: const InputDecoration(
                labelText: 'Current medications',
                hintText: 'e.g. Insulin',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: () async {
                  await profile.setProfile(
                    name: _nameController.text,
                    bloodGroup: _bloodGroup,
                    allergies: _allergiesController.text,
                    conditions: _conditionsController.text,
                    medications: _medicationsController.text,
                  );
                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Profile saved')),
                    );
                  }
                },
                child: const Text('Save'),
              ),
            ),
            const SizedBox(height: 24),
            const Divider(),
            const SizedBox(height: 12),
            const Padding(
              padding: EdgeInsets.only(bottom: 8),
              child: Text(
                'EMERGENCY CONTACT',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8,
                  color: AppColors.primary,
                ),
              ),
            ),
            ListTile(
              contentPadding: EdgeInsets.zero,
              leading: const Icon(Icons.emergency, color: AppColors.coral),
              title: Text(
                contactStore.isConfigured
                    ? (contactStore.name.isNotEmpty
                        ? '${contactStore.name} (${contactStore.number})'
                        : contactStore.number)
                    : 'Not set',
              ),
              subtitle: Text(
                contactStore.isConfigured ? 'Who SOS alerts reach' : 'Set a number for SOS to reach',
              ),
              trailing: const Icon(Icons.chevron_right),
              onTap: () {
                Navigator.of(context).pop(); // close the drawer first
                SosSheet.show(context, startInEditing: true);
              },
            ),
          ],
        ),
      ),
    );
  }
}

class _PhotoPicker extends StatelessWidget {
  const _PhotoPicker({required this.profile});
  final UserProfileStore profile;

  Future<void> _choose(BuildContext context) async {
    final action = await showModalBottomSheet<String>(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.photo_camera_outlined),
              title: const Text('Take a photo'),
              onTap: () => Navigator.of(ctx).pop('camera'),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library_outlined),
              title: const Text('Choose from gallery'),
              onTap: () => Navigator.of(ctx).pop('gallery'),
            ),
            if (profile.hasPhoto)
              ListTile(
                leading: const Icon(Icons.delete_outline, color: AppColors.coral),
                title: const Text('Remove photo', style: TextStyle(color: AppColors.coral)),
                onTap: () => Navigator.of(ctx).pop('remove'),
              ),
          ],
        ),
      ),
    );
    if (action == null) return;
    if (action == 'remove') {
      await profile.clearPhoto();
      return;
    }
    try {
      final picker = ImagePicker();
      final picked = await picker.pickImage(
        source: action == 'camera' ? ImageSource.camera : ImageSource.gallery,
        maxWidth: 800,
        imageQuality: 85,
      );
      if (picked != null) await profile.setPhoto(File(picked.path));
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not set photo: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => _choose(context),
      child: Stack(
        alignment: Alignment.bottomRight,
        children: [
          CircleAvatar(
            radius: 44,
            backgroundColor: AppColors.primary.withValues(alpha: 0.12),
            backgroundImage: profile.hasPhoto ? FileImage(File(profile.photoPath!)) : null,
            child: profile.hasPhoto
                ? null
                : const Icon(Icons.person_outline, size: 40, color: AppColors.primary),
          ),
          Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: AppColors.primary,
              shape: BoxShape.circle,
              border: Border.all(color: Theme.of(context).scaffoldBackgroundColor, width: 2),
            ),
            child: const Icon(Icons.edit, size: 14, color: Colors.white),
          ),
        ],
      ),
    );
  }
}
