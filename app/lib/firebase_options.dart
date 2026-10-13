// PLACEHOLDER -- not a real Firebase project. The app runs fully without it (see AuthService.init,
// which never throws); only Google Sign-In is unavailable until this file is replaced.
//
// To enable real Google Sign-In:
//   1. Create a Firebase project at https://console.firebase.google.com (needs your own Google
//      account -- this is not something that can be automated on your behalf).
//   2. Add an Android app to it with package name com.hepquantlab.hep_quant_lab (see
//      android/app/build.gradle.kts `applicationId`).
//   3. Enable the Google sign-in provider under Authentication → Sign-in method.
//   4. Run `dart pub global activate flutterfire_cli` then `flutterfire configure` from this app/
//      directory and select the project you just created. It overwrites this exact file with real
//      values and also drops android/app/google-services.json -- no other code changes needed.
//
// This mirrors the shape `flutterfire configure` generates, so that replacement is a drop-in.
import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart' show defaultTargetPlatform, kIsWeb, TargetPlatform;

class DefaultFirebaseOptions {
  static const bool isPlaceholder = true;

  static FirebaseOptions get currentPlatform {
    if (kIsWeb) return web;
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for $defaultTargetPlatform.',
        );
    }
  }

  static const web = FirebaseOptions(
    apiKey: 'PLACEHOLDER-run-flutterfire-configure',
    appId: '1:000000000000:web:0000000000000000000000',
    messagingSenderId: '000000000000',
    projectId: 'hep-quant-lab-placeholder',
    authDomain: 'hep-quant-lab-placeholder.firebaseapp.com',
    storageBucket: 'hep-quant-lab-placeholder.appspot.com',
  );

  static const android = FirebaseOptions(
    apiKey: 'PLACEHOLDER-run-flutterfire-configure',
    appId: '1:000000000000:android:0000000000000000000000',
    messagingSenderId: '000000000000',
    projectId: 'hep-quant-lab-placeholder',
    storageBucket: 'hep-quant-lab-placeholder.appspot.com',
  );

  static const ios = FirebaseOptions(
    apiKey: 'PLACEHOLDER-run-flutterfire-configure',
    appId: '1:000000000000:ios:0000000000000000000000',
    messagingSenderId: '000000000000',
    projectId: 'hep-quant-lab-placeholder',
    storageBucket: 'hep-quant-lab-placeholder.appspot.com',
    iosBundleId: 'com.hepquantlab.hepQuantLab',
  );
}
