// Google Sign-In via Firebase Auth. This mainly demonstrates a real, working auth integration --
// there is currently no private content in the app to gate behind it (everything shown is the same
// public frozen research snapshot the website serves). See README.md "Firebase setup" for the one
// manual step needed before sign-in actually works: creating the Firebase project is something only
// the project owner's own Google account can do.
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:google_sign_in/google_sign_in.dart';

class AuthService {
  static bool _firebaseReady = false;
  static String? initError;

  /// Call once at app start. Never throws: if Firebase isn't configured yet (placeholder
  /// firebase_options.dart), the app still runs fully in "signed-out" mode -- sign-in is the only
  /// thing unavailable, and the UI says so plainly instead of crashing.
  static Future<void> init(FirebaseOptions options) async {
    try {
      await Firebase.initializeApp(options: options);
      _firebaseReady = true;
    } catch (e) {
      initError = e.toString();
      _firebaseReady = false;
    }
  }

  static bool get isConfigured => _firebaseReady;

  static Stream<User?> get authStateChanges =>
      _firebaseReady ? FirebaseAuth.instance.authStateChanges() : const Stream.empty();

  static User? get currentUser => _firebaseReady ? FirebaseAuth.instance.currentUser : null;

  static Future<User?> signInWithGoogle() async {
    if (!_firebaseReady) {
      throw StateError(
          'Firebase is not configured yet. See README.md "Firebase setup" to enable Google Sign-In.');
    }
    final googleSignIn = GoogleSignIn.instance;
    await googleSignIn.initialize();
    final account = await googleSignIn.authenticate();
    final googleAuth = account.authentication;
    final credential = GoogleAuthProvider.credential(idToken: googleAuth.idToken);
    final result = await FirebaseAuth.instance.signInWithCredential(credential);
    return result.user;
  }

  static Future<void> signOut() async {
    if (!_firebaseReady) return;
    await GoogleSignIn.instance.signOut();
    await FirebaseAuth.instance.signOut();
  }
}
