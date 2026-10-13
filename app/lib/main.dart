import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'firebase_options.dart';
import 'screens/home_shell.dart';
import 'screens/sign_in_screen.dart';
import 'services/auth_service.dart';
import 'theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await AuthService.init(DefaultFirebaseOptions.currentPlatform);
  runApp(const HepQuantLabApp());
}

class HepQuantLabApp extends StatelessWidget {
  const HepQuantLabApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HEP-Trigger Quantization Lab',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: const _AuthGate(),
    );
  }
}

/// Google Sign-In is optional, not a lock on the content: every screen here shows the same public
/// frozen research results regardless of sign-in state. "Continue without signing in" always works.
class _AuthGate extends StatefulWidget {
  const _AuthGate();
  @override
  State<_AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<_AuthGate> {
  bool _skippedSignIn = false;

  @override
  Widget build(BuildContext context) {
    if (!AuthService.isConfigured) {
      // Firebase project not set up yet (placeholder config) -- go straight to the app.
      return const HomeShell(user: null);
    }
    return StreamBuilder<User?>(
      stream: AuthService.authStateChanges,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            backgroundColor: AppColors.bg,
            body: Center(child: CircularProgressIndicator(color: AppColors.accent)),
          );
        }
        final user = snapshot.data;
        if (user != null || _skippedSignIn) {
          return HomeShell(user: user);
        }
        return SignInScreen(onSkip: () => setState(() => _skippedSignIn = true));
      },
    );
  }
}
