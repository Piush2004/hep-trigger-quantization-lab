import 'package:flutter/material.dart';
import '../services/auth_service.dart';
import '../theme.dart';
import '../widgets/common.dart';

class SignInScreen extends StatefulWidget {
  final VoidCallback onSkip;
  const SignInScreen({required this.onSkip, super.key});

  @override
  State<SignInScreen> createState() => _SignInScreenState();
}

class _SignInScreenState extends State<SignInScreen> {
  bool _busy = false;
  String? _error;

  Future<void> _signIn() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await AuthService.signInWithGoogle();
      // On success, the StreamBuilder in main.dart picks up the new auth state automatically.
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: AppColors.panel,
                      border: Border.all(color: AppColors.line),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Icon(Icons.insights_rounded, color: AppColors.accent, size: 36),
                  ),
                ],
              ),
              const SizedBox(height: 22),
              const Text('HEP-Trigger\nQuantization Lab',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, height: 1.2)),
              const SizedBox(height: 10),
              const Text(
                'A reproducible research benchmark on neural-network quantization for simulated HEP jet classification.',
                textAlign: TextAlign.center,
                style: TextStyle(color: AppColors.muted, fontSize: 13.5, height: 1.5),
              ),
              const SizedBox(height: 32),
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 14),
                  child: ErrorCard(message: _error!),
                ),
              FilledButton.icon(
                onPressed: _busy || !AuthService.isConfigured ? null : _signIn,
                style: FilledButton.styleFrom(
                  backgroundColor: AppColors.accent,
                  foregroundColor: const Color(0xFF04211C),
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                icon: _busy
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF04211C)))
                    : const Icon(Icons.login_rounded),
                label: Text(_busy ? 'Signing in…' : 'Sign in with Google'),
              ),
              const SizedBox(height: 12),
              OutlinedButton(
                onPressed: widget.onSkip,
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  side: const BorderSide(color: AppColors.line),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Continue without signing in'),
              ),
              const SizedBox(height: 18),
              Callout(
                child: Text(
                  AuthService.isConfigured
                      ? 'All results shown in this app are public. Signing in demonstrates a working Firebase/Google authentication flow; it does not unlock any extra content.'
                      : 'Google Sign-In is not configured in this build yet (placeholder Firebase project). All results are still fully available — see README.md "Firebase setup".',
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
