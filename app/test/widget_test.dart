// Basic smoke test: the app boots without Firebase configured (placeholder options) and shows
// either the home dashboard directly, or a sign-in screen with a working "skip" path -- either way
// it must not crash, since there's no real Firebase project behind the placeholder config yet.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:hep_quant_lab/main.dart';
import 'package:hep_quant_lab/services/auth_service.dart';

void main() {
  testWidgets('App boots to a Scaffold without crashing', (tester) async {
    await tester.pumpWidget(const HepQuantLabApp());
    await tester.pump();
    expect(find.byType(Scaffold), findsWidgets);
  });

  test('AuthService.isConfigured is false before Firebase.init runs', () {
    // Sanity check only (init() is called from main(), not from this unit test).
    expect(AuthService.isConfigured, isFalse);
  });
}
