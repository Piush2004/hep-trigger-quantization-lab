// Dark scientific theme, matching demo/app/static/styles.css so the app and website feel like one
// product. Keep in sync if the website palette changes.
import 'package:flutter/material.dart';

class AppColors {
  static const bg = Color(0xFF0A0D17);
  static const panel = Color(0xFF131826); // --surface
  static const panel2 = Color(0xFF0E1220); // --surface-sunken
  static const line = Color(0xFF262E42); // --border
  static const lineStrong = Color(0xFF3A4560); // --border-strong
  static const text = Color(0xFFEDF1F9);
  static const muted = Color(0xFF93A1BD);
  static const accent = Color(0xFF4FA6FF); // signal blue — the one action color
  static const accentInk = Color(0xFF071022); // text set on top of accent fills
  static const accent2 = Color(0xFFB9A6F7); // data-violet — secondary chart series only, never a control
  static const warn = Color(0xFFD9A441); // ESTIMATED: desaturated brass, not SaaS warning-yellow
  static const bad = Color(0xFFE2645C); // NOT AVAILABLE / error
  static const ok = Color(0xFF3BC48D); // MEASURED
  static const mixed = Color(0xFFB9A6F7); // same as accent2 — SIMULATED tier shares the data-violet hue

  static const tierColors = {
    'MEASURED': ok,
    'SIMULATED': mixed, // off cyan, onto violet — see reports/design_brief.md
    'ESTIMATED': warn,
    'NOT AVAILABLE': bad,
  };
}

ThemeData buildAppTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: AppColors.bg,
    colorScheme: base.colorScheme.copyWith(
      primary: AppColors.accent,
      secondary: AppColors.accent2,
      surface: AppColors.panel,
      error: AppColors.bad,
    ),
    textTheme: base.textTheme.apply(bodyColor: AppColors.text, displayColor: AppColors.text),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.bg,
      foregroundColor: AppColors.text,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: AppColors.panel,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: const BorderSide(color: AppColors.line),
      ),
      margin: EdgeInsets.zero,
    ),
    navigationBarTheme: NavigationBarThemeData(
      backgroundColor: AppColors.panel,
      indicatorColor: AppColors.accent.withValues(alpha: 0.18),
      labelTextStyle: WidgetStateProperty.resolveWith((states) => TextStyle(
            fontSize: 11,
            fontWeight: states.contains(WidgetState.selected) ? FontWeight.w700 : FontWeight.w500,
            color: states.contains(WidgetState.selected) ? AppColors.accent : AppColors.muted,
          )),
      iconTheme: WidgetStateProperty.resolveWith((states) => IconThemeData(
            color: states.contains(WidgetState.selected) ? AppColors.accent : AppColors.muted,
          )),
    ),
    segmentedButtonTheme: SegmentedButtonThemeData(
      style: ButtonStyle(
        backgroundColor: WidgetStateProperty.resolveWith((states) =>
            states.contains(WidgetState.selected) ? AppColors.accent : AppColors.panel2),
        foregroundColor: WidgetStateProperty.resolveWith((states) =>
            states.contains(WidgetState.selected) ? AppColors.accentInk : AppColors.muted),
        side: WidgetStateProperty.all(const BorderSide(color: AppColors.line)),
      ),
    ),
    dividerColor: AppColors.line,
    splashFactory: InkSparkle.splashFactory,
  );
}
