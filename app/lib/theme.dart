// Dark scientific theme, matching demo/app/static/styles.css so the app and website feel like one
// product. Keep in sync if the website palette changes.
import 'package:flutter/material.dart';

class AppColors {
  static const bg = Color(0xFF0B1020);
  static const panel = Color(0xFF111A2E);
  static const panel2 = Color(0xFF0F1728);
  static const line = Color(0xFF22304D);
  static const text = Color(0xFFE6ECF7);
  static const muted = Color(0xFF93A3BF);
  static const accent = Color(0xFF5EEAD4); // teal
  static const accent2 = Color(0xFF7DD3FC); // cyan
  static const warn = Color(0xFFFBBF24);
  static const bad = Color(0xFFF87171);
  static const ok = Color(0xFF34D399);
  static const mixed = Color(0xFFC4B5FD);

  static const tierColors = {
    'MEASURED': ok,
    'SIMULATED': accent2,
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
            states.contains(WidgetState.selected) ? const Color(0xFF04211C) : AppColors.muted),
        side: WidgetStateProperty.all(const BorderSide(color: AppColors.line)),
      ),
    ),
    dividerColor: AppColors.line,
    splashFactory: InkSparkle.splashFactory,
  );
}
