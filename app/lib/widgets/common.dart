import 'package:flutter/material.dart';
import '../theme.dart';

/// Evidence-tier chip: MEASURED / SIMULATED / ESTIMATED / NOT AVAILABLE. Mirrors the website's
/// badges so a reader can tell, at a glance, what kind of evidence a number is.
class TierBadge extends StatelessWidget {
  final String tier;
  const TierBadge(this.tier, {super.key});

  static const _descriptions = {
    'MEASURED': 'PyTorch evaluation on held-out test split',
    'SIMULATED': 'hls4ml C-simulation (fixed-point arithmetic only)',
    'ESTIMATED': 'software hardware-resource heuristic; not validated by synthesis',
    'NOT AVAILABLE': 'no tool or device in the experimental environment',
  };

  @override
  Widget build(BuildContext context) {
    // Tinted-fill chip (tier color at low alpha, a brighter tint for the border, a near-white blend
    // for the text) rather than an outline/currentColor chip — matches the website's badge styling.
    final tint = AppColors.tierColors[tier] ?? AppColors.muted;
    final fg = Color.lerp(tint, Colors.white, 0.15)!;
    return Tooltip(
      message: _descriptions[tier] ?? tier,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
        decoration: BoxDecoration(
          color: tint.withValues(alpha: 0.14),
          border: Border.all(color: tint.withValues(alpha: 0.4)),
          borderRadius: BorderRadius.circular(999),
        ),
        child: Text(tier,
            style: TextStyle(
                color: fg, fontSize: 10.5, fontWeight: FontWeight.w500, letterSpacing: 0.5)),
      ),
    );
  }
}

class SectionHeader extends StatelessWidget {
  final String eyebrow;
  final String title;
  final String? subtitle;
  const SectionHeader({required this.eyebrow, required this.title, this.subtitle, super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 8, 4, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Sentence case, quiet tracking — numbered section eyebrows are earned (the content is a
          // genuine sequence), but the ALL-CAPS/heavy-tracking treatment itself was the generic tell.
          Text(eyebrow,
              style: const TextStyle(
                  color: AppColors.muted,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  letterSpacing: 0.3,
                  fontFamily: 'monospace')),
          const SizedBox(height: 4),
          Text(title,
              style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w800, height: 1.15)),
          if (subtitle != null) ...[
            const SizedBox(height: 6),
            Text(subtitle!, style: const TextStyle(color: AppColors.muted, fontSize: 13, height: 1.4)),
          ],
        ],
      ),
    );
  }
}

class MetricTile extends StatelessWidget {
  final String label;
  final String value;
  const MetricTile({required this.label, required this.value, super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: AppColors.panel2,
        border: Border.all(color: AppColors.line),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label.toUpperCase(),
              style: const TextStyle(
                  color: AppColors.muted,
                  fontSize: 10,
                  letterSpacing: 0.6,
                  fontFamily: 'monospace')),
          const SizedBox(height: 2),
          Text(value,
              style: const TextStyle(
                  fontSize: 17, fontFamily: 'monospace', fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class Callout extends StatelessWidget {
  final Widget child;
  final Color? borderColor;
  const Callout({required this.child, this.borderColor, super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.panel2,
        border: Border.all(color: borderColor ?? AppColors.line),
        borderRadius: BorderRadius.circular(10),
      ),
      child: DefaultTextStyle(
        style: const TextStyle(color: AppColors.text, fontSize: 13.5, height: 1.5),
        child: child,
      ),
    );
  }
}

class LoadingCard extends StatelessWidget {
  const LoadingCard({super.key});
  @override
  Widget build(BuildContext context) => const SizedBox(
        height: 220,
        child: Center(child: CircularProgressIndicator(color: AppColors.accent, strokeWidth: 2.4)),
      );
}

class ErrorCard extends StatelessWidget {
  final String message;
  final VoidCallback? onRetry;
  const ErrorCard({required this.message, this.onRetry, super.key});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.panel2,
          border: Border.all(color: AppColors.bad.withValues(alpha: 0.5)),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              const Icon(Icons.wifi_off_rounded, color: AppColors.bad, size: 18),
              const SizedBox(width: 8),
              const Expanded(
                  child: Text('Could not load live results',
                      style: TextStyle(fontWeight: FontWeight.w700))),
            ]),
            const SizedBox(height: 6),
            Text(message, style: const TextStyle(color: AppColors.muted, fontSize: 12)),
            if (onRetry != null) ...[
              const SizedBox(height: 10),
              OutlinedButton(onPressed: onRetry, child: const Text('Retry')),
            ],
          ],
        ),
      );
}

class Chip2 extends StatelessWidget {
  final String text;
  const Chip2(this.text, {super.key});
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 5),
        decoration: BoxDecoration(
          color: AppColors.panel2,
          border: Border.all(color: AppColors.line),
          borderRadius: BorderRadius.circular(999),
        ),
        child: Text(text,
            style: const TextStyle(
                color: AppColors.accent2, fontSize: 11.5, fontFamily: 'monospace')),
      );
}
