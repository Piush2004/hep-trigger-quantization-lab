import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../services/data_service.dart';
import '../theme.dart';
import '../widgets/common.dart';

const _configs = ['FP32', 'INT8 PTQ', 'INT6 PTQ', 'INT4 PTQ', 'INT4 QAT'];

class ArchitectureTab extends StatefulWidget {
  final DataService data;
  const ArchitectureTab({required this.data, super.key});

  @override
  State<ArchitectureTab> createState() => _ArchitectureTabState();
}

class _ArchitectureTabState extends State<ArchitectureTab> {
  late Future<(List<ArchPanel>, String)> _future;
  String _chosen = 'INT4 QAT';

  @override
  void initState() {
    super.initState();
    _future = _load();
  }

  Future<(List<ArchPanel>, String)> _load() async {
    final panels = await widget.data.architectureComparison();
    final caveat = await widget.data.architectureCaveat();
    return (panels, caveat);
  }

  void _retry() => setState(() => _future = _load());

  String _label(ResultRow r) =>
      r.configuration.isNotEmpty ? r.configuration : (r.raw['method'] == 'none' ? r.raw['precision'] : '${r.raw['precision']} ${r.raw['method']}');

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      color: AppColors.accent,
      onRefresh: () async {
        _retry();
        await _future;
      },
      child: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
        children: [
          const SectionHeader(
            eyebrow: '04 · Architecture generalization',
            title: 'Same pattern across two MLPs',
            subtitle: 'Architecture A (CUDA + CPU) and Architecture B (CPU), side by side. Devices are not merged.',
          ),
          FutureBuilder(
            future: _future,
            builder: (context, snap) {
              if (snap.connectionState == ConnectionState.waiting) return const LoadingCard();
              if (snap.hasError) return ErrorCard(message: '${snap.error}', onRetry: _retry);
              final (panels, caveat) = snap.data!;

              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  DropdownButtonFormField<String>(
                    initialValue: _chosen,
                    dropdownColor: AppColors.panel,
                    decoration: InputDecoration(
                      filled: true,
                      fillColor: AppColors.panel2,
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: const BorderSide(color: AppColors.line)),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    ),
                    items: _configs.map((c) => DropdownMenuItem(value: c, child: Text(c))).toList(),
                    onChanged: (v) => setState(() => _chosen = v!),
                  ),
                  const SizedBox(height: 16),
                  for (final p in panels) ...[
                    _ArchCard(panel: p, chosen: _chosen, labelOf: _label),
                    const SizedBox(height: 14),
                  ],
                  Callout(
                    borderColor: AppColors.line,
                    child: Text.rich(TextSpan(children: [
                      const TextSpan(text: 'Observational comparison.  ', style: TextStyle(fontWeight: FontWeight.w800)),
                      TextSpan(text: caveat),
                    ])),
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}

ResultRow? _findByLabel(List<ResultRow> rows, String label, String Function(ResultRow) labelOf) {
  for (final r in rows) {
    if (labelOf(r) == label) return r;
  }
  return null;
}

class _ArchCard extends StatelessWidget {
  final ArchPanel panel;
  final String chosen;
  final String Function(ResultRow) labelOf;
  const _ArchCard({required this.panel, required this.chosen, required this.labelOf});

  @override
  Widget build(BuildContext context) {
    final rows = [for (final c in _configs) _findByLabel(panel.rows, c, labelOf)].whereType<ResultRow>().toList();
    final chosenRow = _findByLabel(panel.rows, chosen, labelOf);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(panel.label, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 14)),
            const SizedBox(height: 2),
            Text('device: ${panel.device} · ${panel.parameters} parameters',
                style: const TextStyle(color: AppColors.muted, fontSize: 11, fontFamily: 'monospace')),
            const SizedBox(height: 10),
            SizedBox(
              height: 160,
              child: BarChart(
                BarChartData(
                  minY: 0.85,
                  maxY: 0.95,
                  gridData: FlGridData(show: true, getDrawingHorizontalLine: (v) => const FlLine(color: AppColors.line, strokeWidth: 1)),
                  borderData: FlBorderData(show: false),
                  titlesData: FlTitlesData(
                    leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 36, getTitlesWidget: (v, m) => Text(v.toStringAsFixed(2), style: const TextStyle(color: AppColors.muted, fontSize: 8)))),
                    bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, getTitlesWidget: (v, m) {
                      final i = v.toInt();
                      if (i < 0 || i >= rows.length) return const SizedBox();
                      return Padding(padding: const EdgeInsets.only(top: 4), child: Text(labelOf(rows[i]).replaceAll(' ', '\n'), textAlign: TextAlign.center, style: const TextStyle(color: AppColors.muted, fontSize: 7.5)));
                    })),
                    topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  ),
                  barGroups: [
                    for (int i = 0; i < rows.length; i++)
                      BarChartGroupData(x: i, barRods: [
                        BarChartRodData(
                          toY: rows[i].rocAucMean ?? 0,
                          width: 22,
                          borderRadius: BorderRadius.circular(3),
                          color: labelOf(rows[i]) == chosen ? AppColors.accent : AppColors.lineStrong,
                        ),
                      ]),
                  ],
                ),
              ),
            ),
            if (chosenRow != null) ...[
              const SizedBox(height: 10),
              Wrap(spacing: 8, runSpacing: 8, children: [
                MetricTile(label: 'ROC-AUC', value: '${fmtNum(chosenRow.rocAucMean)} ± ${fmtNum(chosenRow.rocAucSd)}'),
                MetricTile(label: 'Δ vs own FP32', value: fmtSigned(chosenRow.deltaVsOwnFp32)),
                MetricTile(label: 'Storage', value: fmtKb(chosenRow.storageBytes)),
              ]),
            ],
          ],
        ),
      ),
    );
  }
}
