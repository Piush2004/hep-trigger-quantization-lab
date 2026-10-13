import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../services/data_service.dart';
import '../theme.dart';
import '../widgets/common.dart';

class PrecisionTab extends StatefulWidget {
  final DataService data;
  const PrecisionTab({required this.data, super.key});

  @override
  State<PrecisionTab> createState() => _PrecisionTabState();
}

class _PrecisionTabState extends State<PrecisionTab> {
  late Future<(List<ResultRow>, Map<String, List<Map<String, dynamic>>>)> _future;
  String _precision = 'INT6';
  String _method = 'PTQ';

  @override
  void initState() {
    super.initState();
    _future = _load();
  }

  Future<(List<ResultRow>, Map<String, List<Map<String, dynamic>>>)> _load() async {
    final results = await widget.data.results();
    final perSeed = await widget.data.quantizationPerSeed();
    return (results, perSeed);
  }

  void _retry() => setState(() => _future = _load());

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
          const SectionHeader(eyebrow: '02 · Precision explorer', title: 'Inspect one configuration'),
          FutureBuilder(
            future: _future,
            builder: (context, snap) {
              if (snap.connectionState == ConnectionState.waiting) return const LoadingCard();
              if (snap.hasError) return ErrorCard(message: '${snap.error}', onRetry: _retry);
              final (rows, perSeed) = snap.data!;
              final cuda = rows.where((r) => r.architecture.startsWith('A primary')).toList();
              final isFp = _precision == 'FP32';
              final key = isFp ? 'FP32' : '$_precision $_method';
              ResultRow? row;
              for (final r in cuda) {
                if (r.configuration == key) row = r;
              }

              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _segmented(['FP32', 'INT8', 'INT6', 'INT4'], _precision, (v) {
                    setState(() {
                      _precision = v;
                      if (v == 'FP32') _method = 'none';
                      if (v != 'FP32' && _method == 'none') _method = 'PTQ';
                    });
                  }),
                  const SizedBox(height: 10),
                  _segmented(['PTQ', 'QAT'], _method, isFp ? null : (v) => setState(() => _method = v)),
                  const SizedBox(height: 16),
                  if (row == null)
                    const Callout(child: Text('Not recorded for this combination.'))
                  else ...[
                    Wrap(spacing: 10, runSpacing: 10, children: [
                      MetricTile(label: 'ROC-AUC', value: '${fmtNum(row.rocAucMean)} ± ${fmtNum(row.rocAucSd)}'),
                      MetricTile(label: 'PR-AUC', value: fmtNum(row.prAuc)),
                      MetricTile(label: 'Macro F1', value: fmtNum(row.f1)),
                      MetricTile(label: 'Accuracy', value: fmtNum(row.accuracy)),
                      MetricTile(label: 'Storage', value: fmtKb(row.storageBytes)),
                      MetricTile(label: 'Δ vs FP32', value: fmtSigned(row.deltaVsFp32)),
                    ]),
                    const SizedBox(height: 14),
                    if (perSeed.containsKey(key))
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.fromLTRB(8, 16, 16, 8),
                          child: SizedBox(
                            height: 180,
                            child: _SeedBars(seedRows: perSeed[key]!),
                          ),
                        ),
                      ),
                    const SizedBox(height: 8),
                    const Text(
                      'CUDA-era frozen record. INT4 values are device-sensitive — see the Architecture tab for the CPU re-run.',
                      style: TextStyle(color: AppColors.muted, fontSize: 11.5),
                    ),
                  ],
                  const SizedBox(height: 26),
                  const SectionHeader(
                      eyebrow: '03 · PTQ vs QAT',
                      title: 'QAT recovers most of the INT4 loss'),
                  _PtqQatChart(cuda: cuda),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _segmented(List<String> options, String selected, void Function(String)? onChange) {
    return SegmentedButton<String>(
      segments: options.map((o) => ButtonSegment(value: o, label: Text(o))).toList(),
      selected: {selected == 'none' ? options.first : selected},
      onSelectionChanged: onChange == null ? null : (s) => onChange(s.first),
      showSelectedIcon: false,
    );
  }
}

class _SeedBars extends StatelessWidget {
  final List<Map<String, dynamic>> seedRows;
  const _SeedBars({required this.seedRows});

  @override
  Widget build(BuildContext context) {
    return BarChart(
      BarChartData(
        minY: 0.80,
        maxY: 0.95,
        gridData: FlGridData(show: true, getDrawingHorizontalLine: (v) => const FlLine(color: AppColors.line, strokeWidth: 1)),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 40, getTitlesWidget: (v, m) => Text(v.toStringAsFixed(2), style: const TextStyle(color: AppColors.muted, fontSize: 9)))),
          bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, getTitlesWidget: (v, m) {
            final i = v.toInt();
            if (i < 0 || i >= seedRows.length) return const SizedBox();
            return Text('${seedRows[i]['seed']}', style: const TextStyle(color: AppColors.muted, fontSize: 9));
          })),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        barGroups: [
          for (int i = 0; i < seedRows.length; i++)
            BarChartGroupData(x: i, barRods: [
              BarChartRodData(
                toY: (seedRows[i]['roc_auc'] as num).toDouble(),
                color: AppColors.accent2,
                width: 18,
                borderRadius: BorderRadius.circular(3),
              ),
            ]),
        ],
      ),
    );
  }
}

class _PtqQatChart extends StatelessWidget {
  final List<ResultRow> cuda;
  const _PtqQatChart({required this.cuda});

  ResultRow? _find(String config) {
    for (final r in cuda) {
      if (r.configuration == config) return r;
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final fp32 = _find('FP32');
    final ptq = _find('INT4 PTQ');
    final qat = _find('INT4 QAT');
    if (fp32 == null || ptq == null || qat == null) return const LoadingCard();
    final share = ((qat.rocAucMean! - ptq.rocAucMean!) / (fp32.rocAucMean! - ptq.rocAucMean!) * 100);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(8, 16, 16, 8),
            child: SizedBox(
              height: 200,
              child: BarChart(
                BarChartData(
                  minY: 0.85,
                  maxY: 0.95,
                  gridData: FlGridData(show: true, getDrawingHorizontalLine: (v) => const FlLine(color: AppColors.line, strokeWidth: 1)),
                  borderData: FlBorderData(show: false),
                  titlesData: FlTitlesData(
                    leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 40, getTitlesWidget: (v, m) => Text(v.toStringAsFixed(2), style: const TextStyle(color: AppColors.muted, fontSize: 9)))),
                    bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, getTitlesWidget: (v, m) {
                      const labels = ['FP32', 'INT4 PTQ', 'INT4 QAT'];
                      final i = v.toInt();
                      return Text(i >= 0 && i < labels.length ? labels[i] : '', style: const TextStyle(color: AppColors.muted, fontSize: 9));
                    })),
                    topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  ),
                  barGroups: [
                    BarChartGroupData(x: 0, barRods: [BarChartRodData(toY: fp32.rocAucMean!, color: AppColors.accent2, width: 34, borderRadius: BorderRadius.circular(4))]),
                    BarChartGroupData(x: 1, barRods: [BarChartRodData(toY: ptq.rocAucMean!, color: AppColors.bad, width: 34, borderRadius: BorderRadius.circular(4))]),
                    BarChartGroupData(x: 2, barRods: [BarChartRodData(toY: qat.rocAucMean!, color: AppColors.accent, width: 34, borderRadius: BorderRadius.circular(4))]),
                  ],
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 10),
        Callout(
          child: Text('CUDA-era: QAT recovered ≈${share.toStringAsFixed(0)}% of the INT4 PTQ loss '
              '(INT4 PTQ ${fmtNum(ptq.rocAucMean)} → INT4 QAT ${fmtNum(qat.rocAucMean)}, FP32 ${fmtNum(fp32.rocAucMean)}).'),
        ),
      ],
    );
  }
}
