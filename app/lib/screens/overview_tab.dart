import 'dart:math' as math;
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../services/data_service.dart';
import '../theme.dart';
import '../widgets/common.dart';

class OverviewTab extends StatefulWidget {
  final DataService data;
  const OverviewTab({required this.data, super.key});

  @override
  State<OverviewTab> createState() => _OverviewTabState();
}

class _OverviewTabState extends State<OverviewTab> {
  late Future<List<ResultRow>> _future;

  @override
  void initState() {
    super.initState();
    _future = widget.data.results();
  }

  void _retry() => setState(() => _future = widget.data.results());

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
            eyebrow: '01 · Main result',
            title: 'Accuracy vs theoretical weight storage',
            subtitle:
                'Test macro ROC-AUC, mean ± SD over five seeds, against analytical weight storage (bytes, log scale).',
          ),
          FutureBuilder<List<ResultRow>>(
            future: _future,
            builder: (context, snap) {
              if (snap.connectionState == ConnectionState.waiting) return const LoadingCard();
              if (snap.hasError) {
                return ErrorCard(message: '${snap.error}', onRetry: _retry);
              }
              final rows = snap.data!;
              final cuda = rows
                  .where((r) => r.architecture.startsWith('A primary'))
                  .toList();
              final wanted = ['FP32', 'INT8 PTQ', 'INT6 PTQ', 'INT4 PTQ', 'INT4 QAT'];
              final points = [
                ...cuda.where((r) => wanted.contains(r.configuration)),
                ...cuda.where((r) => r.configuration.contains('size-constrained')),
              ]..sort((a, b) => (a.storageBytes ?? 0).compareTo(b.storageBytes ?? 0));

              final fp32 = cuda.firstWhere((r) => r.configuration == 'FP32', orElse: () => rows.first);
              final int6 = cuda.firstWhere((r) => r.configuration == 'INT6 PTQ', orElse: () => rows.first);

              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(8, 18, 16, 8),
                      child: SizedBox(height: 280, child: _StorageChart(points: points)),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Callout(
                    borderColor: AppColors.accent.withValues(alpha: 0.4),
                    child: RichText(
                      text: TextSpan(
                        style: const TextStyle(color: AppColors.text, fontSize: 13.5, height: 1.5),
                        children: [
                          const TextSpan(
                              text: 'INT6 PTQ ', style: TextStyle(fontWeight: FontWeight.w800, color: AppColors.accent)),
                          TextSpan(
                              text:
                                  'reaches ROC-AUC ${fmtNum(int6.rocAucMean)} ± ${fmtNum(int6.rocAucSd)} (Δ FP32 ${fmtSigned(int6.deltaVsFp32)}) at ${fmtKb(int6.storageBytes)} theoretical weight storage — '
                                  '${((1 - (int6.storageBytes ?? 1) / (fp32.storageBytes ?? 1)) * 100).toStringAsFixed(1)}% below FP32 (${fmtKb(fp32.storageBytes)}).'),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 18),
                  _buildCredibilityChips(),
                  const SizedBox(height: 18),
                  _buildFindings(),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildCredibilityChips() {
    const chips = [
      '830K simulated jets', '16 features', '5 seeds', '2 MLP architectures', 'PTQ + QAT', 'hls4ml C-simulation'
    ];
    return Wrap(spacing: 8, runSpacing: 8, children: chips.map(Chip2.new).toList());
  }

  Widget _buildFindings() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('What the evidence supports',
                style: TextStyle(fontWeight: FontWeight.w800, color: AppColors.accent, fontSize: 14)),
            const SizedBox(height: 8),
            ..._bullets(const [
              'INT8 PTQ is practically lossless against FP32.',
              'INT6 PTQ costs about 0.001 ROC-AUC.',
              'INT4 PTQ degrades substantially in both architectures; INT4 QAT recovers most of the loss.',
              'No evaluated mixed-precision layout beat uniform INT6 PTQ under the 3,724 B constraint.',
            ]),
            const SizedBox(height: 14),
            const Text('What it does not support',
                style: TextStyle(fontWeight: FontWeight.w800, color: AppColors.bad, fontSize: 14)),
            const SizedBox(height: 8),
            ..._bullets(const [
              'Any claim about detector data, trigger efficiency, or physics operating points.',
              'Any FPGA resource, latency, or physical measurement.',
              'Device dependence as an established mechanism (observational only).',
              'Generalization beyond two small MLPs on one simulated dataset.',
            ]),
          ],
        ),
      ),
    );
  }

  List<Widget> _bullets(List<String> items) => items
      .map((t) => Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('•  ', style: TextStyle(color: AppColors.muted)),
                Expanded(child: Text(t, style: const TextStyle(fontSize: 13, height: 1.45))),
              ],
            ),
          ))
      .toList();
}

class _StorageChart extends StatelessWidget {
  final List<ResultRow> points;
  const _StorageChart({required this.points});

  @override
  Widget build(BuildContext context) {
    if (points.isEmpty) return const Center(child: Text('No data', style: TextStyle(color: AppColors.muted)));
    final logX = points.map((r) => math.log((r.storageBytes ?? 1).toDouble()) / math.ln10).toList();
    final y = points.map((r) => r.rocAucMean ?? 0).toList();
    final minX = logX.reduce(math.min) - 0.1;
    final maxX = logX.reduce(math.max) + 0.1;
    final minY = (y.reduce(math.min) - 0.02).clamp(0.0, 1.0);
    final maxY = (y.reduce(math.max) + 0.01).clamp(0.0, 1.0);

    return ScatterChart(
      ScatterChartData(
        minX: minX,
        maxX: maxX,
        minY: minY,
        maxY: maxY,
        scatterSpots: [
          for (int i = 0; i < points.length; i++)
            ScatterSpot(
              logX[i],
              y[i],
              dotPainter: FlDotCirclePainter(
                radius: points[i].configuration == 'INT6 PTQ' ? 9 : 6,
                color: points[i].configuration.contains('size-constrained')
                    ? AppColors.mixed
                    : points[i].configuration.contains('QAT')
                        ? AppColors.accent
                        : AppColors.accent2,
                strokeColor: points[i].configuration == 'INT6 PTQ' ? AppColors.text : Colors.transparent,
                strokeWidth: 2,
              ),
            ),
        ],
        scatterTouchData: ScatterTouchData(
          touchTooltipData: ScatterTouchTooltipData(
            getTooltipItems: (spot) {
              final i = logX.indexWhere((x) => (x - spot.x).abs() < 0.001 && (y[logX.indexOf(x)] - spot.y).abs() < 0.001);
              final idx = i >= 0 ? i : 0;
              final r = points[idx];
              return ScatterTooltipItem(
                '${r.configuration}\nROC-AUC ${fmtNum(r.rocAucMean)}\n${fmtKb(r.storageBytes)}',
                textStyle: const TextStyle(color: AppColors.text, fontSize: 11),
              );
            },
          ),
        ),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 48,
              getTitlesWidget: (v, meta) => Text(v.toStringAsFixed(3),
                  style: const TextStyle(color: AppColors.muted, fontSize: 9)),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 28,
              getTitlesWidget: (v, meta) => Text('10^${v.toStringAsFixed(1)}',
                  style: const TextStyle(color: AppColors.muted, fontSize: 9)),
            ),
          ),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        gridData: FlGridData(
          show: true,
          getDrawingHorizontalLine: (v) => const FlLine(color: AppColors.line, strokeWidth: 1),
          getDrawingVerticalLine: (v) => const FlLine(color: AppColors.line, strokeWidth: 1),
        ),
        borderData: FlBorderData(show: true, border: const Border.fromBorderSide(BorderSide(color: AppColors.line))),
      ),
    );
  }
}
