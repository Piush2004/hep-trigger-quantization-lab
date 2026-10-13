import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../services/data_service.dart';
import '../theme.dart';
import '../widgets/common.dart';

class HardwareTab extends StatefulWidget {
  final DataService data;
  const HardwareTab({required this.data, super.key});

  @override
  State<HardwareTab> createState() => _HardwareTabState();
}

class _HardwareTabState extends State<HardwareTab> {
  late Future<(List<Map<String, dynamic>>, List<Map<String, dynamic>>)> _future;

  @override
  void initState() {
    super.initState();
    _future = _load();
  }

  Future<(List<Map<String, dynamic>>, List<Map<String, dynamic>>)> _load() async {
    final hls = await widget.data.hls4ml();
    final calib = await widget.data.calibration();
    return (hls, calib);
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
          const SectionHeader(eyebrow: '05 · Hardware evidence', title: 'From PyTorch to an FPGA, with each stage labelled'),
          FutureBuilder(
            future: _future,
            builder: (context, snap) {
              if (snap.connectionState == ConnectionState.waiting) return const LoadingCard();
              if (snap.hasError) return ErrorCard(message: '${snap.error}', onRetry: _retry);
              final (hls, calib) = snap.data!;
              final fp32 = hls.firstWhere((r) => r['stage'] == 'PyTorch FP32', orElse: () => const {});
              final w16 = hls.firstWhere(
                  (r) => r['stage'] == 'hls4ml C-simulation' && r['width_W'] == 16 && (r['mode'] as String? ?? '').startsWith('default'),
                  orElse: () => const {});

              final steps = [
                ('PyTorch FP32', 'MEASURED', 'accuracy ${fmtNum((fp32['accuracy'] as num?)?.toDouble())}, ROC-AUC ${fmtNum((fp32['roc_auc'] as num?)?.toDouble())}'),
                ('PyTorch quantized model (INT8/INT6/INT4, PTQ & QAT)', 'MEASURED', 'fake-quantized in float32; see the Precision tab for values'),
                ('hls4ml conversion, fixed-point ap_fixed<W,6>', 'SIMULATED',
                    w16.isEmpty ? 'C-simulation rows below' : 'at W=16: accuracy ${fmtNum((w16['accuracy'] as num?)?.toDouble())}, agreement ${fmtNum((w16['agreement_with_fp32'] as num?)?.toDouble())}'),
                ('HLS synthesis (resources, latency, II)', 'NOT AVAILABLE', 'HLS SYNTHESIS NOT AVAILABLE IN EXPERIMENTAL ENVIRONMENT'),
                ('Physical FPGA measurement', 'NOT AVAILABLE', 'no FPGA device or measurement exists'),
              ];

              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  for (final s in steps) ...[
                    _PipelineStep(title: s.$1, tier: s.$2, note: s.$3),
                    const SizedBox(height: 8),
                  ],
                  const SizedBox(height: 10),
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(14),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('hls4ml C-simulation accuracy vs width', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 13)),
                          const SizedBox(height: 10),
                          SizedBox(height: 200, child: _HlsChart(rows: hls, fp32Accuracy: (fp32['accuracy'] as num?)?.toDouble())),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 22),
                  const SectionHeader(eyebrow: '06 · Calibration sensitivity', title: 'INT4 PTQ depends on calibration size'),
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(14),
                      child: SizedBox(height: 200, child: _CalibChart(rows: calib)),
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text('Architecture A, CPU, min/max calibration. Model seed is the larger source of variation; clipping is near zero.',
                      style: TextStyle(color: AppColors.muted, fontSize: 11.5)),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}

class _PipelineStep extends StatelessWidget {
  final String title, tier, note;
  const _PipelineStep({required this.title, required this.tier, required this.note});

  @override
  Widget build(BuildContext context) {
    final na = tier == 'NOT AVAILABLE';
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.panel2,
        border: Border.all(color: AppColors.line, style: na ? BorderStyle.solid : BorderStyle.solid),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12.5)),
                const SizedBox(height: 2),
                Text(note, style: const TextStyle(color: AppColors.muted, fontSize: 10.5)),
              ],
            ),
          ),
          const SizedBox(width: 8),
          TierBadge(tier),
        ],
      ),
    );
  }
}

class _HlsChart extends StatelessWidget {
  final List<Map<String, dynamic>> rows;
  final double? fp32Accuracy;
  const _HlsChart({required this.rows, this.fp32Accuracy});

  @override
  Widget build(BuildContext context) {
    final simRows = rows.where((r) => r['stage'] == 'hls4ml C-simulation').toList();
    final modes = <String, List<Map<String, dynamic>>>{};
    for (final r in simRows) {
      (modes[r['mode'] as String] ??= []).add(r);
    }
    final palette = [AppColors.warn, AppColors.accent2, AppColors.accent, AppColors.mixed];
    int ci = 0;

    return LineChart(
      LineChartData(
        minY: 0,
        maxY: 0.85,
        gridData: FlGridData(show: true, getDrawingHorizontalLine: (v) => const FlLine(color: AppColors.line, strokeWidth: 1)),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 32, getTitlesWidget: (v, m) => Text(v.toStringAsFixed(1), style: const TextStyle(color: AppColors.muted, fontSize: 8)))),
          bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, getTitlesWidget: (v, m) => Text('${v.toInt()}', style: const TextStyle(color: AppColors.muted, fontSize: 8)))),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        lineBarsData: [
          for (final entry in modes.entries)
            LineChartBarData(
              spots: (entry.value..sort((a, b) => (a['width_W'] as int).compareTo(b['width_W'] as int)))
                  .map((r) => FlSpot((r['width_W'] as int).toDouble(), (r['accuracy'] as num).toDouble()))
                  .toList(),
              color: palette[ci++ % palette.length],
              barWidth: 2.2,
              dotData: const FlDotData(show: true),
              isCurved: false,
            ),
          if (fp32Accuracy != null)
            LineChartBarData(
              spots: [FlSpot(8, fp32Accuracy!), FlSpot(16, fp32Accuracy!)],
              color: AppColors.muted,
              barWidth: 1.4,
              dashArray: [6, 4],
              dotData: const FlDotData(show: false),
            ),
        ],
      ),
    );
  }
}

class _CalibChart extends StatelessWidget {
  final List<Map<String, dynamic>> rows;
  const _CalibChart({required this.rows});

  @override
  Widget build(BuildContext context) {
    final series = ['INT8', 'INT6', 'INT4'];
    final colors = [AppColors.accent, AppColors.accent2, AppColors.warn];

    return LineChart(
      LineChartData(
        minY: 0.85,
        maxY: 0.94,
        gridData: FlGridData(show: true, getDrawingHorizontalLine: (v) => const FlLine(color: AppColors.line, strokeWidth: 1)),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 36, getTitlesWidget: (v, m) => Text(v.toStringAsFixed(2), style: const TextStyle(color: AppColors.muted, fontSize: 8)))),
          bottomTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        lineBarsData: [
          for (int i = 0; i < series.length; i++)
            LineChartBarData(
              spots: (rows.where((r) => r['precision'] == series[i]).toList()
                    ..sort((a, b) => (a['calib_samples'] as int).compareTo(b['calib_samples'] as int)))
                  .asMap()
                  .entries
                  .map((e) => FlSpot(e.key.toDouble(), (e.value['roc_auc_mean'] as num).toDouble()))
                  .toList(),
              color: colors[i],
              barWidth: 2.2,
              dotData: const FlDotData(show: true),
            ),
        ],
      ),
    );
  }
}
