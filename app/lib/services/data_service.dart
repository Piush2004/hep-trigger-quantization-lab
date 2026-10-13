// Fetches the SAME frozen-research snapshot the website uses -- the static GitHub Pages mirror built
// by demo/scripts/build_static_site.py from the live FastAPI app's own route functions. No training,
// no experiments, nothing computed on-device: this is strictly a result explorer, like the website.
import 'dart:convert';
import 'package:http/http.dart' as http;

const kDataBaseUrl = 'https://piush2004.github.io/hep-trigger-quantization-lab/data';
const kRepoUrl = 'https://github.com/Piush2004/hep-trigger-quantization-lab';
const kResearchCommit = '970f44a';
const kExcludedRocAuc = 0.9278; // the biased, test-set-selected exploratory value; never shown as a result

class ResultRow {
  final Map<String, dynamic> raw;
  ResultRow(this.raw);

  String get architecture => raw['architecture'] as String? ?? '';
  String get configuration => raw['configuration'] as String? ?? '';
  String get device => raw['device'] as String? ?? '';
  String get tier => raw['tier'] as String? ?? 'MEASURED';
  int get n => raw['n'] as int? ?? 0;
  double? get rocAucMean => (raw['roc_auc_mean'] as num?)?.toDouble();
  double? get rocAucSd => (raw['roc_auc_sd'] as num?)?.toDouble();
  double? get prAuc => (raw['pr_auc_mean'] as num?)?.toDouble();
  double? get f1 => (raw['f1_mean'] as num?)?.toDouble();
  double? get accuracy => (raw['accuracy_mean'] as num?)?.toDouble();
  double? get deltaVsFp32 => (raw['delta_vs_fp32'] as num?)?.toDouble();
  double? get deltaVsOwnFp32 => (raw['delta_vs_own_fp32_mean'] as num?)?.toDouble() ?? deltaVsFp32;
  int? get storageBytes => (raw['storage_bytes'] as num?)?.toInt();

  bool get isPublic {
    final roc = rocAucMean;
    if (roc != null && (roc - kExcludedRocAuc).abs() < 1e-9 && configuration.contains('Mixed')) {
      return false; // defense-in-depth; the static JSON already excludes this at build time
    }
    return true;
  }
}

class ArchPanel {
  final Map<String, dynamic> raw;
  ArchPanel(this.raw);
  String get label => raw['label'] as String? ?? '';
  String get device => raw['device'] as String? ?? '';
  int get parameters => raw['parameters'] as int? ?? 0;
  List<ResultRow> get rows =>
      (raw['rows'] as List).map((r) => ResultRow(r as Map<String, dynamic>)).toList();
}

class DataService {
  final http.Client _client;
  DataService({http.Client? client}) : _client = client ?? http.Client();

  Future<Map<String, dynamic>> _getJson(String name) async {
    final res = await _client
        .get(Uri.parse('$kDataBaseUrl/$name.json'))
        .timeout(const Duration(seconds: 15));
    if (res.statusCode != 200) {
      throw Exception('$name → HTTP ${res.statusCode}');
    }
    return json.decode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> metadata() => _getJson('metadata');

  Future<List<ResultRow>> results() async {
    final body = await _getJson('results');
    return (body['rows'] as List)
        .map((r) => ResultRow(r as Map<String, dynamic>))
        .where((r) => r.isPublic)
        .toList();
  }

  Future<Map<String, List<Map<String, dynamic>>>> quantizationPerSeed() async {
    final body = await _getJson('quantization');
    final perSeed = body['per_seed'] as Map<String, dynamic>;
    return perSeed.map((k, v) => MapEntry(k, (v as List).cast<Map<String, dynamic>>()));
  }

  Future<List<Map<String, dynamic>>> pareto() async {
    final body = await _getJson('pareto');
    return (body['points'] as List).cast<Map<String, dynamic>>();
  }

  Future<List<ArchPanel>> architectureComparison() async {
    final body = await _getJson('architecture-comparison');
    return (body['panels'] as List).map((p) => ArchPanel(p as Map<String, dynamic>)).toList();
  }

  Future<String> architectureCaveat() async {
    final body = await _getJson('architecture-comparison');
    return body['caveat'] as String? ?? '';
  }

  Future<List<Map<String, dynamic>>> calibration() async {
    final body = await _getJson('calibration-sensitivity');
    return (body['rows'] as List).cast<Map<String, dynamic>>();
  }

  Future<List<Map<String, dynamic>>> hls4ml() async {
    final body = await _getJson('hls4ml');
    return (body['rows'] as List).cast<Map<String, dynamic>>();
  }

  void dispose() => _client.close();
}

String fmtNum(double? x, {int digits = 4}) => x == null ? 'not recorded' : x.toStringAsFixed(digits);
String fmtSigned(double? x, {int digits = 4}) =>
    x == null ? '—' : '${x >= 0 ? '+' : ''}${x.toStringAsFixed(digits)}';
String fmtKb(int? bytes) {
  if (bytes == null) return 'not recorded';
  final v = bytes / 1000;
  return '${v < 10 ? v.toStringAsFixed(2) : v.toStringAsFixed(1)} KB';
}
