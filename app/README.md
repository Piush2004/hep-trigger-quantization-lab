# HEP-Trigger Quantization Lab — Android app

A native Flutter/Android companion to the [web dashboard](../demo/README.md). Same frozen research
results, same live data source, a native UI. It does **not** train, run experiments, or compute
anything — it fetches the exact same static JSON snapshot the website serves, over HTTPS:

```
https://piush2004.github.io/hep-trigger-quantization-lab/data/<name>.json
```

(`lib/services/data_service.dart`). If that data changes (a new research run), both the website and
this app pick it up automatically on next load — nothing in the app needs to be rebuilt.

## What's in it

| Screen | Shows |
|---|---|
| Sign-in | Google Sign-In via Firebase Auth, with a "Continue without signing in" path — see **Authentication** below |
| Overview | Main accuracy-vs-storage scatter chart, headline stat, credibility chips, findings |
| Precision | Precision/method explorer (metric tiles + per-seed chart), PTQ-vs-QAT comparison |
| Architecture | Architecture A (CUDA-era + CPU re-run) and B (CPU) shown as **separate** cards — never merged, with the same observational-comparison caveat as the website |
| Hardware | The MEASURED → SIMULATED → NOT AVAILABLE evidence pipeline, hls4ml C-simulation chart, calibration-sensitivity chart |

Tested end-to-end on a real Android emulator (API 36, Google Play, Pixel 7 profile): all four tabs
render correct live data, matching the website's numbers exactly (e.g. INT6 PTQ ROC-AUC
0.9329 ± 0.0014 appears identically in both).

## Authentication

There is currently **no private content** in this app — every screen shows the same public frozen
research snapshot the website serves. Google Sign-In is wired in as a genuine, working integration
(not a mock), mainly to demonstrate the skill; it does not gate anything.

**The app runs fully without any Firebase project configured.** `lib/firebase_options.dart` ships
as an explicitly-labelled placeholder; `AuthService.init()` never throws even with fake keys, so the
dashboard works immediately. Tapping "Sign in with Google" before a real project exists will show
an inline error from Firebase's servers (invalid placeholder key) rather than crash the app.

**To enable real sign-in** (optional, ~5 minutes, needs your own Google account):
1. Create a project at <https://console.firebase.google.com> and add an Android app with package
   `com.hepquantlab.hep_quant_lab` (see `android/app/build.gradle.kts` → `applicationId`).
2. Authentication → Sign-in method → enable **Google**.
3. `dart pub global activate flutterfire_cli`, then from this `app/` directory: `flutterfire configure`
   and select that project. It overwrites `lib/firebase_options.dart` with real values and drops
   `android/app/google-services.json` — no other code change is needed.

## Run locally

Requires the Flutter SDK (stable channel) and an Android SDK with at least one platform + one
emulator system image, or a physical device with USB debugging enabled.

```bash
cd app
flutter pub get
flutter devices          # confirm a device/emulator is attached
flutter run               # debug build, hot reload
```

Build a release-style debug APK without launching an emulator session:
```bash
flutter build apk --debug
# output: build/app/outputs/flutter-apk/app-debug.apk
```

### A Windows-specific build note
If `flutter build apk` fails with `Could not close incremental caches` during `compileDebugKotlin`
(seen consistently across every Kotlin-using plugin on this machine — Windows Defender's real-time
scanning is the usual cause, locking the cache files between the compiler writing and closing them),
`android/gradle.properties` already sets `kotlin.incremental=false` to work around it.

## Test

```bash
cd app
flutter analyze    # static analysis; should report "No issues found!"
flutter test       # widget + unit tests
```

`test/widget_test.dart` checks the app boots to a `Scaffold` without crashing even with the
placeholder Firebase config (the realistic "not yet configured" state), and that
`AuthService.isConfigured` starts `false` before `Firebase.init()` runs.

## Design

Dark scientific theme (`lib/theme.dart`), matching `demo/app/static/styles.css` token-for-token, so
the website and app read as one product. Evidence-tier badges (MEASURED / SIMULATED / ESTIMATED /
NOT AVAILABLE) use the same colors and meanings as the website. The launcher icon is the same
"quantization staircase" mark as the website's favicon (`demo/scripts/generate_icons.py` generates
both; see that script if the palette ever changes).

## Scientific labelling (same rules as the website)

- The exploratory, test-set-selected mixed-precision result (ROC-AUC 0.9278) is excluded from the
  underlying data at build time (the static JSON export never contains it) — nothing in the app
  needs to filter for it separately.
- Storage is theoretical (analytical), never presented as measured.
- Architecture A's CUDA-era and CPU results are two separate cards, never averaged; the observational
  caveat ships with the API response and is rendered verbatim.
- HLS synthesis and physical FPGA are shown as `NOT AVAILABLE`, not omitted.

Independent research project. Not an official or CERN-affiliated project.
