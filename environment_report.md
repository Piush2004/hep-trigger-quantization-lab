# Environment Report

Generated at project start (Gate 0). See `reports/reproducibility_report.md` for the
per-experiment version of this information.

## Execution environment

The host OS is **Windows 11 Home**, but all project code runs inside **WSL2 Ubuntu
26.04**, not natively on Windows. This was a deliberate, user-approved decision made
after hitting a hard blocker (see "Why WSL2, not native Windows" below).

| Component | Value |
|---|---|
| Host OS | Windows 11 Home Single Language 10.0.26200 |
| Execution environment | WSL2 (Ubuntu 26.04.1 LTS), kernel 6.18.33.2-microsoft-standard-WSL2 |
| Python | 3.11.16 (CPython) |
| PyTorch | 2.11.0+cu128 |
| CUDA available | Yes |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM), CUDA 12.8 toolchain |
| CPU | AMD Ryzen 7 |
| RAM | 24 GB (≈23.3 GiB total visible to the OS) |
| Disk | 429 GB volume, 252 GB free at project start |
| Git | 2.54.0.windows.1 (used from both Windows and WSL against the same repo) |
| Docker | Installed (29.7.2) but NOT used — see below |
| Package manager | `uv` (0.11.19 on Windows, 0.12.21 in WSL) |

Project files live on the Windows filesystem at `D:\Projects\CERN`, which WSL2 mounts
at `/mnt/d/Projects/CERN`. The WSL virtual environment lives at `.venv-wsl/` inside the
project (gitignored); a native-Windows `.venv/` also exists from the initial setup
attempt but is not used for running code (gitignored, kept only as a build artifact of
the environment investigation below).

## Why WSL2, not native Windows or Docker

Three approaches were tried, in this order, before landing on WSL2:

1. **Native Windows Python 3.11 venv.** Core scientific packages (NumPy, PyTorch,
   pandas, scikit-learn) imported and ran correctly, including CUDA. However,
   **matplotlib's compiled `_path` extension was blocked by Windows Smart App Control**
   (a system code-integrity policy), with the exact OS-level log entry:

   > Code Integrity determined that a process ... attempted to load
   > `matplotlib\_path.cp311-win_amd64.pyd` that did not meet the Enterprise signing
   > level requirements or violated code integrity policy.

   This is a known Smart App Control limitation with reputation-scored, unsigned
   compiled Python extensions — notably, even an internal Chrome DLL was blocked by
   the same policy in the same time window, confirming this is SAC being broadly
   aggressive rather than a problem specific to this project. Disabling Smart App
   Control is a one-way action on Windows 11 (it can only be re-enabled via a clean
   reinstall), so this was treated as a decision for the user, not something to change
   automatically.

2. **Docker Desktop (Linux container).** The user chose this as the preferred fix,
   since it fully sidesteps Windows code-integrity policy for anything running inside
   the Linux container. An image built successfully once, but the Docker Desktop
   backend then crashed on a second start with an orphaned/unremovable Unix-domain
   socket file (`sailor-ingest.sock` and others under
   `AppData\Local\Docker\run\`), which persisted even after killing all Docker
   processes and running `wsl --shutdown`. This looked like it would require a full
   system reboot to clear a stale OS-level handle — a disruptive action warranting
   the user's own call, not something to do unprompted mid-session.

3. **WSL2 directly (chosen).** The machine already has a working WSL2 Ubuntu distro
   with GPU passthrough already configured (`nvidia-smi` works inside WSL out of the
   box). Running the project's Python environment directly in WSL2 avoids Windows
   Smart App Control entirely (it only applies to Windows PE binaries) without
   depending on Docker Desktop's separate, currently-broken backend. This is the
   environment used for everything in this project from Gate 0 onward.

## Installed packages (WSL `.venv-wsl`)

Core: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `h5py`,
`pytest`, `pytest-cov`.

PyTorch: installed separately from the CUDA 12.8 wheel index
(`https://download.pytorch.org/whl/cu128`) to get a build with kernels for the RTX
5060's Blackwell architecture (compute capability requires a recent CUDA toolkit;
older stable PyTorch/CUDA combinations do not include kernels for this GPU).

`hls4ml` / `qkeras` (optional, Gate 6): not installed by default — see
[docs/hardware.md](docs/hardware.md) for status and the fallback path when
unavailable.

## Known limitations

- The native-Windows `.venv/` and `Dockerfile`/`docker-compose.yml` are kept in the
  repo for documentation/portability (e.g. a Linux CI runner could use the Dockerfile
  directly, where Smart App Control is a non-issue), but are not the environment this
  project's results were produced in.
- Reproducing this project on another machine without the same Smart App Control
  issue can simply use a native Windows/Linux/macOS venv; WSL2 is specific to this
  machine's constraints, not a hard project requirement.
