# Troubleshooting

## Environment setup (Windows)

### "An Application Control policy has blocked this file" / matplotlib ImportError

This is Windows **Smart App Control** (SAC), a system code-integrity policy, blocking
a compiled Python extension (e.g. `matplotlib/_path.cp311-win_amd64.pyd`) because that
specific file hasn't accumulated reputation trust in Microsoft's cloud model yet. It
is not specific to this project — in the same environment, even an internal Chrome DLL
was blocked by the same policy. Confirm the root cause via:

```powershell
Get-WinEvent -LogName "Microsoft-Windows-CodeIntegrity/Operational" -MaxEvents 10 |
    Select-Object TimeCreated, Id, Message | Format-List
```

Look for `Smart App Control Block Details` / `did not meet the Enterprise signing
level requirements`.

Options (in order of how much they change your system):
1. **Run inside WSL2** (what this project does) — SAC only applies to Windows PE
   binaries, so running Python inside a WSL2 Linux distro sidesteps it entirely without
   touching any Windows security setting.
2. **Run inside Docker Desktop** — same idea, a different container boundary. See the
   next section if Docker Desktop itself won't start.
3. **Disable Smart App Control** — Settings > Privacy & security > Windows Security >
   App & browser control > Smart App Control > Off. This is a **one-way action on
   Windows 11**: Microsoft only lets you turn it back on via a clean OS reinstall. Only
   do this if you've decided you don't want SAC at all, not just to unblock one package.

### Docker Desktop backend crashes with an "Ingest server" / orphaned socket error

Symptom: `docker build`/`docker info` fail with `failed to connect to the docker API
at npipe:////./pipe/dockerDesktopLinuxEngine`, and
`%LOCALAPPDATA%\Docker\log\host\com.docker.backend.exe.log` shows something like:

```
backend crashed ...: initializing Ingest server: listening on unix://...\run\sailor-ingest.sock:
remove ...\run\sailor-ingest.sock: The file cannot be accessed by the system.
```

This means a Unix-domain-socket file under `%LOCALAPPDATA%\Docker\run\` is orphaned
(shows as an inaccessible reparse point) and Docker's backend can't remove/recreate it
on startup. In this project, killing all Docker processes and running `wsl --shutdown`
did NOT clear it — a full system reboot is the most reliable fix (releases whatever
kernel-level handle is holding the file). If you'd rather not reboot, running directly
in a WSL2 distro (bypassing Docker Desktop's Windows-side engine) is a reasonable
substitute, since it's the same Linux-sandboxing benefit without depending on Docker
Desktop specifically — this project's `environment_report.md` documents taking exactly
that path.

### PyTorch doesn't detect the GPU, or detects it but ops are slow/fail

RTX 50-series (Blackwell) GPUs need a PyTorch build compiled against a CUDA toolkit new
enough to include kernels for that compute capability. Install from the CUDA 12.8
wheel index explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

A `pip install torch` with no index argument may resolve to an older CUDA build
without Blackwell kernels, or (on some platforms) a CPU-only build.

## hls4ml / QKeras

See [docs/hardware.md](hardware.md) for current status and the documented fallback
path (`UnavailableBackend`) when these aren't usable in your environment.
