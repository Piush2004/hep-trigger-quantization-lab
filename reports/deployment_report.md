# Deployment report — demo/web-release

## 1. Docker

**Status: BLOCKED (NOT RUN) — root cause found, not fully resolved.**

- `docker` CLI is present (Docker Desktop 29.7.2 → 244472 after an auto-update mid-session), but the
  backend repeatedly fails to start: `docker info` fails with
  `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine ... The system
  cannot find the file specified.`
- Root cause identified directly from `%LOCALAPPDATA%\Docker\log\host\com.docker.backend.exe.log`
  (not guessed): the backend crashes on startup trying to remove a stale Unix-socket file,
  `sailor-ingest.sock`, in its runtime directory — `The file cannot be accessed by the system`. This
  matches the same class of issue previously documented in `reports/experiment_log.md` ("orphaned
  socket file").
- Fix attempted: direct deletion of the stale socket failed with the identical OS-level error even
  with the sandbox disabled (confirmed not a permission/sandbox artifact), but **renaming the parent
  `run/` directory out of the way succeeded**, forcing Docker to recreate it fresh. Docker Desktop was
  relaunched after this fix.
- **Result: still did not come up cleanly.** After relaunch, `docker info` hung for several minutes
  (a behavior change from the previous instant-fail, suggesting real progress) before ultimately
  failing with the same pipe error. The underlying cause this time was not captured in detail — likely
  a separate issue (e.g. the WSL2 `docker-desktop` backend distro itself, which `wsl -l -v` showed as
  `Stopped`). **No Docker build or container run was executed.**
- Per explicit user instruction, Docker was deprioritized in favor of Hugging Face deployment (§2),
  which does not require a working local Docker daemon — Hugging Face builds Docker-SDK Spaces on
  its own infrastructure. The Space actually deployed (§2) uses the `static` SDK instead, for a
  separate reason (billing, see below), so this remains genuinely unverified.

**Exact next action for the user:** a full machine restart is the next reasonable step (clears both
the Docker Desktop process state and the WSL2 `docker-desktop` distro state at once), then:
```bash
docker build -f demo/Dockerfile -t hep-trigger-lab-demo .
docker run -p 7860:7860 hep-trigger-lab-demo
curl http://localhost:7860/health
```

## 2. Hugging Face Space

**Status: DONE — live, static-SDK deployment.**

- Credentials: the user logged in locally via `hf auth login` (OAuth device flow, token never seen
  or handled by this session — verified only via `whoami`, which doesn't expose the secret).
- **Docker-SDK Space creation failed with `402 Payment Required`**: Hugging Face now requires a Pro
  subscription to host Docker or Gradio Spaces even on free `cpu-basic` hardware (this account is
  free-tier, confirmed via `whoami`'s `is_pro: false`). This is a real, current platform constraint,
  not a bug — discovered by attempting the real API call, not assumed in advance.
- **Deployed a `static`-SDK Space instead** (free for every account, no Docker involved on either
  side): `piushk/hep-trigger-quantization-lab`, built from `demo/static-site/` (the project's existing
  GitHub Pages mirror, already tested standalone). Required one content fix: the YAML front matter's
  `short_description` exceeded Hugging Face's 60-character limit and was shortened.
- Verified live, not just "pushed": `api.space_info(...).runtime.stage == "RUNNING"`, direct HTTP 200
  on the Space's actual URL (`*.static.hf.space`, not the `*.hf.space` pattern Docker Spaces use —
  found by inspecting the Space page's own rendered iframe `src`, since the naive URL guess 404'd),
  and a full in-browser load confirming the hero particle animation, evidence-tier badges, and the
  GitHub link (resolved from `/data/metadata.json`) all render correctly.

**Live Space URL: https://huggingface.co/spaces/piushk/hep-trigger-quantization-lab**
**Direct app URL: https://piushk-hep-trigger-quantization-lab.static.hf.space/**

**Known gap**: this is the static mirror, not the FastAPI app — no `/api/*` backend, no `/health`
endpoint, no server-rendered PWA service-worker registration test on this specific host. The GitHub
Pages mirror and this Space serve identical content from the same build script
(`demo/scripts/build_static_site.py`), so the existing frontend verification in
`reports/release_test_report.md` §C applies here too; the Docker-SDK FastAPI deployment (§1) remains
the gap if a live backend API is ever needed.

## 3. GitHub

**Status: DONE.**

- You confirmed: public repository, whole project (research + demo).
- Created `https://github.com/Piush2004/hep-trigger-quantization-lab` (public, MIT license auto-detected from the existing `LICENSE` file).
- Before pushing, ran a repo-wide scan (both branches) for AWS/private-key/OpenAI/GitHub/Hugging-Face token-shaped strings — the only match was the string literals inside `tests/test_demo_api.py`'s own secret-detection pattern list (`"ghp_"`, `"hf_"`, etc.), confirmed by inspection, not a real secret. Repo size: 8.01 MiB packed, 95 objects.
- Pushed `master` (research, frozen at `970f44a`, untouched) and `demo/web-release` (release layer, `c1ee1bc` at push time).
- Set the **default branch to `demo/web-release`**: it contains the full research history as an ancestor, plus the root `README.md`'s "Interactive demo" pointer and all the release documentation, so a visitor lands on the most complete picture. `master` remains available as the pure frozen-research line.
- GitHub's own secret scanning and push-protection are enabled on the repo (platform-level, in addition to the pre-push scan above).
- Verified live via `gh repo view`: `isPrivate: false`, `defaultBranchRef: demo/web-release`, `license: mit`.

**Repository URL:** https://github.com/Piush2004/hep-trigger-quantization-lab

## 4. Summary table

| Target | Status |
|---|---|
| Local app (`uvicorn`) | **DONE** — see `reports/release_test_report.md` |
| Docker build/run | **BLOCKED (NOT RUN)** — Docker Desktop backend fails to start (root cause partially fixed; still not fully up) |
| Hugging Face Space (live) | **DONE (static SDK)** — https://huggingface.co/spaces/piushk/hep-trigger-quantization-lab |
| GitHub repository (public) | **DONE** — https://github.com/Piush2004/hep-trigger-quantization-lab |
| GitHub Pages (live, FastAPI-equivalent static mirror) | **DONE** — https://piush2004.github.io/hep-trigger-quantization-lab/ |
