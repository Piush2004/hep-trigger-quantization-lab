# Deployment report — demo/web-release

## 1. Docker

**Status: BLOCKED (NOT RUN).**

- `docker` CLI is present on this machine (Docker Desktop 29.7.2, Windows), but the Docker Desktop **engine/backend is not running**: `docker ps` / `docker info` fail with `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine ... The system cannot find the file specified.`
- I started Docker Desktop once (`Start-Process "Docker Desktop.exe"`) and waited roughly 30 minutes while completing other independent work; the engine never came up.
- This matches a previously documented issue in this same project (`reports/experiment_log.md`, "Failure 2 — Docker Desktop backend crash on orphaned socket file"), where Docker Desktop's backend has failed to start reliably on this machine before, and the project's own standing guidance was explicitly **not** to spend further time troubleshooting Docker Desktop. I honored that rather than escalating into a repair attempt.
- **No Docker build or container run was executed.** I am not claiming otherwise. `demo/Dockerfile` and `.dockerignore` were reviewed, hardened (non-root user, `HEALTHCHECK`, pinned `EXPOSE 7860`/`--host 0.0.0.0 --port 7860`), and are ready to build once the engine is available.

**Exact next action for the user:** restart Docker Desktop (or the machine) to clear the stuck backend, then run:
```bash
docker build -f demo/Dockerfile -t hep-trigger-lab-demo .
docker run -p 7860:7860 hep-trigger-lab-demo
curl http://localhost:7860/health
```
If the build/run succeeds, the health check, every `/api/*` endpoint, and the biased-result exclusion should be re-verified against the container (the same checks already passed against the local `uvicorn` run — see `reports/release_test_report.md`).

## 2. Hugging Face Space

**Status: BLOCKED (credentials required).**

- No `huggingface_hub` package, no `huggingface-cli`, and no cached token were found in this environment (`~/.cache/huggingface/token` does not exist; no `HF_TOKEN`/`HUGGINGFACE_*` environment variable is set).
- I did not attempt to create an account, log in, or guess a token. Deployment to Hugging Face Spaces needs the user's own account.

**Exact next action for the user:**
1. Create (or sign in to) a Hugging Face account at https://huggingface.co.
2. Create a new Space: Settings → SDK → **Docker**. (`demo/README.md` already carries the required YAML front matter — `sdk: docker`, `app_port: 7860` — confirmed against the current Hugging Face Hub "Spaces Configuration Reference", 2026-10-13.)
3. Either push the contents of this repository's `demo/` folder as the Space's repo root (so `demo/README.md` becomes the Space's `README.md`, `demo/Dockerfile` becomes the Space's `Dockerfile`, etc.), or connect the Space to a GitHub repository containing `demo/` at a path the Space's build picks up — either way, give me the Space's git remote URL (or a token with write access to it) and I will push and verify the build/runtime logs and the live endpoints.
4. Alternatively, give me a Hugging Face access token with `write` scope (as an environment variable or via `huggingface-cli login`) and I will create the Space and push directly.

**Live Space URL: does not exist.** No Space has been created.

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
| Docker build/run | **BLOCKED (NOT RUN)** — Docker Desktop engine not starting on this machine |
| Hugging Face Space (live) | **BLOCKED** — no account/token available in this environment |
| GitHub repository (public) | **DONE** — https://github.com/Piush2004/hep-trigger-quantization-lab |
