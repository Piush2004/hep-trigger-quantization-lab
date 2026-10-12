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

**Status: prepared, not published — awaiting your confirmation.**

- `gh auth status` shows an authenticated session: account `Piush2004`, token scopes `gist`, `read:org`, `repo`. This machine **can** create and push to a GitHub repository.
- I searched the account's existing public repositories for one matching this project (cern/hep/quant/trigger/jet in the name) and found **none** — there is no existing repository to push to.
- I have **not** created a repository or pushed anything. Creating a new public GitHub repository and pushing this project's code is an outward-facing, hard-to-reverse publishing action, so it needs your explicit go-ahead in chat before I do it, even though the tooling is available — this is a deliberate pause, not a missing capability.

**To proceed, tell me:**
- The repository name and whether it should be **public** or **private**.
- Whether to push the whole project (research + demo) or only `demo/` (a smaller public surface, pointing back to the private research repo if you keep that separate).
- Confirm you want me to create it now under `Piush2004` and push.

Once confirmed, I will create the repository, push the relevant branch(es), and report the exact URL back.

## 4. Summary table

| Target | Status |
|---|---|
| Local app (`uvicorn`) | **DONE** — see `reports/release_test_report.md` |
| Docker build/run | **BLOCKED (NOT RUN)** — Docker Desktop engine not starting on this machine |
| Hugging Face Space (live) | **BLOCKED** — no account/token available in this environment |
| GitHub repository (public) | **NOT IMPLEMENTED** — prepared; awaiting your explicit confirmation |
