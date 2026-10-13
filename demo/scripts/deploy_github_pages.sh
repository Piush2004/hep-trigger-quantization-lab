#!/usr/bin/env bash
# Rebuild and redeploy the static GitHub Pages mirror from demo/web-release.
# Usage: bash demo/scripts/deploy_github_pages.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "==> Rebuilding static site from demo/data (run demo/scripts/export_results.py first if research results changed)"
python demo/scripts/build_static_site.py

# mktemp -d gives a securely random, race-free path (unlike a PID-based name, which is predictable
# and vulnerable to a symlink pre-plant attack). `git worktree add` requires the target to not
# already exist, so remove the just-created empty dir immediately before handing the same path to it
# -- the directory name itself is still unpredictable, which is what actually defeats the attack.
WORKTREE=$(mktemp -d /tmp/gh-pages-deploy-XXXXXXXX)
rmdir "$WORKTREE"
echo "==> Preparing gh-pages worktree at $WORKTREE"
git worktree add --detach "$WORKTREE" gh-pages >/dev/null
cleanup() { git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Overwrite in place rather than rm -rf: avoids a WSL<->NTFS permission quirk seen during development,
# and the file set here is fully deterministic from the build step above.
cp -r demo/static-site/* "$WORKTREE"/
touch "$WORKTREE"/.nojekyll

cd "$WORKTREE"
git add -A
if git diff --cached --quiet; then
  echo "==> No changes to deploy (static site output is identical to what's already live)."
else
  git commit -q -m "Rebuild static site from demo/web-release"
  # The worktree is a detached-HEAD checkout (see `git worktree add --detach` above), so the local
  # gh-pages branch ref never moves when we commit here. Push HEAD explicitly to the remote branch
  # instead of `git push origin gh-pages`, which would push the unmoved local ref and silently no-op.
  git push origin HEAD:gh-pages
  echo "==> Pushed. Live at https://piush2004.github.io/hep-trigger-quantization-lab/ (allow a minute for GitHub Pages to rebuild)"
fi
