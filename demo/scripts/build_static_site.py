#!/usr/bin/env python
"""Build a static mirror of the demo for GitHub Pages (no FastAPI backend available there).

Calls the SAME route functions in demo/app/main.py directly (import, not HTTP), so the static JSON
is byte-identical in shape and filtering to what the live FastAPI app would return — zero duplicated
logic, zero risk of the static site showing something the live app wouldn't (e.g. the excluded biased
mixed-precision result).

Output: demo/static-site/  (flat: index.html, styles.css, app.js, sw.js, manifest.webmanifest, icons/,
data/*.json). All asset references are relative, so the site works at any path depth (a GitHub Pages
project page is served from a subpath, e.g. /<repo>/, not the domain root).

Re-run this after any change to demo/app/static/* or demo/data/*.json, then redeploy (see
demo/scripts/deploy_github_pages.sh).
"""
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_STATIC = ROOT / "demo" / "app" / "static"
OUT = ROOT / "demo" / "static-site"
REPO_URL = "https://github.com/Piush2004/hep-trigger-quantization-lab"

sys.path.insert(0, str(ROOT / "demo" / "app"))
os.environ.setdefault("DEMO_REPO_URL", REPO_URL)


def write_json(path: Path, payload) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def build_data() -> None:
    import main  # noqa: E402  (demo/app/main.py; imported after sys.path insert above)

    data_out = OUT / "data"
    write_json(data_out / "metadata.json", main.metadata())
    write_json(data_out / "results.json", main.results())
    write_json(data_out / "configurations.json", main.configurations())
    write_json(data_out / "quantization.json", main.quantization())
    write_json(data_out / "pareto.json", main.pareto())
    write_json(data_out / "architecture-comparison.json", main.architecture_comparison())
    write_json(data_out / "calibration-sensitivity.json", main.calibration())
    write_json(data_out / "hls4ml.json", main.hls4ml())
    write_json(data_out / "health.json", main._health_payload())
    print(f"wrote 9 static data files to {data_out}")


def build_assets() -> None:
    # Overwrite-in-place rather than rmtree+recreate: the output file set is fully deterministic
    # (nothing is ever removed between builds), and this also sidesteps a WSL<->NTFS permission quirk
    # where directories created by a native Windows process can resist deletion from WSL.
    OUT.mkdir(parents=True, exist_ok=True)

    # Flat copy: icons/, manifest.webmanifest, sw.js, styles.css, app.js all at the site root,
    # so relative paths ("./icons/x", "./app.js") work regardless of the GitHub Pages subpath.
    shutil.copytree(APP_STATIC / "icons", OUT / "icons", dirs_exist_ok=True)
    for name in ["styles.css", "app.js", "sw.js", "manifest.webmanifest"]:
        shutil.copy2(APP_STATIC / name, OUT / name)
    shutil.copy2(APP_STATIC / "icons" / "favicon-32.png", OUT / "favicon.ico")  # conventional-path fallback

    html = (APP_STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/manifest.webmanifest"', 'href="./manifest.webmanifest"')
    html = html.replace('href="/static/icons/', 'href="./icons/')
    html = html.replace('href="/static/styles.css"', 'href="./styles.css"')
    html = html.replace('src="/static/app.js"', 'src="./app.js"')
    # Inject the static-mode flag before app.js runs (app.js reads window.STATIC_MODE at call time).
    html = html.replace("</head>", '  <script>window.STATIC_MODE = true;</script>\n</head>')
    (OUT / "index.html").write_text(html, encoding="utf-8")

    manifest = (OUT / "manifest.webmanifest").read_text(encoding="utf-8")
    manifest = manifest.replace('"start_url": "/?source=pwa"', '"start_url": "./?source=pwa"')
    manifest = manifest.replace('"scope": "/"', '"scope": "./"')
    manifest = re.sub(r'"/static/icons/', '"./icons/', manifest)
    (OUT / "manifest.webmanifest").write_text(manifest, encoding="utf-8")

    sw = (OUT / "sw.js").read_text(encoding="utf-8")
    sw = sw.replace('"/", "/static/styles.css", "/static/app.js",\n  "/static/icons/favicon.svg", "/static/icons/icon-192.png", "/static/icons/icon-512.png",\n  "/manifest.webmanifest",',
                    '"./", "./styles.css", "./app.js",\n  "./icons/favicon.svg", "./icons/icon-192.png", "./icons/icon-512.png",\n  "./manifest.webmanifest",')
    sw = sw.replace('caches.open(CACHE_VERSION).then((c) => c.put("/", copy));', 'caches.open(CACHE_VERSION).then((c) => c.put("./", copy));')
    sw = sw.replace('.catch(() => caches.match("/"))', '.catch(() => caches.match("./"))')
    # The static site has no /api/ to exclude; data/*.json here is a point-in-time snapshot (not a
    # live API), so it's fine — and useful — for it to be cached offline like the rest of the shell.
    sw = sw.replace('url.pathname.startsWith("/api/")', 'false')
    (OUT / "sw.js").write_text(sw, encoding="utf-8")

    print(f"built static site assets in {OUT}")


def main_() -> int:
    build_assets()  # wipes and recreates OUT, so this must run before build_data() writes into OUT/data
    build_data()
    print(f"\nStatic site ready at {OUT}. Deploy with demo/scripts/deploy_github_pages.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
