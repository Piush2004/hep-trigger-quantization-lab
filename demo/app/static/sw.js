/* HEP-Trigger Quantization Lab — service worker.
 *
 * Scope: caches the static app shell (HTML/CSS/JS/icons/manifest) only, so the page can open offline.
 * Rule: /api/* is NEVER cached here. Scientific results are always fetched fresh from the network;
 * if the network is unavailable the fetch simply fails and the page says so (see app.js), instead of
 * silently showing old numbers as if they were current. Bump CACHE_VERSION to force clients to refresh
 * the cached shell after a deploy (do this whenever index.html/app.js/styles.css change).
 */
const CACHE_VERSION = "shell-v2";
const SHELL_ASSETS = [
  "/", "/results", "/architecture", "/hardware", "/methodology", "/research",
  "/static/styles.css", "/static/app.js",
  "/static/icons/favicon.svg", "/static/icons/icon-192.png", "/static/icons/icon-512.png",
  "/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Only ever cache a successful, same-origin, non-opaque response. Without this check a transient
// error page (or, worse, a tampered/opaque response from a misbehaving intermediary) could be written
// into the offline cache and replayed to the user on every future offline load — a cache-poisoning risk.
// `type === "basic"` additionally rules out opaque cross-origin responses, which this worker should
// never see anyway (cross-origin requests are excluded below) but is checked as defense-in-depth.
function isCacheable(res) {
  return res && res.ok && res.type === "basic";
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  // Never intercept or cache API calls or cross-origin requests (e.g. the Plotly CDN) — let them go
  // straight to the network so scientific data is always fresh and third-party assets keep their own caching.
  if (url.origin !== self.location.origin || url.pathname.startsWith("/api/")) return;

  if (req.mode === "navigate") {
    // Network-first for the page itself, so an online visitor always gets the latest shell. The site
    // is multiple pages now (one per feature), so each is cached and matched under its OWN request —
    // not a single hardcoded "/", which would serve the home page's content for every other URL.
    event.respondWith(
      fetch(req).then((res) => {
        if (isCacheable(res)) {
          const copy = res.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(req, copy));
        }
        return res;
      }).catch(() => caches.match(req).then((cached) => cached || caches.match("/")))
    );
    return;
  }

  // Cache-first for static shell assets (CSS/JS/icons/manifest), refreshed in the background.
  event.respondWith(
    caches.match(req).then((cached) => {
      const network = fetch(req).then((res) => {
        if (isCacheable(res)) caches.open(CACHE_VERSION).then((c) => c.put(req, res.clone()));
        return res;
      }).catch(() => cached);
      return cached || network;
    })
  );
});
