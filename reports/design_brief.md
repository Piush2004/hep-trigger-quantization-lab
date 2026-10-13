# HEP-Trigger Quantization Lab — Visual Design Brief

Prepared as a design-research deliverable. Scope: visual language only (color, type, layout
patterns, motion) for the existing public website (`demo/app/static/`) and the Flutter companion
app (`app/lib/theme.dart`). Information architecture, chart logic, copy, and functionality are
unchanged — this is a re-skin, not a redesign of structure.

---

## What changes, what stays (summary)

**Stays**
- Dark-first theme as the base mood (near-black, instrument-panel field). Both CERN's own
  properties (`home.cern`, `atlas.cern`) and DESY's run dark hero treatments; Linear and Stripe
  prove dark-or-light restraint reads as premium when the palette is disciplined. Nothing here
  moves the product to a light theme.
- The whole information architecture: topbar, hero, nine numbered sections, segmented controls,
  the explorer grid, architecture cards, hardware pipeline, evidence-tier badges, findings grid,
  footer. Not touched.
- The four evidence tiers (MEASURED / SIMULATED / ESTIMATED / NOT AVAILABLE) as a concept, the
  sticky blurred topbar, the skip-link, `prefers-reduced-motion` handling, Plotly with a
  transparent `paper_bgcolor`/`plot_bgcolor`, and the general "panel = bordered box" structure.
- System-font fallback stacks stay in place as the no-JS/no-web-font safety net.
- Numbered section eyebrows (`01 · Main result` … `09 · Research`) — the content genuinely is a
  sequential narrative, so numbering is earned here, unlike a marketing page.

**Changes**
1. **One action color, not two.** Retire the teal (`#5eead4`) + cyan (`#7dd3fc`) dual-bright-accent
   pairing — this combination is the single most common "AI dashboard" tell in the market right
   now. One disciplined "signal blue" becomes the only accent used for actions, links, focus rings
   and the primary chart series.
2. **The evidence-tier palette is rebuilt as its own, separate semantic set** so it never
   competes with the action color. SIMULATED moves off cyan and onto violet — which is already
   defined and unused in `app/lib/theme.dart` as `AppColors.mixed` (`#C4B5FD`) but never wired
   into `tierColors`. This fixes a latent inconsistency and gives SIMULATED its own identity.
3. **Typography moves off `system-ui` onto a deliberate three-role family**: IBM Plex Sans (UI/body),
   IBM Plex Mono (all numerals/data/code), and Source Serif 4 for the hero display line only. This
   is the single highest-leverage change for "classic + enterprise" — see rationale below.
4. **One flat 12px radius everywhere becomes a radius scale** (6/10/14/18px) so cards, buttons,
   inputs and badges carry different visual weight instead of the uniform "SaaS card kit" look.
5. **Background/panel/border hex values are refined** toward a more neutral, less saturated
   near-black, with a fourth elevation tier added for sunken/inset surfaces (metric tiles, code).
6. **Badges switch from outline/`currentColor` chips to tinted-fill chips** (light tint of the tier
   hue + a brighter text tint + a soft border), which is both more legible and reads less like a
   debug-console label.
7. **Marketing eyebrows drop the ALL-CAPS + wide tracking treatment**; data-table/column-header
   labels keep small-caps mono, because that convention is earned in an actual data table and not
   earned as decoration above a heading.
8. **A small, intentional set of motion** is added (currently the site is almost entirely static):
   one orchestrated hero entrance, chart-state transitions on segmented-control changes, and
   quiet hover feedback — explicitly *not* per-section scroll reveals or count-up numbers.
9. **ESTIMATED tier color shifts from bright amber to a desaturated brass/amber** — an "analog
   instrument" reference (oscilloscope/gauge amber) instead of generic SaaS warning-yellow.

---

## Research basis

- **Live site walked end to end**: `https://piush2004.github.io/hep-trigger-quantization-lab/`
  (hero, results chart, precision explorer, PTQ/QAT bars, architecture grid, calibration chart,
  hardware pipeline, methodology, reproducibility, research/findings).
- **Source read directly**: `demo/app/static/index.html`, `styles.css`, `app.js` (color constants
  at `app.js:16`, Plotly layout defaults at `app.js:48-54`), and `app/lib/theme.dart`.
- **CERN ecosystem, visited live**:
  - `home.cern` — near-black header strip, Roboto, large bold white headlines directly on
    full-bleed photography, default-blue text links, left-aligned, dense "latest news" grid below
    the fold. Confirms: dark hero + photography + restraint is CERN's own institutional voice, not
    an invented one.
  - `atlas.cern` — deep blue cinematic hero over detector/beamline photography, bold sans
    headline, generous whitespace, blue underline accent on section headers.
  - `desy.de` — dark navy-black hero, full-bleed macro photography of accelerator hardware (copper
    coils, Cherenkov-blue glow), large bold condensed sans headline, single light-blue pill CTA.
    This is the closest existing precedent to "dark scientific + enterprise-polished" and directly
    informs the accent-color direction below.
- **Enterprise/developer-tool references, visited live with computed-style inspection**:
  - `stripe.com` — Söhne-family sans, H1 set at a *light* weight (300) even at 44px with tight
    tracking (−0.88px), white background, restrained color use outside one gradient hero accent.
    Confirms that "enterprise-premium" does not mean heavy/bold everywhere — weight restraint at
    large sizes is itself a signal of confidence.
  - `linear.app` — background `#08090A`, text `#F7F8F8`, Inter Variable at weight 510 (not 700) for
    a 56px H1, letter-spacing −1.232px. Confirms the near-black + off-white + tight-tracking +
    medium-not-bold formula this brief recommends for the dark theme.
- **Better Design system guidance consulted** (via MCP `get-ui-principle`/`get-motion-guide`):
  color-system rules (one action color used only for actions; tier/status badges should be a
  light tint + darker/brighter text rather than a `currentColor` outline; chroma should be matched
  as a *percentage* of each hue's own gamut, not an absolute value, so the four tier colors read
  as equally vivid); typography rules (spend ~3 sizes per view; weight restraint; tabular numerals
  for anything in a column or that updates; avoid tracked-out ALL-CAPS eyebrows as a generic
  "AI-generated" tell except where the content is a genuine data-table header); spacing rules
  (tighten within groups, loosen between groups; a radius scale, not one flat radius).
  (Note: the `extract-from-url` token-sampling tool required an API key that wasn't available in
  this session, so reference palettes above were read by visiting the live pages and inspecting
  computed styles directly rather than via that tool's automated extraction.)

---

## 1. Color palette

All values are exact hex, for direct use as CSS custom properties / Dart `Color(0xFF......)`
constants. Organized by role, matching the "one action color, separate semantic set" principle.

### Backgrounds & surfaces (4 elevation tiers)

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#0A0D17` | Page background. Slightly more neutral and less saturated than the current `#0b1020` — reads closer to instrument-panel black than "electric SaaS navy." |
| `--surface` | `#131826` | Base panel/card surface (replaces `--panel`). |
| `--surface-raised` | `#1A2133` | Hover/raised state for interactive cards (architecture cards, metric tiles on hover). New tier — the current theme has no distinct hover elevation. |
| `--surface-sunken` | `#0E1220` | Inset wells: metric-tile backgrounds, code blocks, calibration/axis gutters (replaces `--panel-2`, which is nearly indistinguishable from `--panel` today). |
| `--border` | `#262E42` | Hairline dividers, card borders, input borders (replaces `--line`). |
| `--border-strong` | `#3A4560` | Emphasized border: focused input, active segmented-control group outline, card hover border. |

### Text

| Token | Hex | Role |
|---|---|---|
| `--text-primary` | `#EDF1F9` | Headings, primary body copy, metric values. |
| `--text-secondary` | `#93A1BD` | Muted/supporting copy (`.muted`, `.lede` sub-text, chart axis labels). Very close to the current `--muted` — kept almost unchanged since it already passes contrast. |
| `--text-tertiary` | `#5F6B85` | Fine print, disabled states, placeholder values — new third tier so fine-print text doesn't compete with `.muted` body text. |

### Action color (one, and only one)

| Token | Hex | Role |
|---|---|---|
| `--accent` | `#4FA6FF` | **The only accent used for anything actionable**: primary button fill, link color, focus ring, selected segmented-control state, "current configuration" highlight in charts, the primary data series in Plotly (replacing `COLORS.cuda` / `COLORS.cpu`'s dual role). A clear, confident sky-blue — a deliberate nod to Cherenkov glow and CERN's own blue (not copied from any single reference), but used with restraint instead of paired with a second bright hue. |
| `--accent-ink` | `#071022` | Near-black text set *on top of* `--accent` fills (primary button label, selected-segment label) — needs to be dark, not white, against a light-blue fill to clear contrast comfortably. |

### Secondary data accent (chart-series differentiation, not semantic, not an action)

| Token | Hex | Role |
|---|---|---|
| `--data-violet` | `#B9A6F7` | Second Plotly series color (e.g. device B vs device A, or a secondary comparison line) where a single blue series would be ambiguous. Refined from the already-defined-but-unused `AppColors.mixed` (`#C4B5FD`) in `theme.dart` — slightly desaturated here for restraint. This is explicitly *not* a second "brand" accent; it never appears on an interactive control. |

### Evidence-tier colors (semantic, fixed meaning, never reused elsewhere)

| Tier | Token | Hex | Notes |
|---|---|---|---|
| MEASURED | `--tier-measured` | `#3BC48D` | Desaturated slightly from `#34d399` — reads as "lab-precision green" rather than neon success-green. |
| SIMULATED | `--tier-simulated` | `#B9A6F7` | **Moved off cyan onto violet** (same value as `--data-violet` above — intentional, since SIMULATED *is* the tier for computed/synthetic evidence). This frees cyan entirely and removes the teal/cyan collision with the action color. |
| ESTIMATED | `--tier-estimated` | `#D9A441` | Shifted from bright amber (`#fbbf24`) to a desaturated brass tone — an analog-instrument reference (oscilloscope/gauge amber) that reads more "classic scientific instrument," less "generic UI warning." |
| NOT AVAILABLE | `--tier-na` | `#E2645C` | Slightly deepened from `#f87171` for restraint; still unambiguously "stop/unavailable" red. |

Badge treatment changes from outline/`currentColor` chips to **tinted fill + border + brightened
text**, per accessible-badge convention (a saturated outline chip reads as louder than the metadata
it's labeling):

```
background: color-mix(in srgb, var(--tier-X) 14%, transparent);
border: 1px solid color-mix(in srgb, var(--tier-X) 40%, transparent);
color: color-mix(in srgb, var(--tier-X) 85%, white);
```

(If `color-mix()` support is a concern for the target browsers, precompute the three values per
tier as flat hex instead — the formula above is given so a developer can derive them exactly.)

### Error/validation (shares the NA red family — intentional, both are "problem" states)

`--error: #E2645C` (same as `--tier-na`), `--error-ink: #2A0E0C` for text on error fills.

---

## 2. Typography

### Families

| Role | Family | Why |
|---|---|---|
| Display (hero H1 only) | **Source Serif 4** (Google Fonts), fallback `Georgia, "Times New Roman", serif` | The one deliberately "classic" move in the system. A serif display line on an otherwise sans/mono data product reads as scientific-journal/preprint gravitas (the visual register of a physics paper's title) without turning the whole UI into a journal page — it is used *only* for the hero H1, nowhere else. |
| UI / body / headings H2–H3 | **IBM Plex Sans** (Google Fonts), fallback `-apple-system, "Segoe UI", Roboto, sans-serif` | IBM Plex was designed as an enterprise/engineering type system — it is literally grounded in "enterprise," has a wide, reliable weight range (300–700, true italics), and was built as a family alongside Plex Mono, so body and data numerals share the same skeleton and x-height instead of visually clashing (the current `system-ui` + `ui-monospace` pairing is two unrelated type systems stitched together). Deliberately *not* Inter/Roboto/Open Sans — those are the default AI-agent picks and the brief specifically asks to avoid a "generic SaaS" read. |
| Monospace (all numerals, code, data labels, tier badges, axis ticks) | **IBM Plex Mono** (Google Fonts), fallback `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace` | Shares proportions with Plex Sans by design. Set `font-variant-numeric: tabular-nums` everywhere it renders a number that sits in a column or updates (metric tiles, chart hover values, the seed/precision explorer). |

Load via Google Fonts (`Source Serif 4`, `IBM Plex Sans` weights 400/500/600, `IBM Plex Mono`
weights 400/500) with `font-display: swap`; the existing system-font stacks remain as the
fallback, so nothing breaks if the font request fails.

### Type scale (rem, 16px root)

| Role | Size | Weight | Line-height | Letter-spacing | Family |
|---|---|---|---|---|---|
| Display (hero H1) | `clamp(2.25rem, 1.6rem + 3vw, 3.25rem)` (36–52px) | 600 | 1.08 | −0.01em | Source Serif 4 |
| H2 (section titles) | `1.75rem` (28px) | 600 | 1.2 | −0.01em | IBM Plex Sans |
| H3 (card/sub-section titles) | `1.125rem` (18px) | 600 | 1.3 | normal | IBM Plex Sans |
| Body / lede | `1rem` body, `1.125rem` lede (16 / 18px) | 400 | 1.6 | normal | IBM Plex Sans |
| Caption / eyebrow (marketing) | `0.8125rem` (13px) | 500 | 1.4 | 0.02em, **sentence case** | IBM Plex Mono |
| Data-table column header | `0.75rem` (12px) | 500 | 1.4 | 0.04em, **small caps / uppercase retained** | IBM Plex Mono |
| Metric value (tiles, chart hover) | `1.375rem` (22px) | 500 | 1.2 | normal, `tabular-nums` | IBM Plex Mono |
| Fine print / footnotes | `0.8125rem` (13px) | 400 | 1.5 | normal | IBM Plex Sans, `--text-tertiary` |

Two deliberate typographic rules to carry through implementation:
- **Eyebrow labels above section headings drop the ALL-CAPS + 0.12em tracking** currently in
  `.eyebrow` (`styles.css:70`) and go to sentence case at 0.02em tracking (e.g. "01 · Main
  result" stays as-is in content, loses the shouty treatment). This is the one explicit
  "generic AI-page" tell the current CSS has, and it's easy to fix without touching markup.
- **Data-table/column-header labels (`.metrics dt`, chart axis titles) keep uppercase mono** — this
  is a legitimate, earned convention in an actual data table (scientific tables/instrument
  readouts use exactly this), so it's kept distinct from the marketing-eyebrow treatment above
  rather than blanket-removed.
- Headings use `text-wrap: balance`; body/fine-print paragraphs use `text-wrap: pretty`.
- Constrain `.lede`, `.prose`, `.callout`, `.fineprint` to a measure of roughly `68ch` (the current
  760px max-width on `.lede` is already close to this and can stay).

---

## 3. Layout & component patterns

### Radius scale (replaces the single flat `--radius: 12px`)

| Token | Value | Used for |
|---|---|---|
| `--radius-sm` | 6px | Inputs, small controls, segmented-control buttons |
| `--radius-md` | 10px | Primary/secondary buttons, select dropdown |
| `--radius-lg` | 14px | Panels/section cards (`.panel`), architecture cards |
| `--radius-xl` | 18px | The hero's caution callout and the main-result chart panel only — one slightly larger radius reserved for the two most prominent containers, so the eye reads them as "the important ones" |
| `--radius-pill` | 999px | Evidence-tier badges, credibility chips, segmented-control container |

### Hero

Keep structure (eyebrow → H1 → lede → actions → credibility chips → caution callout). Changes:
- H1 moves to Source Serif 4 per above — this is the single biggest "classic" signal on the page,
  seen in the first two seconds.
- Credibility chips (`830K simulated jets`, `16 features`, …) switch from bright-accent-bordered
  pills to a quieter `--surface-raised` fill with `--text-secondary` text and a 1px `--border` —
  currently they compete with the primary CTA for attention; chips should read as supporting
  metadata, not a second set of buttons.
- The caution callout keeps its left-accent-bar treatment but the bar color changes from
  `--warn` to `--tier-na`/error red when it's flagging unavailability (as the current copy
  — "HLS synthesis not available" — actually does), reserving amber strictly for ESTIMATED.

### Navigation / topbar

Keep sticky + blurred (`backdrop-filter: blur(6px)`, `rgba(10,13,23,0.92)` using the new `--bg`).
Nav links: `--text-secondary` default, `--text-primary` on hover, `--accent` only for the current
section (if scroll-spy is ever added) — never multiple accent colors in the nav.

### Cards / panels

- `.panel`: `--surface` background, `--border` 1px border, `--radius-lg`, no shadow by default —
  borders carry depth on dark surfaces, not shadows (shadows read as muddy smears on near-black
  backgrounds and the current theme is right to avoid them).
- **One deliberate shadow exception**: the main-result chart panel (`#results`) and architecture
  cards *on hover* get a soft shadow (`0 8px 24px rgba(0,0,0,0.35)`) plus a border-color shift
  toward `--border-strong` — a single, intentional elevation moment rather than shadow-on-every-card.
- Padding scale: base unit 4px, panel padding stays near the current 28px desktop / 18px mobile,
  but internal component padding (badges, metric tiles, pipeline steps) should be visibly tighter
  than panel-to-panel spacing (the "tighten within groups, loosen between groups" rule) — e.g.
  8–10px internal padding on a metric tile vs. 22px margin between stacked panels.

### Chart containers

No border/background change beyond the panel itself — Plotly continues to render on a transparent
`paper_bgcolor`/`plot_bgcolor` so it inherits `--surface`. Update `COLORS` in `app.js:16`:
```
cuda  -> var(--accent)        (#4FA6FF)   // primary series
cpu   -> var(--data-violet)   (#B9A6F7)   // secondary/comparison series
b     -> var(--tier-estimated)(#D9A441)   // architecture-B or estimated-adjacent series
mixed -> var(--data-violet)                // was unused as a distinct series color; now shares with cpu or gets a 4th muted tone if a true 4th series is needed
grid  -> var(--border)        (#262E42)
text  -> var(--text-primary)  (#EDF1F9)
muted -> var(--text-secondary)(#93A1BD)
```
Hover label background becomes `--surface-raised` with `--border-strong` border, font family
IBM Plex Mono (already monospace, just swap the font stack).

### Buttons

- **Primary** (`.btn.primary`): fill `--accent`, text `--accent-ink`, weight 600, `--radius-md`,
  no border. Hover: slightly lighten fill (`color-mix(in srgb, var(--accent) 90%, white)`), no
  layout shift.
- **Secondary** (`.btn`): `--surface` fill, `--text-primary` text, 1px `--border`, `--radius-md`.
  Hover: border becomes `--accent`, text unchanged (border-only hover feedback, not a fill change,
  so secondary buttons stay visually quiet next to primary).
- **Outline/ghost** (nav links, "ext" GitHub link): no fill, no border, `--text-secondary` →
  `--accent` on hover. Reserve underline for inline prose links only.

### Segmented controls

Keep the current pill-container pattern. Active segment: `--accent` fill + `--accent-ink` text,
weight 600. Inactive: `--surface-sunken` fill, `--text-secondary` text. Container border
`--border`, `--radius-pill`. This mirrors the existing CSS almost exactly — only the color tokens
change, not the structure.

### Badge / chip styling

Tier badges switch to the tinted-fill formula in the color section above, `--radius-pill`,
`0.75rem` IBM Plex Mono, weight 500 (down from 700 — the heavier weight read as shouty at this
size), letter-spacing 0.04em, uppercase retained (earned here as a status-indicator convention).

### Table / metric-tile styling

- `.metrics dt`: IBM Plex Mono, `0.75rem`, uppercase, `--text-secondary`, letter-spacing 0.04em.
- `.metrics dd`: IBM Plex Mono, `1.375rem`, weight 500, `tabular-nums`, `--text-primary`; the "N/A"
  state keeps a visually distinct smaller/muted treatment as today.
- Metric tile container: `--surface-sunken` fill (not `--surface-raised` — these are inset data
  wells, not interactive cards), `--radius-md`, `--border` 1px, internal padding on the tighter
  end of the spacing scale (10–14px).

### Spacing scale

Base unit 4px: `4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96px`. Map existing ad hoc values onto it
(e.g. the current 22px panel margin → 24px, 28px panel padding → 32px desktop / stays ~18–20px
mobile) so every gap in the system is traceable to one scale instead of hand-picked pixel values.

---

## 4. Motion

Current state: the site is almost entirely static (smooth-scroll, a skip-link transition, implicit
button states) — reduced-motion handling exists but has very little to reduce. Add a small,
named set of intentional motion, nothing else:

1. **One hero entrance, once.** On first paint, the eyebrow, H1, lede, and action buttons fade and
   rise 8px into place as a single orchestrated sequence (~220ms each, ease-out, ~50ms stagger
   between the four blocks, total under 500ms). This is the one "boldness spent in one place"
   moment for the whole page. No other section gets a scroll-triggered reveal.
2. **Chart state transitions.** When the segmented controls (precision/method) or the architecture
   configuration `<select>` change, call Plotly with a `transition` config (`duration: 250,
   easing: "cubic-in-out"`) instead of a hard re-render, so markers/bars animate to their new
   position/value. This is motion that directly answers the user's action, not decoration.
3. **Quiet hover feedback.** Buttons, segmented-control options, and architecture cards get a
   120–150ms ease-out transition on `border-color`/`background-color`; architecture cards
   additionally get a 2px `translateY(-1px)` lift paired with the shadow exception above. No
   scale-bounce, no color-cycling.
4. **Explicitly excluded**: per-section scroll-reveal animations, count-up/odometer number
   animations on the metric tiles or headline stats, parallax on the hero, auto-advancing
   carousels, skeleton shimmer beyond a genuine loading state.
5. All of the above must sit inside (or respect) the existing
   `@media (prefers-reduced-motion: reduce)` block in `styles.css:34-37`, which already collapses
   animation/transition durations to near-zero — no changes needed there, just make sure new
   `transition` declarations are covered by the existing wildcard selector.

---

## 5. Why this reads as "modern, classic, enterprise-level" for *this* product

A generic "make it look premium" pass on a dashboard reaches for Inter, a single bright accent,
rounded cards with soft shadows, and scroll-triggered fade-ins — exactly the dark-SaaS-dashboard
look this product currently has and the brief wants it to move past. The choices above are
grounded in what this product actually is: a frozen, reproducible physics benchmark, not a
startup's analytics dashboard.

**Modern** comes from restraint and precision, not decoration: one action color used only where
something is actionable (so the eye always knows what it can press), tabular numerals wherever a
number sits in a column or updates, a disciplined radius and spacing scale instead of one
flat value repeated everywhere, and exactly one orchestrated motion moment instead of scattered
hover/scroll effects. This is the same discipline Stripe and Linear apply — visible in their
light-weight large headlines and tight tracking — translated into a dark, data-dense context.

**Classic** comes from the serif hero display line (a nod to the scientific-publication register
this result would eventually appear in, as a preprint or report) and from favoring hairline
borders and quiet tinted badges over heavy shadows and shouty outline chips. The brass/amber
ESTIMATED tier and the Cherenkov-adjacent signal-blue accent are drawn from the subject's own
visual vocabulary — analog instrumentation and detector glow — rather than from generic brand
color theory, which is what keeps the palette from reading as "borrowed from a SaaS kit."

**Enterprise-level** comes from IBM Plex specifically (a type system literally built for
engineering/enterprise products, with body and data sharing one skeleton instead of two
unrelated font stacks stitched together), from a badge and color system that follows accessible,
documented conventions (tinted fills, chroma matched by percentage across the four tier hues so
none reads as more or less "important" than the others by accident), and from treating the
evidence-tier system — the product's actual intellectual contribution — as the thing the entire
visual hierarchy serves, rather than decorating around it.

**What this deliberately does not do**: it does not add a second bright brand hue, does not
introduce heavy drop shadows or gradient washes, does not switch to a light theme, does not add
scroll-driven animation, and does not touch the chart logic, data, copy, or section structure.
The physics benchmark and its evidence-tier discipline stay exactly as rigorous and exactly as
visible as they are today — only the surface it's presented on changes.
