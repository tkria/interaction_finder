"""CSS and JavaScript assets for report generation.

Contains inline CSS and JS code as Python strings for embedding
in self-contained HTML reports.
"""

THEME_CSS = """
/* Shared theme tokens (Pico CSS palette mappings). Used by the report and the
   web UI so both surfaces share one colour/spacing system. */

:root {
    --spacing-compact: 0.5rem;
    --spacing-card: 0.75rem;
}

/* Light theme colors (default, explicit, and prefers-color-scheme: light) */
:root:not([data-theme=dark]),
[data-theme=light] {
    /* Semantic color mappings for theming */
    /* Entity highlighting colors */
    --entity1-bg: var(--pico-color-violet-200);
    --entity1-bg-faint: var(--pico-color-violet-150);
    --entity1-bg-hover: var(--pico-color-violet-300);
    --entity1-text: var(--pico-color-violet-450);
    --entity2-bg: var(--pico-color-lime-100);
    --entity2-bg-faint: var(--pico-color-lime-50);
    --entity2-bg-hover: var(--pico-color-lime-200);
    --entity2-text: var(--pico-color-lime-450);
    --entity-other-bg: var(--pico-color-slate-150);
    --entity-other-bg-hover: var(--pico-color-slate-250);

    /* State colors */
    --selected-bg: var(--pico-color-azure-100);
    --selected-hover-bg: var(--pico-color-azure-50);
    --rejected-bg: var(--pico-color-red-50);
    --rejected-border: var(--pico-color-red-600);
    --contentious-border: var(--pico-color-amber-100);

    /* Evidence level badge colors (1-9 scale: red→indigo gradient) */
    --evidence-1-bg: var(--pico-color-red-500);
    --evidence-2-bg: var(--pico-color-orange-500);
    --evidence-3-bg: var(--pico-color-pumpkin-500);
    --evidence-4-bg: var(--pico-color-amber-500);
    --evidence-5-bg: var(--pico-color-yellow-500);
    --evidence-6-bg: var(--pico-color-green-500);
    --evidence-7-bg: var(--pico-color-jade-500);
    --evidence-8-bg: var(--pico-color-cyan-500);
    --evidence-9-bg: var(--pico-color-indigo-500);

    /* Quote colors */
    --quote-bg: var(--pico-color-zinc-100);
    --quote-border: var(--pico-color-zinc-400);
    --quote-emphasis-bg: var(--pico-color-amber-200);
    --quote-emphasis-border: var(--pico-color-amber-600);
    --quote-dim-bg: var(--pico-color-zinc-50);

    /* Navigation colors */
    --nav-hover-bg: var(--pico-color-cyan-50);
    --nav-active-bg: var(--pico-color-cyan-100);
    --nav-border: var(--pico-color-cyan-600);
    --nav-badge-bg: var(--pico-color-zinc-400);

    /* Document badge colors */
    --doc-badge-bg: var(--pico-color-azure-600);
    --doc-badge-hover-bg: var(--pico-color-azure-500);
    --doc-badge-text: var(--pico-color-slate-50);
    --polarity-positive-bg: var(--pico-color-green-50);
    --polarity-positive-text: var(--pico-color-green-700);
    --polarity-negative-bg: var(--pico-color-red-100);
    --polarity-negative-text: var(--pico-color-red-700);
    --polarity-neutral-bg: var(--pico-color-zinc-100);
    --polarity-neutral-text: var(--pico-color-slate-600);
    --doc-link-hover-bg: var(--pico-color-azure-100);
}

/* Dark theme colors (explicit or via prefers-color-scheme) */
@media (prefers-color-scheme: dark) { :root:not([data-theme]) {
    --entity1-bg: var(--pico-color-violet-800);
    --entity1-bg-faint: var(--pico-color-violet-850);
    --entity1-bg-hover: var(--pico-color-violet-700);
    --entity1-text: var(--pico-color-violet-550);
    --entity2-bg: var(--pico-color-lime-650);
    --entity2-bg-faint: var(--pico-color-lime-800);
    --entity2-bg-hover: var(--pico-color-lime-600);
    --entity2-text: var(--pico-color-lime-550);
    --entity-other-bg: var(--pico-color-slate-800);
    --entity-other-bg-hover: var(--pico-color-slate-750);
    --selected-bg: var(--pico-color-azure-900);
    --selected-hover-bg: var(--pico-color-azure-950);
    --rejected-bg: var(--pico-color-red-950);
    --rejected-border: var(--pico-color-red-400);
    --contentious-border: var(--pico-color-amber-300);
    /* Evidence level badge colors (1-9 scale: red→indigo gradient, dark mode) */
    --evidence-1-bg: var(--pico-color-red-400);
    --evidence-2-bg: var(--pico-color-orange-400);
    --evidence-3-bg: var(--pico-color-pumpkin-400);
    --evidence-4-bg: var(--pico-color-amber-400);
    --evidence-5-bg: var(--pico-color-yellow-400);
    --evidence-6-bg: var(--pico-color-green-400);
    --evidence-7-bg: var(--pico-color-jade-400);
    --evidence-8-bg: var(--pico-color-cyan-400);
    --evidence-9-bg: var(--pico-color-indigo-400);
    --quote-bg: var(--pico-color-zinc-750);
    --quote-border: var(--pico-color-zinc-600);
    --quote-emphasis-bg: var(--pico-color-amber-800);
    --quote-emphasis-border: var(--pico-color-amber-400);
    --quote-dim-bg: var(--pico-color-zinc-850);
    --nav-hover-bg: var(--pico-color-cyan-950);
    --nav-active-bg: var(--pico-color-cyan-900);
    --nav-border: var(--pico-color-cyan-400);
    --nav-badge-bg: var(--pico-color-zinc-600);
    --doc-badge-bg: var(--pico-color-azure-400);
    --doc-badge-hover-bg: var(--pico-color-azure-500);
    --doc-badge-text: var(--pico-color-slate-950);
    --polarity-positive-bg: var(--pico-color-green-850);
    --polarity-positive-text: var(--pico-color-green-300);
    --polarity-negative-bg: var(--pico-color-red-900);
    --polarity-negative-text: var(--pico-color-red-300);
    --polarity-neutral-bg: var(--pico-color-slate-900);
    --polarity-neutral-text: var(--pico-color-slate-200);
    --doc-link-hover-bg: var(--pico-color-azure-750);
} }
[data-theme=dark] {
    --entity1-bg: var(--pico-color-violet-800);
    --entity1-bg-faint: var(--pico-color-violet-850);
    --entity1-bg-hover: var(--pico-color-violet-700);
    --entity1-text: var(--pico-color-violet-550);
    --entity2-bg: var(--pico-color-lime-650);
    --entity2-bg-faint: var(--pico-color-lime-800);
    --entity2-bg-hover: var(--pico-color-lime-600);
    --entity2-text: var(--pico-color-lime-550);
    --entity-other-bg: var(--pico-color-slate-800);
    --entity-other-bg-hover: var(--pico-color-slate-750);
    --selected-bg: var(--pico-color-azure-900);
    --selected-hover-bg: var(--pico-color-azure-950);
    --rejected-bg: var(--pico-color-red-950);
    --rejected-border: var(--pico-color-red-400);
    --contentious-border: var(--pico-color-amber-300);
    /* Evidence level badge colors (1-9 scale: red→indigo gradient, dark mode) */
    --evidence-1-bg: var(--pico-color-red-400);
    --evidence-2-bg: var(--pico-color-orange-400);
    --evidence-3-bg: var(--pico-color-pumpkin-400);
    --evidence-4-bg: var(--pico-color-amber-400);
    --evidence-5-bg: var(--pico-color-yellow-400);
    --evidence-6-bg: var(--pico-color-green-400);
    --evidence-7-bg: var(--pico-color-jade-400);
    --evidence-8-bg: var(--pico-color-cyan-400);
    --evidence-9-bg: var(--pico-color-indigo-400);
    --quote-bg: var(--pico-color-zinc-750);
    --quote-border: var(--pico-color-zinc-600);
    --quote-emphasis-bg: var(--pico-color-amber-800);
    --quote-emphasis-border: var(--pico-color-amber-400);
    --quote-dim-bg: var(--pico-color-zinc-850);
    --nav-hover-bg: var(--pico-color-cyan-950);
    --nav-active-bg: var(--pico-color-cyan-900);
    --nav-border: var(--pico-color-cyan-400);
    --nav-badge-bg: var(--pico-color-zinc-600);
    --doc-badge-bg: var(--pico-color-azure-400);
    --doc-badge-hover-bg: var(--pico-color-azure-500);
    --doc-badge-text: var(--pico-color-slate-950);
    --polarity-positive-bg: var(--pico-color-green-850);
    --polarity-positive-text: var(--pico-color-green-300);
    --polarity-negative-bg: var(--pico-color-red-900);
    --polarity-negative-text: var(--pico-color-red-300);
    --polarity-neutral-bg: var(--pico-color-slate-900);
    --polarity-neutral-text: var(--pico-color-slate-200);
    --doc-link-hover-bg: var(--pico-color-azure-750);
}
"""


# Report layout and component styling, built on the shared theme tokens above.
REPORT_CSS = (
    THEME_CSS
    + """
body {
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-areas:
        "header  header     header  header     header"
        "sidebar resizer-l  content resizer-r  rightbar";
    grid-template-columns: 320px 6px 1fr 6px 320px;
    grid-template-rows: auto 1fr;
    height: 100vh;
    overflow: hidden;
}
body.col-resizing {
    cursor: col-resize;
    user-select: none;
}
body.col-resizing * {
    cursor: col-resize !important;
}

/* Column resizers */
.col-resizer {
    background: transparent;
    cursor: col-resize;
    position: relative;
    z-index: 2;
    transition: background 0.15s;
}
#resizer-left { grid-area: resizer-l; }
#resizer-right { grid-area: resizer-r; }
.col-resizer::before {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 1px;
    background: var(--pico-muted-border-color);
    transform: translateX(-50%);
    transition: opacity 0.15s;
}
.col-resizer:hover,
.col-resizer.dragging,
.col-resizer:focus-visible {
    background: var(--pico-primary-focus, rgba(99, 102, 241, 0.2));
}
.col-resizer:hover::before,
.col-resizer.dragging::before,
.col-resizer:focus-visible::before {
    opacity: 0;
}
.col-resizer:focus-visible {
    outline: none;
}

/* Header */
header {
    grid-area: header;
    padding: var(--spacing-card);
    border-bottom: 1px solid var(--pico-muted-border-color);
    background: var(--pico-background-color);
}

.header-content {
    max-width: 100%;
}

.header-title {
    margin: 0 0 var(--spacing-compact) 0;
    font-size: 1.5rem;
}

.header-stats {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    font-size: 0.9rem;
    color: var(--pico-muted-color);
    margin-bottom: var(--spacing-card);
}

.stat-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
}

.stat-label {
    font-weight: 600;
}

.header-controls {
    display: flex;
    gap: 1rem;
    align-items: center;
    margin-top: var(--spacing-compact);
}

.header-controls label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
    font-size: 0.9rem;
    cursor: pointer;
}

/* Search input with clear button */
.search-wrapper {
    position: relative;
    flex: 1;
}
.search-wrapper input {
    width: 100%;
    margin: 0;
    padding-right: 2rem;
}
.search-clear-btn {
    position: absolute;
    right: 0.4rem;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    padding: 0.2rem 0.4rem;
    margin: 0;
    font-size: 1.1rem;
    line-height: 1;
    color: var(--pico-muted-color);
    cursor: pointer;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
}
.search-wrapper input:not(:placeholder-shown) ~ .search-clear-btn {
    opacity: 1;
    pointer-events: auto;
}
.search-clear-btn:hover {
    color: var(--pico-color);
}

/* Header icon buttons (filter gear, export, etc.) */
.icon-btn {
    position: relative;
    width: 2.5rem;
    height: 2.5rem;
    padding: 0;
    margin: 0;
    flex-shrink: 0;
    align-self: center;
}
.icon-btn-svg {
    width: 1.2rem;
    height: 1.2rem;
    fill: currentColor;
}
/* Gear icon rotates 60° when the filter panel opens, back when it closes. */
#filter-toggle .icon-btn-svg {
    transition: transform 0.25s ease-out;
}
#filter-toggle[aria-expanded="true"] .icon-btn-svg {
    transform: rotate(60deg);
}
.icon-btn-borderless {
    background: transparent;
    border: none;
    color: var(--pico-muted-color);
    box-shadow: none;
}
.icon-btn-borderless:hover,
.icon-btn-borderless:focus-visible {
    background: transparent;
    border: none;
    color: var(--pico-primary);
    box-shadow: none;
}

/* Active filter indicator dot */
.filter-active-dot {
    display: none;
    position: absolute;
    top: 0.25rem;
    right: 0.25rem;
    width: 0.5rem;
    height: 0.5rem;
    background: var(--pico-primary);
    border-radius: 50%;
}
#filter-toggle.has-active-filters .filter-active-dot {
    display: block;
}

/* Filter panel layout */
.filter-panel {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: center;
    margin-top: var(--spacing-compact);
}
.filter-panel[hidden] {
    display: none;
}
.filter-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.filter-group label,
.filter-checkbox {
    margin: 0;
    font-size: 0.9rem;
}
.filter-group select {
    width: auto;
    margin: 0;
    padding: 0.25rem 0.5rem;
    font-size: 0.85rem;
}
.filter-range-sep {
    color: var(--pico-muted-color);
}

/* Sort direction toggle */
#sort-dir-toggle {
    width: 2rem;
    height: 2rem;
    padding: 0;
    margin: 0;
    transition: transform 0.2s;
}
#sort-dir-toggle[data-dir="asc"] {
    transform: rotate(180deg);
}

/* Filter checkboxes */
.filter-checkbox {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
}
.filter-checkbox input {
    margin: 0;
}

/* Clear-filters icon button: right-flushed within the filter panel */
.filter-clear-btn {
    margin: 0 0 0 auto;
    color: var(--pico-color-red-550);
}
.filter-clear-btn:hover,
.filter-clear-btn:focus-visible {
    color: var(--pico-color-red-550);
}

/* Horizontal layout for wider screens */
@media (min-width: 1200px) {
    .header-content {
        display: flex;
        gap: 2rem;
        align-items: flex-start;
    }

    .header-info {
        flex: 1;
        min-width: 0;
    }

    .header-controls-wrapper {
        flex: 0 1 auto;
        min-width: 400px;
        max-width: 600px;
    }

    .header-stats {
        margin-bottom: 0;
    }

    .header-controls {
        margin-top: 0;
    }
}

/* Sidebar (left) */
#sidebar {
    grid-area: sidebar;
    overflow-y: auto;
    padding: var(--spacing-card);
    background: var(--pico-background-color);
    display: flex;
    flex-direction: column;
}

.pair-card {
    padding: var(--spacing-card);
    margin-bottom: var(--spacing-compact);
    border: 1px solid var(--pico-muted-border-color);
    border-radius: var(--pico-border-radius);
    cursor: pointer;
    transition: all 0.2s;
    background: var(--pico-card-background-color);
}

.pair-card:hover {
    border-color: var(--pico-primary);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.pair-card.selected {
    border-color: var(--pico-primary);
    background: var(--selected-bg);
}

.pair-card.rejected {
    background: var(--rejected-bg);
    border-color: var(--rejected-border);
}

.pair-card.contentious {
    border-color: var(--contentious-border);
    box-shadow: 0 0 0 2px var(--pico-color-amber-200) inset;
}

.pair-entities {
    display: grid;
    grid-template-columns: minmax(0, max-content) minmax(0, max-content);
    justify-content: space-between;
    gap: var(--spacing-compact);
    align-items: center;
    margin-bottom: 0.25rem;
}

.pair-entities > span {
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
}

.pair-entities > span:hover {
    color: var(--pico-primary);
}

.pair-entities > span:first-child {
    text-align: left;
}

.pair-entities > span:last-child {
    text-align: right;
}

.pair-kinds {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: var(--spacing-compact);
    font-size: 0.75rem;
    color: var(--pico-muted-color);
    margin-bottom: var(--spacing-compact);
}

.pair-kinds > span:first-child {
    text-align: left;
}

.pair-kinds > span:last-child {
    text-align: right;
}

.relationship-label {
    font-style: italic;
    font-size: 0.7rem;
    color: var(--pico-muted-color);
    cursor: pointer;
}
.relationship-label:hover {
    color: var(--pico-primary);
}

.pair-relationship-only {
    text-align: center;
    margin-bottom: var(--spacing-compact);
}

.pair-summary {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
}

.polarity-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.15rem;
    border-radius: 0.6em;
}

.polarity-count {
    padding: 0.1rem 0.35rem;
    border-radius: 0.5em;
    font-size: 0.85em;
    font-weight: 600;
    border-bottom: none !important;
}

.polarity-badge {
    padding: 0.1rem 0.35rem;
    border-radius: 0.6em;
    font-weight: 600;
}

.polarity-count.polarity-positive,
.polarity-badge.polarity-positive {
    background: var(--polarity-positive-bg);
    color: var(--polarity-positive-text);
}
.polarity-count.polarity-negative,
.polarity-badge.polarity-negative {
    background: var(--polarity-negative-bg);
    color: var(--polarity-negative-text);
}
.polarity-count.polarity-neutral,
.polarity-badge.polarity-neutral {
    background: var(--polarity-neutral-bg);
    color: var(--polarity-neutral-text);
}

.evidence-badge {
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    font-weight: 600;
    color: white;
    margin-left: auto;
}
.evidence-badge.evidence-1 { background: var(--evidence-1-bg); }
.evidence-badge.evidence-2 { background: var(--evidence-2-bg); }
.evidence-badge.evidence-3 { background: var(--evidence-3-bg); }
.evidence-badge.evidence-4 { background: var(--evidence-4-bg); }
.evidence-badge.evidence-5 { background: var(--evidence-5-bg); }
.evidence-badge.evidence-6 { background: var(--evidence-6-bg); }
.evidence-badge.evidence-7 { background: var(--evidence-7-bg); }
.evidence-badge.evidence-8 { background: var(--evidence-8-bg); }
.evidence-badge.evidence-9 { background: var(--evidence-9-bg); }

.pc-pill {
    display: inline-flex;
    gap: 0.2em;
    font-size: 0.65rem;
    font-weight: 600;
}
.pc-pill > span {
    padding: 0.1rem 0.45rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.pc-pill > span:first-child {
    border: 2px solid;
    border-radius: 2em 0 0 2em;
}
.pc-pill > span:last-child {
    border-radius: 0 2em 2em 0;
}
/* Polarity styling */
.pc-pill > .positive {
    background: var(--polarity-positive-bg);
    color: var(--polarity-positive-text);
}
.pc-pill > .negative {
    background: var(--polarity-negative-bg);
    color: var(--polarity-negative-text);
}
.pc-pill > .neutral,
.pc-pill > .irrelevant {
    background: var(--polarity-neutral-bg);
    color: var(--polarity-neutral-text);
}
/* Evidence level styling */
.pc-pill > .evidence-1 { background: var(--evidence-1-bg); color: white; }
.pc-pill > .evidence-2 { background: var(--evidence-2-bg); color: white; }
.pc-pill > .evidence-3 { background: var(--evidence-3-bg); color: white; }
.pc-pill > .evidence-4 { background: var(--evidence-4-bg); color: white; }
.pc-pill > .evidence-5 { background: var(--evidence-5-bg); color: white; }
.pc-pill > .evidence-6 { background: var(--evidence-6-bg); color: white; }
.pc-pill > .evidence-7 { background: var(--evidence-7-bg); color: white; }
.pc-pill > .evidence-8 { background: var(--evidence-8-bg); color: white; }
.pc-pill > .evidence-9 { background: var(--evidence-9-bg); color: white; }

.pair-counts {
    color: var(--pico-muted-color);
    display: inline-flex;
    align-items: center;
    gap: 0.5em;
}
.pair-counts .count-item {
    display: inline-flex;
    align-items: center;
    gap: 0.25em;
    white-space: nowrap;
    /* Suppress Pico's dotted underline on tooltip-bearing inline elements. */
    border-bottom: none !important;
}
.pair-counts .count-icon {
    width: 0.9em;
    height: 0.9em;
    flex: 0 0 auto;
    fill: currentColor;
}

/* Content area */
#content {
    grid-area: content;
    overflow-y: auto;
    padding: var(--spacing-card);
}

.content-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--pico-muted-color);
    font-size: 1.1rem;
}

.document-accordion {
    margin-bottom: var(--spacing-compact);
}
.document-accordion.doc-link-hover .document-header {
    border-color: var(--pico-color-azure-450);
    background: var(--doc-link-hover-bg);
}

.document-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-card);
    border: 1px solid var(--pico-muted-border-color);
    border-radius: var(--pico-border-radius);
    cursor: pointer;
    background: var(--pico-card-background-color);
    transition: all 0.2s;
}

.document-header:hover {
    border-color: var(--pico-primary);
    background: var(--selected-hover-bg);
}

.document-header.open {
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
    border-color: var(--pico-primary);
    background: var(--selected-bg);
}

.document-title {
    font-weight: 600;
    font-size: 1rem;
    flex: 1;
    margin-right: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.document-date {
    font-weight: 300;
    font-size: 0.85rem;
    color: var(--pico-muted-color);
    opacity: 0.7;
}

.document-stats {
    display: flex;
    gap: 1rem;
    align-items: center;
    font-size: 0.85rem;
}

.relationship-count {
    font-size: 0.8rem;
    color: var(--pico-muted-color);
    font-weight: 500;
}

.relationship-label-small {
    font-size: 0.8rem;
    color: var(--pico-muted-color);
    font-style: italic;
}

.document-content {
    display: none;
    padding: var(--spacing-card);
    border: 1px solid var(--pico-primary);
    border-top: none;
    border-bottom-left-radius: var(--pico-border-radius);
    border-bottom-right-radius: var(--pico-border-radius);
    background: var(--pico-card-background-color);
    position: relative;
}

.document-content.open {
    display: block;
}

.document-links {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.document-url-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    background: var(--doc-badge-bg);
    color: var(--doc-badge-text);
    font-weight: 600;
    font-size: 0.85rem;
    text-decoration: none;
}

.document-url-badge:hover {
    background: var(--doc-badge-hover-bg);
    color: var(--doc-badge-text);
}

.document-doi-link {
    font-size: 0.85rem;
    color: var(--pico-muted-color);
    text-decoration: none;
}

.document-doi-link:hover {
    color: var(--pico-primary);
    text-decoration: underline;
}

.document-text {
    white-space: pre-wrap;
    font-family: var(--pico-font-family);
    line-height: 1.6;
    position: relative;
}

/* Entity highlights in document and reasoning panel */
/* Default styling treats all entities as "other" until classified by JS */
.entity-highlight,
.entity-span {
    padding: 2px 4px;
    border-radius: 3px;
    cursor: pointer;
    position: relative;
    background: var(--entity-other-bg);
}
:is(.entity-highlight, .entity-span):hover {
    background: var(--entity-other-bg-hover);
}

/* Entity colors by position (entity1 vs entity2 in current pair) */
/* Outside quotes: fainter backgrounds */
:is(.entity-highlight, .entity-span).entity1 {
    background: var(--entity1-bg-faint);
    font-weight: 600;
}
:is(.entity-highlight, .entity-span).entity1:hover {
    background: var(--entity1-bg-hover);
}
:is(.entity-highlight, .entity-span).entity2 {
    background: var(--entity2-bg-faint);
    font-weight: 600;
}
:is(.entity-highlight, .entity-span).entity2:hover {
    background: var(--entity2-bg-hover);
}

/* Inside quotes: stronger backgrounds */
.quote-span :is(.entity-highlight, .entity-span).entity1 {
    background: var(--entity1-bg);
}
.quote-span :is(.entity-highlight, .entity-span).entity2 {
    background: var(--entity2-bg);
}

/* Reasoning panel: text color instead of background */
.reasoning-panel .entity-highlight {
    padding: 0;
    background: none;
    font-weight: 500;
}
.reasoning-panel .entity-highlight:hover {
    background: none;
}
.reasoning-panel .entity-highlight.entity1 {
    color: var(--entity1-text);
}
.reasoning-panel .entity-highlight.entity2 {
    color: var(--entity2-text);
}

/* Document citation links in reasoning */
.doc-link {
    color: var(--pico-color-azure-450);
    border: 1px solid;
    border-radius: 0.4em;
    font-size: 0.8em;
    padding: 0 0.2em;
    cursor: pointer;
}
.doc-link:hover {
    background: var(--doc-link-hover-bg);
}

.quote-highlight {
    background: var(--quote-bg);
    padding: 0.25rem 0;
    border-left: 4px solid var(--quote-border);
    transition: background 0.5s ease-out, border-color 0.5s ease-out;
}

.quote-highlight + .quote-highlight {
    border-left: none;
}

.quote-dim {
    background: var(--quote-dim-bg);
    padding: 0.05em 0 0.25em 0;
}

.quote-blink {
    background: var(--quote-emphasis-bg);
    border-color: var(--pico-color-amber-400);
}

.quote-highlight:not(.quote-span + .quote-highlight),
.quote-dim:not(.quote-span + .quote-dim),
.quote-blink:not(.quote-span + .quote-blink) {
    padding-left: 0.2em;
    margin-left: -0.2em;
}

.quote-highlight:not(:has(+ .quote-highlight)),
.quote-dim:not(:has(+ .quote-dim)),
.quote-blink:not(:has(+ .quote-blink)) {
    padding-right: 0.2em;
    margin-right: -0.2em;
}

/* Right sidebar */
#rightbar {
    grid-area: rightbar;
    overflow-y: auto;
    padding: var(--spacing-card);
    background: var(--pico-background-color);
}

.reasoning-panel {
    font-size: 0.9rem;
}

.pair-header {
    display: flex;
    align-items: baseline;
    gap: 0.4em;
    flex-wrap: wrap;
    padding: var(--spacing-card);
    margin-bottom: var(--spacing-card);
    border: 1px solid var(--pico-muted-border-color);
    border-radius: var(--pico-border-radius);
    background: var(--pico-card-background-color);
}
.pair-entity {
    font-weight: 600;
}
.pair-relation {
    font-style: italic;
    color: var(--pico-muted-color);
}

.reasoning-title {
    font-weight: 600;
    margin-bottom: var(--spacing-compact);
    color: var(--pico-primary);
}

.reasoning-doc-title {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--pico-primary);
    margin-bottom: 0.25rem;
}

.paper-quality {
    display: flex;
    align-items: baseline;
}
.paper-quality progress {
    width: auto;
}

.reasoning-subtitle {
    font-size: 0.85em;
    color: var(--pico-muted-color);
    margin-bottom: var(--spacing-card);
    font-style: italic;
}

.assessment-section {
    margin-bottom: var(--spacing-card);
    padding: var(--spacing-compact);
    border: 1px solid var(--pico-muted-border-color);
    border-radius: var(--pico-border-radius);
    background: var(--pico-card-sectioning-background-color);
}

.assessment-header {
    display: flex;
    align-items: center;
    cursor: pointer;
    user-select: none;
}
.assessment-header > .pc-pill {
    margin-left: auto;
}
.assessment-section.open .assessment-header {
    margin-bottom: var(--spacing-compact);
    padding-bottom: var(--spacing-compact);
    border-bottom: 1px solid var(--pico-muted-border-color);
}

.assessment-label {
    font-weight: 600;
    font-size: 0.9rem;
}

.assessment-content {
    line-height: 1.6;
}
.assessment-section:not(.open) .assessment-content {
    display: none;
}
.assessment-source {
    text-align: center;
    font-style: italic;
    font-size: 0.7rem;
    color: var(--pico-muted-color);
    margin-bottom: var(--spacing-compact);
}

.reasoning-content {
    line-height: 1.6;
    color: var(--pico-color);
}

.alias-tooltip {
    font-size: 0.85rem;
    color: var(--pico-muted-color);
    margin-top: 0.25rem;
}

.entity-details-section {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--pico-muted-border-color);
}

.entity-detail-item {
    margin-bottom: 1rem;
}

.entity-detail-item:last-child {
    margin-bottom: 0;
}

.quote-navigation {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--pico-muted-border-color);
}

.quote-nav-title {
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
    color: var(--pico-primary);
}

.quote-nav-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.quote-nav-item {
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
    padding: 0.5rem;
    margin-bottom: 0.25rem;
    border: 1px solid var(--pico-muted-border-color);
    border-radius: var(--pico-border-radius);
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
    background: var(--pico-card-background-color);
}

.quote-nav-item:hover {
    background: var(--nav-hover-bg);
    border-color: var(--nav-border);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.quote-nav-item.active {
    background: var(--nav-active-bg);
    border-color: var(--nav-border);
}

.quote-nav-left {
    display: flex;
    flex-direction: column;
    align-items: center;
    align-self: stretch;
    flex-shrink: 0;
}
.quote-number {
    width: 1.5rem;
    height: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--nav-badge-bg);
    color: white;
    border-radius: 50%;
    font-weight: 600;
    font-size: 0.75rem;
    margin-bottom: 0.2em;
}
.quote-assess-badges {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    flex: 1;
    justify-content: center;
}
.assess-num-badge {
    padding: 0.1rem 0.3rem;
    border: 1px solid var(--pico-primary);
    color: var(--pico-primary);
    border-radius: 3px;
    font-size: 0.6rem;
    font-weight: 600;
}

.quote-preview {
    flex: 1;
    line-height: 1.4;
    color: var(--pico-color);
    font-size: 0.8rem;
}

.other-pair-faded {
    color: var(--pico-muted-color);
}
.other-pair-highlight {
    font-weight: 600;
}

/* Utility classes */
.hidden {
    display: none !important;
}

/* Export dialog */
.export-modal {
    width: min(720px, 95vw);
    max-width: none;
    max-height: 90vh;
    overflow-y: auto;
}
.export-modal > footer {
    display: flex;
    gap: 0.5rem;
    justify-content: flex-end;
    /* Pico's default footer band is oversized for a utility dialog: trim the
       vertical padding and the gap it leaves above the content. */
    padding: 0.75rem var(--pico-block-spacing-horizontal);
    margin-top: 1rem;
}
.export-modal > footer button {
    width: auto;
    margin: 0;
    padding: 0.4rem 1rem;
    font-size: 0.9rem;
}
.export-modal fieldset {
    margin-bottom: 1rem;
}
.export-modal fieldset:last-of-type {
    margin-bottom: 0;
}
.export-json-hint {
    display: block;
    margin: 0.5rem 0 0.35rem;
    color: var(--pico-muted-color);
    font-style: italic;
}
.export-filters-summary {
    margin-left: 0.4rem;
    color: var(--pico-muted-color);
    font-style: italic;
    font-weight: normal;
}
.export-col-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.25rem 1rem;
}
.export-col-grid label {
    margin: 0;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
}
.export-col-grid[data-disabled="true"] label {
    opacity: 0.5;
    cursor: not-allowed;
}
.export-preview-section {
    margin-top: 1rem;
}
.export-preview-section > strong {
    display: block;
    margin-bottom: 0.4rem;
}
.export-preview-wrap {
    max-height: 220px;
    overflow: auto;
    border: 1px solid var(--pico-muted-border-color);
    border-radius: var(--pico-border-radius);
}
#export-preview-table {
    margin: 0;
    font-size: 0.8rem;
    white-space: nowrap;
    border-collapse: separate;
    border-spacing: 0;
}
#export-preview-table th,
#export-preview-table td {
    padding: 0.3rem 0.5rem;
}
/* Keep column labels visible while scrolling the full row set. */
#export-preview-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: var(--pico-card-background-color, var(--pico-background-color));
    box-shadow: inset 0 -1px var(--pico-muted-border-color);
}
#export-row-count {
    display: block;
    margin-top: 0.4rem;
    color: var(--pico-muted-color);
}
/* Entity sort: keep the label and select on one line below the switches. */
#export-columns-entities label[for="export-entity-sort"] {
    display: inline-block;
    margin: 0.5rem 0.5rem 0 0;
}
#export-columns-entities #export-entity-sort {
    display: inline-block;
    width: auto;
    margin: 0;
    vertical-align: middle;
}
.export-toast {
    position: fixed;
    bottom: 1.5rem;
    left: 50%;
    transform: translateX(-50%);
    background: var(--pico-primary);
    color: var(--pico-primary-inverse, white);
    padding: 0.5rem 1rem;
    border-radius: var(--pico-border-radius);
    z-index: 10000;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
}
.export-toast.visible {
    opacity: 1;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--pico-background-color);
}

::-webkit-scrollbar-thumb {
    background: var(--pico-muted-border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--pico-muted-color);
}
"""
)

REPORT_JS = """
// Report interactivity

// NOTE: Markdown rendering is done in Python during report generation.
// Pre-rendered document HTML is embedded in <template> tags and cloned on-demand.
// This approach provides fast initial page load and efficient DOM reuse.

// Global state
const state = {
    selectedPairId: null,
    openDocumentIdx: null,
    searchQuery: '',
    showRejected: false,
    // Filter/sort state
    sortField: 'default',     // 'default', 'evidence', 'relevance', 'docs', 'quotes', 'entity'
    sortDir: 'desc',          // 'asc', 'desc'
    evidenceMin: 0,           // 0 = any, 1-9 = specific
    evidenceMax: 10,          // 10 = any, 1-9 = specific
    relevanceMin: 0,          // 0 = any, 1-5 = specific
    relevanceMax: 6,          // 6 = any, 1-5 = specific
    showContentious: false,
    // Strict-by-default: hide pairs whose subject-side entity failed the
    // taxonomic check. They are kept and scored, just collapsed until expanded.
    // 'na' pairs (gate did not run) are never hidden by this.
    strictOnTopic: true,
};

// URL state management
function getStateFromURL() {
    const params = new URLSearchParams(window.location.search);
    // Parse scroll positions (sidebar, content, rightbar)
    const scrollParam = params.get('scroll');
    let scroll = [0, 0, 0];
    if (scrollParam) {
        const parts = scrollParam.split(',').map(n => parseInt(n, 10));
        if (parts.length === 3 && parts.every(n => !Number.isNaN(n))) {
            scroll = parts;
        }
    }
    return {
        pair: params.has('pair') ? parseInt(params.get('pair'), 10) : null,
        doc: params.has('doc') ? parseInt(params.get('doc'), 10) : null,
        search: params.get('search') || '',
        rejected: params.get('rejected') === '1',
        scroll: scroll,
        // Filter/sort params
        sortField: params.get('sort') || 'default',
        sortDir: params.get('dir') || 'desc',
        evidenceMin: params.has('emin') ? parseInt(params.get('emin'), 10) : 0,
        evidenceMax: params.has('emax') ? parseInt(params.get('emax'), 10) : 10,
        relevanceMin: params.has('rmin') ? parseInt(params.get('rmin'), 10) : 0,
        relevanceMax: params.has('rmax') ? parseInt(params.get('rmax'), 10) : 6,
        contentious: params.get('contentious') === '1',
        strictOnTopic: params.get('alltopics') !== '1',
    };
}

function getScrollPositions() {
    const sidebar = document.getElementById('sidebar');
    const content = document.getElementById('content');
    const rightbar = document.getElementById('rightbar');
    return [
        sidebar ? sidebar.scrollTop : 0,
        content ? content.scrollTop : 0,
        rightbar ? rightbar.scrollTop : 0,
    ];
}

function updateURL(usePushState = false, outgoingScroll = null) {
    const params = new URLSearchParams();
    if (state.selectedPairId !== null) {
        params.set('pair', state.selectedPairId);
    }
    if (state.openDocumentIdx !== null) {
        params.set('doc', state.openDocumentIdx);
    }
    if (state.searchQuery) {
        params.set('search', state.searchQuery);
    }
    if (state.showRejected) {
        params.set('rejected', '1');
    }
    // Filter/sort params (only when non-default)
    if (state.sortField !== 'default') {
        params.set('sort', state.sortField);
    }
    if (state.sortDir !== 'desc') {
        params.set('dir', state.sortDir);
    }
    if (state.evidenceMin > 0) {
        params.set('emin', state.evidenceMin);
    }
    if (state.evidenceMax < 10) {
        params.set('emax', state.evidenceMax);
    }
    if (state.relevanceMin > 0) {
        params.set('rmin', state.relevanceMin);
    }
    if (state.relevanceMax < 6) {
        params.set('rmax', state.relevanceMax);
    }
    if (state.showContentious) {
        params.set('contentious', '1');
    }
    if (!state.strictOnTopic) {
        params.set('alltopics', '1');
    }
    // Include scroll positions
    const scroll = getScrollPositions();
    if (scroll.some(v => v > 0)) {
        params.set('scroll', scroll.join(','));
    }
    const newURL = params.toString() ? `?${params.toString()}` : window.location.pathname;
    if (usePushState) {
        // Cancel pending scroll updates and save current scroll to state we're leaving
        cancelPendingScrollUpdate();
        const prev = history.state;
        if (prev) {
            history.replaceState({ ...prev, scroll: outgoingScroll || getScrollPositions() }, '');
        }
        // Push new state with previous pair/doc for back-detection
        history.pushState({
            pair: state.selectedPairId,
            doc: state.openDocumentIdx,
            scroll: scroll,
            prevPair: prev?.pair ?? null,
            prevDoc: prev?.doc ?? null,
        }, '', newURL);
    } else {
        history.replaceState({
            ...history.state,
            pair: state.selectedPairId,
            doc: state.openDocumentIdx,
            scroll: scroll,
        }, '', newURL);
    }
}

// Check if navigating to (newPairId, newDocIdx) would return to the previous history state
function wouldReturnToPrevious(newPairId, newDocIdx) {
    const s = history.state;
    if (!s || s.prevPair === undefined) return false;
    return s.prevPair === newPairId && s.prevDoc === newDocIdx;
}

// Debounced scroll handler for URL updates
let scrollUpdateTimeout = null;
function cancelPendingScrollUpdate() {
    if (scrollUpdateTimeout) {
        clearTimeout(scrollUpdateTimeout);
        scrollUpdateTimeout = null;
    }
}
function handleScrollForURL() {
    if (scrollUpdateTimeout) return;
    scrollUpdateTimeout = setTimeout(() => {
        scrollUpdateTimeout = null;
        updateURL(false);  // replaceState for scroll
    }, 500);
}

// Save current scroll to history.state before navigating away
function saveScrollAndGoBack() {
    cancelPendingScrollUpdate();
    if (history.state) {
        history.replaceState({ ...history.state, scroll: getScrollPositions() }, '');
    }
    history.back();
}

function restoreStateFromURL(scrollOverride) {
    const urlState = getStateFromURL();
    // Restore search and rejected filter first (affects pair visibility)
    state.searchQuery = urlState.search;
    state.showRejected = urlState.rejected;
    // Restore filter/sort state
    state.sortField = urlState.sortField;
    state.sortDir = urlState.sortDir;
    state.evidenceMin = urlState.evidenceMin;
    state.evidenceMax = urlState.evidenceMax;
    state.relevanceMin = urlState.relevanceMin;
    state.relevanceMax = urlState.relevanceMax;
    state.showContentious = urlState.contentious;
    state.strictOnTopic = urlState.strictOnTopic;
    // Update UI controls to match
    document.getElementById('search-input').value = urlState.search;
    document.getElementById('show-rejected').checked = urlState.rejected;
    document.getElementById('sort-field').value = urlState.sortField;
    document.getElementById('sort-dir-toggle').dataset.dir = urlState.sortDir;
    document.getElementById('evidence-min').value = urlState.evidenceMin;
    document.getElementById('evidence-max').value = urlState.evidenceMax;
    document.getElementById('relevance-min').value = urlState.relevanceMin;
    document.getElementById('relevance-max').value = urlState.relevanceMax;
    document.getElementById('contentious-only').checked = urlState.contentious;
    document.getElementById('on-topic-only').checked = urlState.strictOnTopic;
    // Update filter UI state
    updateFilterActiveIndicator();
    updateRangeFilterOptions();
    // Reset selection state (will be set below if URL specifies valid pair)
    state.selectedPairId = null;
    state.openDocumentIdx = null;
    // Restore pair selection if valid
    if (urlState.pair !== null && !Number.isNaN(urlState.pair)) {
        const filtered = getFilteredPairs();
        const match = findFilteredPairById(urlState.pair, filtered);
        if (match) {
            state.selectedPairId = urlState.pair;
            // Restore doc selection if valid and within bounds
            const docGroups = JSON.parse(match.card.dataset.docGroups || '[]');
            if (urlState.doc !== null && !Number.isNaN(urlState.doc) &&
                urlState.doc >= 0 && urlState.doc < docGroups.length) {
                state.openDocumentIdx = urlState.doc;
            }
        }
    }
    // Store scroll positions to restore after render (history.state takes precedence)
    state.pendingScroll = scrollOverride ?? urlState.scroll;
}

function applyPendingScroll() {
    if (!state.pendingScroll) return;
    const [sidebarScroll, contentScroll, rightbarScroll] = state.pendingScroll;
    const sidebar = document.getElementById('sidebar');
    const content = document.getElementById('content');
    const rightbar = document.getElementById('rightbar');
    if (sidebar) sidebar.scrollTop = sidebarScroll;
    if (content) content.scrollTop = contentScroll;
    if (rightbar) rightbar.scrollTop = rightbarScroll;
    state.pendingScroll = null;
}

function ensureSelectedPairVisible() {
    if (state.selectedPairId === null) return;
    const card = document.getElementById(`pair-${state.selectedPairId}`);
    if (card) card.scrollIntoView({ block: 'nearest' });
}

function getPairIdFromCard(card) {
    if (!card || !card.id) return NaN;
    const parts = card.id.split('-');
    if (parts.length < 2) return NaN;
    const id = parseInt(parts[1], 10);
    return Number.isNaN(id) ? NaN : id;
}

function findFilteredPairById(pairId, filteredPairs) {
    const list = filteredPairs || getFilteredPairs();
    for (let idx = 0; idx < list.length; idx += 1) {
        const card = list[idx];
        if (getPairIdFromCard(card) === pairId) {
            return { card, index: idx };
        }
    }
    return null;
}

// Clear search input and update display
function clearSearch() {
    const searchInput = document.getElementById('search-input');
    if (!searchInput.value) return;
    searchInput.value = '';
    handleSearch({ target: searchInput }, true);
}

// DEBUG: Trace all sidebar scroll changes
let _debugTraceScroll = false;
{
    const sidebarEl = document.getElementById('sidebar');
    if (sidebarEl) {
        sidebarEl.addEventListener('scroll', () => {
            if (_debugTraceScroll) {
                console.trace('[scroll trace] sidebar scrollTop:', sidebarEl.scrollTop);
            }
        });
    }
}

// Initialize report
function initReport() {
    // Restore column widths and wire up the resizer handles.
    initColumnResizers();
    // Wire up the export dialog (Pico modal).
    initExportDialog();

    // Set up event listeners
    const searchInput = document.getElementById('search-input');
    searchInput.addEventListener('input', handleSearch);
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            clearSearch();
            searchInput.blur();
        }
    });
    document.getElementById('search-clear').addEventListener('click', clearSearch);
    // Filter panel toggle
    document.getElementById('filter-toggle').addEventListener('click', toggleFilterPanel);
    // Filter/sort controls
    document.getElementById('show-rejected').addEventListener('change', handleToggleRejected);
    document.getElementById('sort-field').addEventListener('change', handleSortChange);
    document.getElementById('sort-dir-toggle').addEventListener('click', handleSortDirToggle);
    document.getElementById('evidence-min').addEventListener('change', handleEvidenceMinChange);
    document.getElementById('evidence-max').addEventListener('change', handleEvidenceMaxChange);
    document.getElementById('relevance-min').addEventListener('change', handleRelevanceMinChange);
    document.getElementById('relevance-max').addEventListener('change', handleRelevanceMaxChange);
    document.getElementById('contentious-only').addEventListener('change', handleToggleContentious);
    document.getElementById('on-topic-only').addEventListener('change', handleToggleOnTopicOnly);
    document.getElementById('filter-clear').addEventListener('click', clearAllFilters);

    // Add click handlers to pre-rendered pair cards
    const sidebar = document.getElementById('sidebar');
    sidebar.querySelectorAll('.pair-card').forEach((card) => {
        const pairId = getPairIdFromCard(card);
        card.addEventListener('click', () => selectPair(pairId));
        // Add click handlers to entity names for search filtering (uses KIND:name syntax)
        const entitySpans = card.querySelectorAll('.pair-entities > span');
        entitySpans.forEach((span) => {
            span.addEventListener('click', (e) => {
                e.stopPropagation();
                const kind = span.dataset.kind || '';
                const name = span.textContent.trim();
                const searchInput = document.getElementById('search-input');
                searchInput.value = formatFilterQuery(kind, name);
                handleSearch({ target: searchInput }, true);
            });
        });
        // Add click handlers to relationship labels for search filtering (uses relation:type syntax)
        const relLabels = card.querySelectorAll('.relationship-label');
        relLabels.forEach((label) => {
            label.addEventListener('click', (e) => {
                e.stopPropagation();
                const rel = card.dataset.rel || '';
                const searchInput = document.getElementById('search-input');
                searchInput.value = formatFilterQuery('relation', rel);
                handleSearch({ target: searchInput }, true);
            });
        });
    });

    // Handle browser back/forward navigation
    window.addEventListener('popstate', (event) => {
        _debugTraceScroll = true;
        restoreStateFromURL(event.state?.scroll);
        updateHeaderCounts();
        updatePairListDisplay();
        renderContent();
        renderReasoning(false);
        applyPendingScroll();
        ensureSelectedPairVisible();
        setTimeout(() => { _debugTraceScroll = false; }, 1000);
    });

    // Add scroll listeners for URL updates (debounced)
    document.getElementById('sidebar').addEventListener('scroll', handleScrollForURL);
    document.getElementById('content').addEventListener('scroll', handleScrollForURL);
    document.getElementById('rightbar').addEventListener('scroll', handleScrollForURL);
    // Set up delegated doc-link hover handlers (once, not per-render)
    initDocLinkHover();

    // Restore state from URL before initial render
    restoreStateFromURL();
    const hasScrollToRestore = state.pendingScroll && state.pendingScroll.some(v => v > 0);

    // Initial render
    updateHeaderCounts();
    updatePairListDisplay();
    renderContent();
    renderReasoning(!hasScrollToRestore);  // Don't reset scroll if restoring from URL

    // Apply scroll positions after render
    applyPendingScroll();

    // Set initial history.state so scroll restoration works on first back navigation
    updateURL(false);

    // Clear startup-busy indicators. The content placeholder has already been
    // overwritten by renderContent() (which inserts its own non-busy
    // placeholder); the header title carries the busy spinner until now.
    document.querySelector('.header-title')?.removeAttribute('aria-busy');
}

// Format a filter query string, quoting if value contains spaces
function formatFilterQuery(filterName, value) {
    if (value.includes(' ')) {
        return `${filterName}:"${value}"`;
    }
    return `${filterName}:${value}`;
}

// Parse search query into structured filters and free text
// Supports: relation:VALUE, relation:"quoted value", ENTITY_KIND:VALUE, etc.
function parseSearchQuery(query) {
    const filters = [];
    // Match FILTER:VALUE or FILTER:"quoted value" patterns
    // Filter names are alphanumeric/underscore, values are quoted or unquoted
    const filterPattern = /(\\w+):(?:"([^"]+)"|(\\S+))/gi;
    let match;
    let lastIndex = 0;
    const textParts = [];
    while ((match = filterPattern.exec(query)) !== null) {
        // Collect text before this match
        if (match.index > lastIndex) {
            textParts.push(query.slice(lastIndex, match.index));
        }
        lastIndex = filterPattern.lastIndex;
        const filterName = match[1].toLowerCase();
        const filterValue = (match[2] || match[3]).toLowerCase();
        filters.push({ name: filterName, value: filterValue });
    }
    // Collect remaining text after last match
    if (lastIndex < query.length) {
        textParts.push(query.slice(lastIndex));
    }
    const freeText = textParts.join(' ').trim().toLowerCase();
    return { filters, freeText };
}

// Check if a pair card matches a structured filter
function matchesFilter(card, filter) {
    const { name, value } = filter;
    if (name === 'relation' || name === 'rel') {
        // Match relationship exactly (case-insensitive, normalise underscores to spaces)
        const rel = card.dataset.rel.toLowerCase().replace(/_/g, ' ');
        const target = value.replace(/_/g, ' ');
        return rel === target;
    }
    // For entity kind filters: match canonical entity name where kind matches
    const entitySpans = card.querySelectorAll('.pair-entities > span');
    const e1Kind = (entitySpans[0]?.dataset.kind || '').toLowerCase();
    const e2Kind = (entitySpans[1]?.dataset.kind || '').toLowerCase();
    const e1Name = card.dataset.e1.toLowerCase();
    const e2Name = card.dataset.e2.toLowerCase();
    // Check if filter name matches either entity's kind
    if (name === e1Kind && e1Name === value) return true;
    if (name === e2Kind && e2Name === value) return true;
    return false;
}

// Search handler
function handleSearch(e, usePushState = false) {
    // Snapshot scroll before display changes so the outgoing history entry
    // remembers where the user was, not where the sidebar ended up post-filter
    const scrollBeforeChange = usePushState ? getScrollPositions() : null;
    state.searchQuery = e.target.value.toLowerCase();
    updateHeaderCounts();
    updatePairListDisplay();  // May clear selection if pair no longer matches filter
    updateURL(usePushState, scrollBeforeChange);
    renderContent();
    renderReasoning();
}

// Toggle rejected pairs
function handleToggleRejected(e) {
    state.showRejected = e.target.checked;
    applyFiltersAndSort();
}

// Toggle the on-topic-only filter (hide pairs that failed the subject check)
function handleToggleOnTopicOnly(e) {
    state.strictOnTopic = e.target.checked;
    applyFiltersAndSort();
}

// Toggle filter panel visibility
function toggleFilterPanel() {
    const panel = document.getElementById('filter-panel');
    const toggle = document.getElementById('filter-toggle');
    const isOpen = !panel.hidden;
    panel.hidden = isOpen;
    toggle.setAttribute('aria-expanded', !isOpen);
}

// Check if any non-default filters are active
function hasNonDefaultFilters() {
    return (
        state.sortField !== 'default' ||
        state.sortDir !== 'desc' ||
        state.evidenceMin > 0 ||
        state.evidenceMax < 10 ||
        state.relevanceMin > 0 ||
        state.relevanceMax < 6 ||
        state.showRejected ||
        state.showContentious ||
        !state.strictOnTopic
    );
}

// Update filter active indicator on gear button, and show/hide the
// "Clear filters" button accordingly.
function updateFilterActiveIndicator() {
    const active = hasNonDefaultFilters();
    document.getElementById('filter-toggle').classList.toggle('has-active-filters', active);
    document.getElementById('filter-clear').hidden = !active;
}

// Reset every filter/sort control to its default and re-render.
function clearAllFilters() {
    state.searchQuery = '';
    state.showRejected = false;
    state.showContentious = false;
    state.strictOnTopic = true;
    state.sortField = 'default';
    state.sortDir = 'desc';
    state.evidenceMin = 0;
    state.evidenceMax = 10;
    state.relevanceMin = 0;
    state.relevanceMax = 6;
    // Sync UI controls with state.
    document.getElementById('search-input').value = '';
    document.getElementById('show-rejected').checked = false;
    document.getElementById('contentious-only').checked = false;
    document.getElementById('on-topic-only').checked = true;
    document.getElementById('sort-field').value = 'default';
    document.getElementById('sort-dir-toggle').dataset.dir = 'desc';
    document.getElementById('evidence-min').value = '0';
    document.getElementById('evidence-max').value = '10';
    document.getElementById('relevance-min').value = '0';
    document.getElementById('relevance-max').value = '6';
    updateRangeFilterOptions();
    applyFiltersAndSort();
    updateURL(true);
}

// Sort field change handler
function handleSortChange() {
    state.sortField = document.getElementById('sort-field').value;
    applyFiltersAndSort();
}

// Sort direction toggle handler
function handleSortDirToggle() {
    const btn = document.getElementById('sort-dir-toggle');
    state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
    btn.dataset.dir = state.sortDir;
    applyFiltersAndSort();
}

// Evidence min filter handler
function handleEvidenceMinChange() {
    state.evidenceMin = parseInt(document.getElementById('evidence-min').value, 10);
    applyFiltersAndSort();
}

// Evidence max filter handler
function handleEvidenceMaxChange() {
    state.evidenceMax = parseInt(document.getElementById('evidence-max').value, 10);
    applyFiltersAndSort();
}

// Relevance min filter handler
function handleRelevanceMinChange() {
    state.relevanceMin = parseInt(document.getElementById('relevance-min').value, 10);
    applyFiltersAndSort();
}

// Relevance max filter handler
function handleRelevanceMaxChange() {
    state.relevanceMax = parseInt(document.getElementById('relevance-max').value, 10);
    applyFiltersAndSort();
}

// Update disabled state of range filter options to prevent invalid ranges
function updateRangeFilterOptions() {
    const ranges = [
        ['evidence-min', 'evidence-max', state.evidenceMin, state.evidenceMax, 0, 10],
        ['relevance-min', 'relevance-max', state.relevanceMin, state.relevanceMax, 0, 6],
    ];
    for (const [minId, maxId, minVal, maxVal, minSentinel, maxSentinel] of ranges) {
        document.getElementById(minId).querySelectorAll('option').forEach(opt => {
            const v = parseInt(opt.value, 10);
            opt.disabled = v > minSentinel && maxVal < maxSentinel && v > maxVal;
        });
        document.getElementById(maxId).querySelectorAll('option').forEach(opt => {
            const v = parseInt(opt.value, 10);
            opt.disabled = v < maxSentinel && minVal > minSentinel && v < minVal;
        });
    }
}

// Contentious filter handler
function handleToggleContentious(e) {
    state.showContentious = e.target.checked;
    applyFiltersAndSort();
}

// Combined filter/sort application
function applyFiltersAndSort() {
    updateFilterActiveIndicator();
    updateRangeFilterOptions();
    updateHeaderCounts();
    updatePairListDisplay();
    updateURL(false);
    renderContent();
    renderReasoning();
}

// Update header counts based on current filter
function updateHeaderCounts() {
    const filtered = getFilteredPairs();

    // Count entities by kind (from data-kind attributes on entity name spans)
    const entityKinds = {};
    filtered.forEach(card => {
        const entitySpans = card.querySelectorAll('.pair-entities > span');
        const kind1 = entitySpans[0]?.dataset.kind || 'unknown';
        const kind2 = entitySpans[1]?.dataset.kind || 'unknown';

        if (!entityKinds[kind1]) entityKinds[kind1] = new Set();
        if (!entityKinds[kind2]) entityKinds[kind2] = new Set();

        entityKinds[kind1].add(card.dataset.e1);
        entityKinds[kind2].add(card.dataset.e2);
    });

    // Count unique documents
    const docIndices = new Set();
    filtered.forEach(card => {
        const docs = card.dataset.docs.trim();
        if (docs) {
            docs.split(' ').forEach(idx => docIndices.add(idx));
        }
    });

    // Build stat values
    const stats = {
        pairs: filtered.length,
        documents: docIndices.size,
    };
    for (const [kind, names] of Object.entries(entityKinds)) {
        stats[`entity-kind-${kind}`] = names.size;
    }

    // Update all elements with data-stat attributes
    document.querySelectorAll('[data-stat]').forEach(el => {
        const statName = el.dataset.stat;
        if (statName in stats) {
            el.textContent = stats[statName];
        }
    });
}

// Filter pairs based on search and rejected toggle
function getFilteredPairs() {
    const allPairs = document.querySelectorAll('.pair-card');
    let filtered = Array.from(allPairs).filter(card => {
        // Filter rejected
        if (!state.showRejected && card.dataset.accepted === 'false') {
            return false;
        }
        // Subject-trust gate (strict by default): hide pairs whose subject-side
        // entity failed the same-kind taxonomic check. 'pass' and 'na' (gate did
        // not run on this pair) are always shown.
        if (state.strictOnTopic && card.dataset.subjectTrust === 'fail') {
            return false;
        }
        // Evidence range filter
        const overall = parseInt(card.dataset.overall, 10);
        if (state.evidenceMin > 0 && overall < state.evidenceMin) {
            return false;
        }
        if (state.evidenceMax < 10 && overall > state.evidenceMax) {
            return false;
        }
        // Relevance range filter
        const relevance = parseInt(card.dataset.relevance, 10);
        if (state.relevanceMin > 0 && relevance < state.relevanceMin) {
            return false;
        }
        if (state.relevanceMax < 6 && relevance > state.relevanceMax) {
            return false;
        }
        // Contentious filter
        if (state.showContentious && card.dataset.contentious !== 'true') {
            return false;
        }
        // Search filter with structured filter support
        if (state.searchQuery) {
            const { filters, freeText } = parseSearchQuery(state.searchQuery);
            // All structured filters must match
            for (const filter of filters) {
                if (!matchesFilter(card, filter)) {
                    return false;
                }
            }
            // Free text must match (substring search across entity names, aliases, relationship)
            if (freeText) {
                const searchable = [
                    card.dataset.e1,
                    card.dataset.e1a,
                    card.dataset.e2,
                    card.dataset.e2a,
                    card.dataset.rel
                ].join(' ').toLowerCase();
                if (!searchable.includes(freeText)) {
                    return false;
                }
            }
        }
        return true;
    });
    // Sort the filtered results
    filtered.sort((a, b) => {
        // Primary: accepted before rejected
        const aAccepted = a.dataset.accepted === 'true';
        const bAccepted = b.dataset.accepted === 'true';
        if (aAccepted !== bAccepted) return bAccepted - aAccepted;
        // Secondary: user-selected sort field
        let comparison = 0;
        switch (state.sortField) {
            case 'default':
                // Rank-sum fusion of pair_topic_rel and age_w: lower = better.
                // Invert the subtraction so the direction toggle works the
                // same way as the other (higher = better) fields.
                comparison = parseFloat(b.dataset.rankScore) - parseFloat(a.dataset.rankScore);
                break;
            case 'evidence':
                comparison = parseInt(a.dataset.overall, 10) - parseInt(b.dataset.overall, 10);
                break;
            case 'relevance':
                comparison = parseInt(a.dataset.relevance, 10) - parseInt(b.dataset.relevance, 10);
                break;
            case 'docs':
                comparison = parseInt(a.dataset.docCount, 10) - parseInt(b.dataset.docCount, 10);
                break;
            case 'quotes':
                comparison = parseInt(a.dataset.quoteCount, 10) - parseInt(b.dataset.quoteCount, 10);
                break;
            case 'entity':
                comparison = a.dataset.e1.localeCompare(b.dataset.e1, undefined, { sensitivity: 'base' });
                if (comparison === 0) {
                    comparison = a.dataset.e2.localeCompare(b.dataset.e2, undefined, { sensitivity: 'base' });
                }
                break;
            default:
                // Fall back to evidence if unknown sort field
                comparison = parseInt(a.dataset.overall, 10) - parseInt(b.dataset.overall, 10);
        }
        // Apply sort direction (desc = higher values first)
        return state.sortDir === 'desc' ? -comparison : comparison;
    });
    return filtered;
}

// Update pair list visibility and selection based on filters
function updatePairListDisplay() {
    const filtered = getFilteredPairs();
    const sidebar = document.getElementById('sidebar');
    const allCards = sidebar.querySelectorAll('.pair-card');
    const filteredSet = new Set(filtered);
    // Clear selection if no longer visible
    if (state.selectedPairId !== null) {
        const selectedExists = !!findFilteredPairById(state.selectedPairId, filtered);
        if (!selectedExists) {
            state.selectedPairId = null;
            state.openDocumentIdx = null;
        }
    }
    // Hide/show and update selection state
    allCards.forEach((card) => {
        const pairId = getPairIdFromCard(card);
        if (filteredSet.has(card)) {
            card.classList.remove('hidden');
        } else {
            card.classList.add('hidden');
        }
        if (state.selectedPairId === pairId) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
    // Reorder DOM using CSS order to match sort order
    filtered.forEach((card, index) => {
        card.style.order = index;
    });
    // Set high order for hidden cards so they appear last if somehow shown
    allCards.forEach((card) => {
        if (!filteredSet.has(card)) {
            card.style.order = 9999;
        }
    });
}

// Select a pair (or close open document if same pair clicked)
function selectPair(pairId) {
    const filtered = getFilteredPairs();
    const match = findFilteredPairById(pairId, filtered);
    if (!match) {
        console.warn(`Pair ${pairId} not available with current filters`);
        return;
    }
    // No-op if already viewing this pair with no document open
    if (state.selectedPairId === pairId && state.openDocumentIdx === null) {
        return;
    }
    // Check if this would return to the previous history state
    if (wouldReturnToPrevious(pairId, null)) {
        saveScrollAndGoBack();
        return;
    }
    state.selectedPairId = pairId;
    state.openDocumentIdx = null;
    updateURL(true);  // pushState for pair selection
    updatePairListDisplay();
    renderContent();
    renderReasoning();
}

// Render content area
function renderContent() {
    const content = document.getElementById('content');

    if (state.selectedPairId === null) {
        content.innerHTML = '<div class="content-placeholder">Select a pair to view documents</div>';
        return;
    }

    const filtered = getFilteredPairs();
    const match = findFilteredPairById(state.selectedPairId, filtered);
    if (!match) {
        content.innerHTML = '<div class="content-placeholder">Select a pair to view documents</div>';
        return;
    }

    const pairCard = match.card;
    const pairIdx = getPairIdFromCard(pairCard);
    const docGroups = JSON.parse(pairCard.dataset.docGroups || '[]');

    // Render one accordion per document (groups may consolidate multiple assessments)
    content.innerHTML = docGroups.map((group, idx) => {
        const {doc_idx: docIdx, total_quotes: quotes, assessment_count: count,
               relationships, polarity, overall, label} = group;

        const template = document.getElementById(`doc-template-${docIdx}`);
        if (!template) return '';

        // Extract title and date from template
        const temp = document.createElement('div');
        temp.appendChild(template.content.cloneNode(true));
        const titleEl = temp.querySelector('.document-title');
        const titleClone = titleEl?.cloneNode(true);
        titleClone?.querySelector('.document-date')?.remove();
        const title = titleClone?.textContent?.trim() || 'Untitled';
        const date = temp.querySelector('.document-date')?.textContent || '';

        // Build relationship/assessment display
        const relDisplay = count > 1
            ? `<span class="relationship-count">${count} assessments</span>`
            : relationships?.length > 0
                ? `<span class="relationship-label-small">${escapeHtml(relationships[0].replace(/_/g, ' '))}</span>`
                : '';

        // Build polarity/evidence badge (two-segment pill)
        const polarityMap = {positive: '+', negative: '-', neutral: 'N', irrelevant: 'I'};
        const relTooltip = relationships?.length > 0
            ? ` data-tooltip="${escapeHtml(relationships.map(r => r.replace(/_/g, ' ')).join(', '))}"`
            : '';
        const badge = polarity && overall
            ? `<span class="pc-pill">
                   <span class="${escapeHtml(polarity)}"${relTooltip}>${escapeHtml(polarityMap[polarity] || polarity)}</span>
                   <span class="evidence-${overall}">${escapeHtml(label || '')}</span>
               </span>`
            : overall ? `<span class="evidence-badge evidence-${overall}">${escapeHtml(label || '')}</span>`
            : polarity ? `<span class="polarity-badge polarity-${escapeHtml(polarity)}"${relTooltip}>${escapeHtml(polarityMap[polarity] || polarity)}</span>`
            : '';

        const isOpen = state.openDocumentIdx === idx;
        return `
        <div class="document-accordion" data-doc="${docIdx}">
            <div class="document-header ${isOpen ? 'open' : ''}" onclick="toggleDocument(${idx})">
                <div class="document-title">
                    <span>${escapeHtml(title)}</span>${date ? `<span class="document-date">${escapeHtml(date)}</span>` : ''}
                </div>
                <div class="document-stats">
                    <span>${quotes} quote${quotes !== 1 ? 's' : ''}</span>
                    ${relDisplay}
                    ${badge}
                </div>
            </div>
            <div class="document-content ${isOpen ? 'open' : ''}" id="doc-content-${idx}">
                ${isOpen ? renderDocument(docIdx, pairIdx) : ''}
            </div>
        </div>`;
    }).join('');
}

// Toggle document accordion
function toggleDocument(idx) {
    // Validate pair is still in filtered list (consistent with other functions)
    if (state.selectedPairId === null) return;
    const filtered = getFilteredPairs();
    const match = findFilteredPairById(state.selectedPairId, filtered);
    if (!match) {
        console.warn(`Pair ${state.selectedPairId} not available with current filters`);
        return;
    }
    // Capture the clicked header's viewport position before any changes
    const clickedHeader = document.querySelectorAll('.document-header')[idx];
    const headerTopBeforeToggle = clickedHeader ? clickedHeader.getBoundingClientRect().top : null;
    // Determine the new document index after toggle
    const newDocIdx = state.openDocumentIdx === idx ? null : idx;
    // Check if this would return to the previous history state
    if (wouldReturnToPrevious(state.selectedPairId, newDocIdx)) {
        saveScrollAndGoBack();
        return;
    }
    state.openDocumentIdx = newDocIdx;
    updateURL(true);  // pushState for document toggle
    renderContent();
    renderReasoning();
    // After rendering, adjust scroll to keep clicked header at same viewport position
    if (headerTopBeforeToggle !== null) {
        requestAnimationFrame(() => {
            // Query the newly rendered header element (old reference is stale)
            const newClickedHeader = document.querySelectorAll('.document-header')[idx];
            if (newClickedHeader) {
                const headerTopAfterToggle = newClickedHeader.getBoundingClientRect().top;
                const scrollAdjustment = headerTopAfterToggle - headerTopBeforeToggle;
                if (scrollAdjustment !== 0) {
                    const contentArea = document.getElementById('content');
                    contentArea.scrollTop += scrollAdjustment;
                }
            }
        });
    }
}

// Highlight quotes for the current pair
function highlightQuotesForAssessment(docIdx, pairIdx) {
    const doc = document.getElementById(`doc-${docIdx}`);
    if (!doc) return;

    const pairIdStr = String(pairIdx);

    // Reset all quotes
    doc.querySelectorAll('.quote-span').forEach(span => {
        span.classList.remove('quote-highlight', 'quote-dim', 'quote-blink');
    });

    // Highlight quotes for this pair
    const hasMatchingQuotes = doc.querySelector(`.quote-span[data-pairs~="${pairIdStr}"]`) !== null;

    doc.querySelectorAll('.quote-span').forEach(span => {
        const quotePairs = (span.dataset.pairs || '').split(' ').filter(p => p);
        if (quotePairs.includes(pairIdStr)) {
            span.classList.add('quote-highlight');
        } else if (hasMatchingQuotes && quotePairs.length > 0) {
            span.classList.add('quote-dim');
        }
    });
}

// Update highlighting classes on entity spans based on current pair selection
function updateDocumentHighlights(docIdx, currentPairIdx) {
    const doc = document.getElementById(`doc-${docIdx}`);
    if (!doc) return;

    const pairCard = document.getElementById(`pair-${currentPairIdx}`);
    if (!pairCard) return;

    const entity1Name = pairCard.dataset.e1;
    const entity2Name = pairCard.dataset.e2;

    // Update all entity spans
    // Default styling (no entity1/entity2 class) displays as "other"
    doc.querySelectorAll('.entity-span').forEach(span => {
        const entityName = span.dataset.entity;
        const entityPairs = (span.dataset.pairs || '').split(' ').filter(p => p);

        // Clear existing classification classes
        span.classList.remove('entity1', 'entity2');
        span.onclick = null;

        // Classify entity - only entity1/entity2 need explicit classes
        if (entityName === entity1Name) {
            span.classList.add('entity1');
        } else if (entityName === entity2Name) {
            span.classList.add('entity2');
        } else if (entityPairs.length > 0) {
            // Other entity with pair references - make clickable
            // Navigate to pair while keeping same document open
            span.onclick = () => {
                const targetPairIdx = parseInt(entityPairs[0]);
                const spanId = span.id;
                selectPairAndDocument(targetPairIdx, docIdx);
                // Scroll to the clicked entity after render
                if (spanId) {
                    setTimeout(() => {
                        const el = document.getElementById(spanId);
                        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 0);
                }
            };
        }
    });
}

// Render document with highlights
function renderDocument(docIdx, pairIdx) {
    const templateId = `doc-template-${docIdx}`;
    const template = document.getElementById(templateId);

    if (!template) {
        console.error(`Template not found: ${templateId}`);
        return '<p>Document template not found</p>';
    }

    // Clone the template content and extract just the document body
    const clone = template.content.cloneNode(true);
    const tempContainer = document.createElement('div');
    tempContainer.appendChild(clone);

    // Get the document body (which contains the links and text with quotes/entities)
    const docBody = tempContainer.querySelector('.document-body');
    const html = docBody ? docBody.innerHTML : '<p>Document content not found</p>';

    // After rendering, update the highlighting classes
    setTimeout(() => {
        updateDocumentHighlights(docIdx, pairIdx);
        highlightQuotesForAssessment(docIdx, pairIdx);
    }, 0);

    return html;
}

// Render reasoning sidebar
function renderReasoning(resetScroll = true) {
    const rightbar = document.getElementById('rightbar');

    if (state.selectedPairId === null) {
        rightbar.innerHTML = '<div class="reasoning-panel"><p>Select a pair to view reasoning</p></div>';
        return;
    }

    const filtered = getFilteredPairs();
    const match = findFilteredPairById(state.selectedPairId, filtered);
    if (!match) {
        rightbar.innerHTML = '<div class="reasoning-panel"><p>Select a pair to view reasoning</p></div>';
        return;
    }

    const pairCard = match.card;
    const pairIdx = getPairIdFromCard(pairCard);

    // Determine template ID based on state
    let templateId;
    if (state.openDocumentIdx === null) {
        templateId = `reasoning-pair-${pairIdx}-overall`;
    } else {
        // Get doc index from document groups
        const docGroups = JSON.parse(pairCard.dataset.docGroups || '[]');
        if (state.openDocumentIdx < docGroups.length) {
            const docIdx = docGroups[state.openDocumentIdx].doc_idx;
            templateId = `reasoning-pair-${pairIdx}-doc-${docIdx}`;
        } else {
            templateId = `reasoning-pair-${pairIdx}-overall`;
        }
    }

    const template = document.getElementById(templateId);
    if (!template) {
        console.error(`Template not found: ${templateId}`);
        rightbar.innerHTML = '<div class="reasoning-panel"><p>Error loading reasoning content</p></div>';
        return;
    }

    // Clone and insert template
    const clone = template.content.cloneNode(true);
    rightbar.innerHTML = '';
    rightbar.appendChild(clone);
    // Reset scroll to top when showing new content (unless restoring from history)
    if (resetScroll) {
        rightbar.scrollTop = 0;
    }
}

// Doc-link hover state
let docLinkHoverDoc = null;
let docLinkScrollTimer = null;
let docLinkLastScrollTime = 0;
// Set up delegated hover handlers for .doc-link elements (called once at init)
function initDocLinkHover() {
    const rightbar = document.getElementById('rightbar');
    const content = document.getElementById('content');
    if (!rightbar || !content) return;
    const scrollIntoViewIfNeeded = (accordion) => {
        const rect = accordion.getBoundingClientRect();
        const contentRect = content.getBoundingClientRect();
        if (rect.top < contentRect.top || rect.bottom > contentRect.bottom) {
            accordion.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        docLinkLastScrollTime = Date.now();
    };
    rightbar.addEventListener('mouseover', (e) => {
        const link = e.target.closest('.doc-link');
        const docIdx = link?.dataset.doc;
        if (docIdx === docLinkHoverDoc) return;
        // Clear previous
        if (docLinkHoverDoc) {
            document.querySelector(`.document-accordion[data-doc="${docLinkHoverDoc}"]`)
                ?.classList.remove('doc-link-hover');
        }
        clearTimeout(docLinkScrollTimer);
        docLinkHoverDoc = docIdx;
        if (!docIdx) return;
        const accordion = document.querySelector(`.document-accordion[data-doc="${docIdx}"]`);
        if (!accordion) return;
        accordion.classList.add('doc-link-hover');
        // Immediate scroll if recently scrolled, otherwise 1s delay
        if (Date.now() - docLinkLastScrollTime < 5000) {
            scrollIntoViewIfNeeded(accordion);
        } else {
            docLinkScrollTimer = setTimeout(() => scrollIntoViewIfNeeded(accordion), 1000);
        }
    });
    rightbar.addEventListener('mouseleave', () => {
        if (docLinkHoverDoc) {
            document.querySelector(`.document-accordion[data-doc="${docLinkHoverDoc}"]`)
                ?.classList.remove('doc-link-hover');
        }
        clearTimeout(docLinkScrollTimer);
        docLinkHoverDoc = null;
    });
}

// Scroll to quote with emphasis
function scrollToQuote(quoteId) {
    const selector = `.quote-span.${CSS.escape(quoteId)}`;
    const firstQuote = document.querySelector(selector);
    if (firstQuote) {
        firstQuote.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    document.querySelectorAll(selector).forEach(span => {
        span.classList.add('quote-blink');
        setTimeout(() => {
            span.classList.remove('quote-blink');
        }, 2000);
    });
}

// Open document by doc_idx within currently selected pair
function openDocument(docIdx) {
    if (state.selectedPairId === null) return;
    // Validate pair is in filtered list (consistent with other functions)
    const filtered = getFilteredPairs();
    const match = findFilteredPairById(state.selectedPairId, filtered);
    if (!match) {
        console.warn(`Pair ${state.selectedPairId} not available with current filters`);
        return;
    }
    const pairCard = match.card;
    const docIndices = pairCard.dataset.docs.trim().split(' ').map(n => parseInt(n));
    const assessIdx = docIndices.indexOf(docIdx);
    if (assessIdx === -1) {
        console.warn(`Document ${docIdx} not found in pair ${state.selectedPairId}`);
        return;
    }
    // Check if this would return to the previous history state
    if (wouldReturnToPrevious(state.selectedPairId, assessIdx)) {
        saveScrollAndGoBack();
        return;
    }
    state.openDocumentIdx = assessIdx;
    updateURL(true);  // pushState for document open
    renderContent();
    renderReasoning();
    // Scroll to the document accordion in the content panel
    setTimeout(() => {
        const doc = document.getElementById(`doc-${docIdx}`);
        const accordion = doc?.closest('.document-accordion');
        if (accordion) {
            accordion.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 0);
}

// Select pair and open specific document by doc_idx
function selectPairAndDocument(pairIdx, docIdx) {
    // Find pairIdx in filtered pairs
    const filtered = getFilteredPairs();
    const match = findFilteredPairById(pairIdx, filtered);
    if (!match) {
        console.warn(`Pair ${pairIdx} not in filtered list`);
        return;
    }
    const pairCard = match.card;
    const docIndices = pairCard.dataset.docs.trim().split(' ').map(n => parseInt(n));
    const assessIdx = docIndices.indexOf(docIdx);
    const newDocIdx = assessIdx !== -1 ? assessIdx : 0;
    // No-op if already at the target state
    if (state.selectedPairId === pairIdx && state.openDocumentIdx === newDocIdx) {
        return;
    }
    // Check if this would return to the previous history state
    if (wouldReturnToPrevious(pairIdx, newDocIdx)) {
        saveScrollAndGoBack();
        return;
    }
    state.selectedPairId = pairIdx;
    state.openDocumentIdx = newDocIdx;
    updateURL(true);  // pushState for pair+document selection
    updatePairListDisplay();
    match.card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    renderContent();
    renderReasoning();
}

// Utility: escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Column resizing
// ---------------------------------------------------------------------------

const COLUMN_WIDTH_STORAGE_KEY = 'if-report-column-widths';
const COLUMN_MIN_PX = 180;
const COLUMN_CONTENT_MIN_PX = 240;
const COLUMN_RESIZER_PX = 6;
const COLUMN_KEY_STEP_PX = 24;

function readStoredColumnWidths() {
    try {
        const raw = localStorage.getItem(COLUMN_WIDTH_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        const left = Number(parsed.left);
        const right = Number(parsed.right);
        if (!Number.isFinite(left) || !Number.isFinite(right)) return null;
        return { left, right };
    } catch (e) {
        return null;
    }
}

function writeStoredColumnWidths(left, right) {
    try {
        localStorage.setItem(
            COLUMN_WIDTH_STORAGE_KEY,
            JSON.stringify({ left, right })
        );
    } catch (e) {
        // Storage may be unavailable (private mode, quota); ignore.
    }
}

function applyColumnWidths(left, right) {
    const viewport = window.innerWidth;
    const reservedForContent = COLUMN_CONTENT_MIN_PX + 2 * COLUMN_RESIZER_PX;
    const maxSide = Math.max(COLUMN_MIN_PX, viewport - reservedForContent - COLUMN_MIN_PX);
    const clampedLeft = Math.min(maxSide, Math.max(COLUMN_MIN_PX, left));
    const clampedRight = Math.min(maxSide, Math.max(COLUMN_MIN_PX, right));
    document.body.style.gridTemplateColumns =
        `${clampedLeft}px ${COLUMN_RESIZER_PX}px 1fr ${COLUMN_RESIZER_PX}px ${clampedRight}px`;
    return { left: clampedLeft, right: clampedRight };
}

function getCurrentColumnWidths() {
    const sidebar = document.getElementById('sidebar');
    const rightbar = document.getElementById('rightbar');
    return {
        left: sidebar ? sidebar.getBoundingClientRect().width : COLUMN_MIN_PX,
        right: rightbar ? rightbar.getBoundingClientRect().width : COLUMN_MIN_PX,
    };
}

function initColumnResizers() {
    // Restore persisted widths (clamped to viewport).
    const stored = readStoredColumnWidths();
    if (stored) {
        applyColumnWidths(stored.left, stored.right);
    }

    const leftResizer = document.getElementById('resizer-left');
    const rightResizer = document.getElementById('resizer-right');
    if (!leftResizer || !rightResizer) return;

    function startDrag(side, startEvent) {
        const startX = startEvent.clientX;
        const initial = getCurrentColumnWidths();
        const resizer = side === 'left' ? leftResizer : rightResizer;
        resizer.classList.add('dragging');
        document.body.classList.add('col-resizing');

        function onMove(moveEvent) {
            const dx = moveEvent.clientX - startX;
            let left = initial.left;
            let right = initial.right;
            if (side === 'left') {
                left = initial.left + dx;
            } else {
                right = initial.right - dx;
            }
            applyColumnWidths(left, right);
        }

        function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            resizer.classList.remove('dragging');
            document.body.classList.remove('col-resizing');
            const final = getCurrentColumnWidths();
            writeStoredColumnWidths(final.left, final.right);
        }

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        startEvent.preventDefault();
    }

    leftResizer.addEventListener('mousedown', (e) => startDrag('left', e));
    rightResizer.addEventListener('mousedown', (e) => startDrag('right', e));

    // Double-click to reset to default width.
    leftResizer.addEventListener('dblclick', () => {
        const cur = getCurrentColumnWidths();
        const applied = applyColumnWidths(320, cur.right);
        writeStoredColumnWidths(applied.left, applied.right);
    });
    rightResizer.addEventListener('dblclick', () => {
        const cur = getCurrentColumnWidths();
        const applied = applyColumnWidths(cur.left, 320);
        writeStoredColumnWidths(applied.left, applied.right);
    });

    // Keyboard support for accessibility.
    function handleKey(side, event) {
        const isLeftArrow = event.key === 'ArrowLeft';
        const isRightArrow = event.key === 'ArrowRight';
        if (!isLeftArrow && !isRightArrow) return;
        event.preventDefault();
        const cur = getCurrentColumnWidths();
        const delta = (isRightArrow ? 1 : -1) * COLUMN_KEY_STEP_PX;
        let left = cur.left;
        let right = cur.right;
        if (side === 'left') {
            left = cur.left + delta;
        } else {
            right = cur.right - delta;
        }
        const applied = applyColumnWidths(left, right);
        writeStoredColumnWidths(applied.left, applied.right);
    }
    leftResizer.addEventListener('keydown', (e) => handleKey('left', e));
    rightResizer.addEventListener('keydown', (e) => handleKey('right', e));

    // Re-clamp on window resize so columns don't push content below its minimum.
    window.addEventListener('resize', () => {
        const cur = getCurrentColumnWidths();
        const applied = applyColumnWidths(cur.left, cur.right);
        if (applied.left !== cur.left || applied.right !== cur.right) {
            writeStoredColumnWidths(applied.left, applied.right);
        }
    });
}

// ---------------------------------------------------------------------------
// Export feature
// ---------------------------------------------------------------------------

const EXPORT_MODAL_ANIMATION_MS = 400;

// Column definitions. `id` is the JSON key and the fallback label; `label`
// is the checkbox label and CSV header (after kind-substitution for the
// entity columns). `jsonOnly` columns are disabled when CSV/TSV is selected
// and render in the second section. `default` controls the initial state.
// Array/object values are flattened with `|` for delimited formats and
// preserved as-is in JSON.
const EXPORT_PAIR_COLUMNS = [
    { id: 'entity1', label: 'Entity 1', default: true, get: (p) => p.e1 },
    { id: 'entity1_kind', label: 'Entity 1 kind',
      get: (p) => p.e1Kind, onlyWhenMixedSide: 'e1' },
    { id: 'entity2', label: 'Entity 2', default: true, get: (p) => p.e2 },
    { id: 'entity2_kind', label: 'Entity 2 kind',
      get: (p) => p.e2Kind, onlyWhenMixedSide: 'e2' },
    { id: 'relationship', label: 'Relationship', default: true, get: (p) => p.rel },
    { id: 'evidence', label: 'Evidence', get: (p) => p.overall },
    { id: 'evidence_label', label: 'Evidence label', get: (p) => p.evidenceLabel },
    { id: 'topic_relevance', label: 'Topic relevance', default: true, get: (p) => p.relevance },
    { id: 'on_topic', label: 'On-topic',
      get: (p) => ({ pass: 'yes', fail: 'no', na: '' }[p.subjectTrust] ?? '') },
    { id: 'doc_count', label: 'Document count', default: true, get: (p) => p.docCount },
    { id: 'quote_count', label: 'Quote count', default: true, get: (p) => p.quoteCount },
    { id: 'polarity_counts', label: 'Polarity counts', get: (p) => p.polarityCounts },
    { id: 'document_urls', label: 'Document URLs',
      get: (p) => p.docs.map((d) => d.url).filter(Boolean) },
    { id: 'document_dois', label: 'Document DOIs',
      get: (p) => p.docs.map((d) => d.doi).filter(Boolean) },
    { id: 'entity1_aliases', label: 'Entity 1 aliases', jsonOnly: true, default: true,
      get: (p) => p.e1Aliases, onlyWhenAliasesPresent: 'e1' },
    { id: 'entity2_aliases', label: 'Entity 2 aliases', jsonOnly: true, default: true,
      get: (p) => p.e2Aliases, onlyWhenAliasesPresent: 'e2' },
];

// Cached map of doc_idx -> { title, url, doi }. Populated lazily on first
// export by parsing the document templates.
let exportDocLinksCache = null;

function buildDocLinksCache() {
    if (exportDocLinksCache) return exportDocLinksCache;
    exportDocLinksCache = {};
    document.querySelectorAll('#document-templates template').forEach((tpl) => {
        const idMatch = tpl.id.match(/^doc-template-(\\d+)$/);
        if (!idMatch) return;
        const docIdx = Number(idMatch[1]);
        const content = tpl.content;
        const titleEl = content.querySelector('.document-title')?.cloneNode(true);
        titleEl?.querySelector('.document-date')?.remove();
        const title = titleEl?.textContent.trim() || '';
        const url = content.querySelector('a.document-url-badge')?.getAttribute('href') || '';
        // DOI lives in the link's href as "https://doi.org/{doi}" — parsing
        // the href avoids whitespace artefacts from the rendered text content.
        const doiHref = content.querySelector('a.document-doi-link')?.getAttribute('href') || '';
        const doi = doiHref.replace(/^https?:\\/\\/doi\\.org\\//, '');
        exportDocLinksCache[docIdx] = { title, url, doi };
    });
    return exportDocLinksCache;
}

function readDocGroups(card) {
    try {
        return JSON.parse(card.dataset.docGroups || '[]');
    } catch (e) {
        return [];
    }
}

// Return the unique supporting-document records for this card, in display order.
function getDocsForCard(card) {
    const byIdx = buildDocLinksCache();
    const docs = [];
    const seen = new Set();
    for (const group of readDocGroups(card)) {
        if (seen.has(group.doc_idx)) continue;
        seen.add(group.doc_idx);
        const entry = byIdx[group.doc_idx];
        if (entry) docs.push(entry);
    }
    return docs;
}

function formatKindForHeader(kind) {
    return kind ? kind.charAt(0).toUpperCase() + kind.slice(1) : '';
}

function formatPolarityCounts(card) {
    const counts = { positive: 0, negative: 0, neutral: 0, irrelevant: 0 };
    for (const g of readDocGroups(card)) {
        if (g.polarity in counts) counts[g.polarity] += g.assessment_count || 1;
    }
    return counts;
}

// Read a normalised pair record from a card. Aliases come back as arrays.
function readPairFromCard(card) {
    const aliasList = (s) => (s || '').split(',').filter((x) => x);
    return {
        e1: card.dataset.e1 || '',
        e1Kind: card.querySelector('.pair-entities > span:first-child')?.dataset.kind || '',
        e1Aliases: aliasList(card.dataset.e1a),
        e2: card.dataset.e2 || '',
        e2Kind: card.querySelector('.pair-entities > span:last-child')?.dataset.kind || '',
        e2Aliases: aliasList(card.dataset.e2a),
        rel: card.dataset.rel || '',
        overall: Number(card.dataset.overall) || 0,
        evidenceLabel: card.querySelector('.evidence-badge')?.textContent.trim() || '',
        relevance: Number(card.dataset.relevance) || 0,
        subjectTrust: card.dataset.subjectTrust || '',
        docCount: Number(card.dataset.docCount) || 0,
        quoteCount: Number(card.dataset.quoteCount) || 0,
        polarityCounts: formatPolarityCounts(card),
        docs: getDocsForCard(card),
    };
}

// Inspect the cards currently being exported and decide which columns are
// applicable: entity-kind columns appear only when that side has >1 kind;
// alias columns appear only when at least one card on that side has aliases.
function buildExportContext(cards) {
    const e1Kinds = new Set();
    const e2Kinds = new Set();
    let e1HasAliases = false;
    let e2HasAliases = false;
    for (const card of cards) {
        const span1 = card.querySelector('.pair-entities > span:first-child');
        const span2 = card.querySelector('.pair-entities > span:last-child');
        if (span1?.dataset.kind) e1Kinds.add(span1.dataset.kind);
        if (span2?.dataset.kind) e2Kinds.add(span2.dataset.kind);
        if ((card.dataset.e1a || '').length > 0) e1HasAliases = true;
        if ((card.dataset.e2a || '').length > 0) e2HasAliases = true;
    }
    const e1Mixed = e1Kinds.size > 1;
    const e2Mixed = e2Kinds.size > 1;
    return {
        e1Mixed,
        e2Mixed,
        e1SoleKind: e1Mixed ? null : [...e1Kinds][0] || null,
        e2SoleKind: e2Mixed ? null : [...e2Kinds][0] || null,
        e1HasAliases,
        e2HasAliases,
        allKinds: new Set([...e1Kinds, ...e2Kinds]),
    };
}

// Resolve a column header label given the export context (substituting kind
// names for "Entity 1/2" when only one kind is present on that side).
function columnHeaderLabel(col, ctx) {
    if (col.id === 'entity1' && ctx.e1SoleKind) return formatKindForHeader(ctx.e1SoleKind);
    if (col.id === 'entity2' && ctx.e2SoleKind) return formatKindForHeader(ctx.e2SoleKind);
    if (col.id === 'entity1_aliases' && ctx.e1SoleKind)
        return `${formatKindForHeader(ctx.e1SoleKind)} aliases`;
    if (col.id === 'entity2_aliases' && ctx.e2SoleKind)
        return `${formatKindForHeader(ctx.e2SoleKind)} aliases`;
    return col.label;
}

// Return columns applicable to the current report (after side-mixing filter).
function applicableColumns(ctx) {
    return EXPORT_PAIR_COLUMNS.filter((col) => {
        if (col.onlyWhenMixedSide === 'e1' && !ctx.e1Mixed) return false;
        if (col.onlyWhenMixedSide === 'e2' && !ctx.e2Mixed) return false;
        if (col.onlyWhenAliasesPresent === 'e1' && !ctx.e1HasAliases) return false;
        if (col.onlyWhenAliasesPresent === 'e2' && !ctx.e2HasAliases) return false;
        return true;
    });
}

function getExportCards() {
    const applyFilters = document.getElementById('export-apply-filters').checked;
    if (applyFilters) {
        return getFilteredPairs();  // already an array of cards
    }
    return Array.from(document.querySelectorAll('#sidebar .pair-card'));
}

// Mode selection. Radio values are 'pairs' or 'entities:<kind>'.
function getExportModeValue() {
    const checked = document.querySelector('input[name="export-mode"]:checked');
    return checked ? checked.value : 'pairs';
}

function getExportMode() {
    return getExportModeValue().startsWith('entities:') ? 'entities' : 'pairs';
}

function getSelectedEntityKind() {
    const value = getExportModeValue();
    return value.startsWith('entities:') ? value.slice('entities:'.length) : null;
}

function getExportFormat() {
    const checked = document.querySelector('input[name="export-format"]:checked');
    return checked ? checked.value : 'csv';
}

// Rebuild the column checkbox lists based on the current export context.
// Preserves user check state for columns that still exist.
function rebuildColumnControls(ctx) {
    const standardGrid = document.querySelector('.export-col-grid[data-section="standard"]');
    const jsonGrid = document.querySelector('.export-col-grid[data-section="json"]');
    const existing = {};
    document.querySelectorAll('input[name="export-col"]').forEach((cb) => {
        existing[cb.value] = cb.checked;
    });
    standardGrid.innerHTML = '';
    jsonGrid.innerHTML = '';
    for (const col of applicableColumns(ctx)) {
        const checked = col.id in existing ? existing[col.id] : !!col.default;
        const label = document.createElement('label');
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.name = 'export-col';
        input.value = col.id;
        if (checked) input.checked = true;
        label.appendChild(input);
        label.appendChild(document.createTextNode(' ' + columnHeaderLabel(col, ctx)));
        (col.jsonOnly ? jsonGrid : standardGrid).appendChild(label);
    }
    const hint = document.querySelector('.export-json-hint');
    const showJson = jsonGrid.children.length > 0;
    jsonGrid.style.display = showJson ? '' : 'none';
    hint.style.display = showJson ? '' : 'none';
}

function updateColumnsSectionVisibility() {
    const isEntities = getExportMode() === 'entities';
    document.getElementById('export-columns-pairs').style.display = isEntities ? 'none' : '';
    document.getElementById('export-columns-entities').style.display = isEntities ? '' : 'none';
}

function updateJsonOnlyAvailability() {
    const isJson = getExportFormat() === 'json';
    const jsonGrid = document.querySelector('.export-col-grid[data-section="json"]');
    jsonGrid.dataset.disabled = isJson ? 'false' : 'true';
    jsonGrid.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
        cb.disabled = !isJson;
    });
}

// Render the mode radios as a flat list: 'Pairs' + one entry per detected
// kind. Preserves selection across rebuilds when possible.
function rebuildModeControls(ctx) {
    const fieldset = document.getElementById('export-mode-fieldset');
    const previous = getExportModeValue();
    fieldset.querySelectorAll('input, label').forEach((el) => el.remove());
    const kinds = [...ctx.allKinds].sort();
    const values = ['pairs', ...kinds.map((k) => `entities:${k}`)];
    const selected = values.includes(previous) ? previous : 'pairs';
    for (const value of values) {
        const id = `export-mode-${value.replace(/[^a-z0-9-]/gi, '-')}`;
        const input = document.createElement('input');
        input.type = 'radio';
        input.name = 'export-mode';
        input.id = id;
        input.value = value;
        if (value === selected) input.checked = true;
        const label = document.createElement('label');
        label.htmlFor = id;
        label.textContent = value === 'pairs'
            ? 'Pairs'
            : `${formatKindForHeader(value.slice('entities:'.length))}s`;
        fieldset.appendChild(input);
        fieldset.appendChild(label);
    }
}


// Are any filters narrowing the visible set (excluding sort order)? Unlike
// hasNonDefaultFilters (which tracks non-default *configuration*), this asks
// whether the export set is actually being reduced -- so the default-on
// on-topic-only switch counts as narrowing whenever it is engaged.
function hasActiveExportFilters() {
    return (
        state.searchQuery.length > 0 ||
        state.evidenceMin > 0 ||
        state.evidenceMax < 10 ||
        state.relevanceMin > 0 ||
        state.relevanceMax < 6 ||
        state.showRejected ||
        state.showContentious ||
        state.strictOnTopic
    );
}

// Render a min/max selection as "≥ N", "≤ N", or "N–M" (any-bound omitted).
function describeRange(label, min, max, minAny, maxAny) {
    const hasMin = min > minAny;
    const hasMax = max < maxAny;
    if (hasMin && hasMax) return `${label} ${min}–${max}`;
    if (hasMin) return `${label} ≥ ${min}`;
    if (hasMax) return `${label} ≤ ${max}`;
    return null;
}

// Human-readable phrases for each filter currently narrowing the export set,
// in the order they appear in the filter panel. Empty when nothing is active.
function describeActiveExportFilters() {
    const parts = [];
    if (state.strictOnTopic) parts.push('on topic');
    if (state.searchQuery.length > 0) parts.push(`“${state.searchQuery}”`);
    const evidence = describeRange('evidence', state.evidenceMin, state.evidenceMax, 0, 10);
    if (evidence) parts.push(evidence);
    const relevance = describeRange('relevance', state.relevanceMin, state.relevanceMax, 0, 6);
    if (relevance) parts.push(relevance);
    if (state.showContentious) parts.push('contentious');
    if (state.showRejected) parts.push('including rejected');
    return parts;
}

function updateScopeCount() {
    const fieldset = document.getElementById('export-filters-fieldset');
    if (!hasActiveExportFilters()) {
        // No filters narrow the set, so the toggle is meaningless: hide it.
        // Leave `checked = true` so the export pulls from getFilteredPairs(),
        // which returns the same full set when no filters are active.
        fieldset.style.display = 'none';
        document.getElementById('export-apply-filters').checked = true;
        return;
    }
    fieldset.style.display = '';
    const summary = describeActiveExportFilters().join(', ');
    document.getElementById('export-filters-summary').textContent =
        summary ? `(${summary})` : '';
}

// Build the array of row objects to be exported, given the current settings.
function buildExportRows(cards, ctx) {
    return getExportMode() === 'entities' ? buildEntityRows(cards) : buildPairRows(cards, ctx);
}

function selectedColumnIds() {
    const ids = [];
    document.querySelectorAll('input[name="export-col"]:checked').forEach((cb) => {
        if (cb.disabled) return;
        ids.push(cb.value);
    });
    return ids;
}

// JSON-only attribute IDs that get hoisted to top-level keys in the JSON
// payload rather than appearing on individual rows. Rows reference the
// hoisted data instead (documents by index, aliases by entity name).
const HOISTED_DOC_COLS = new Set(['document_urls', 'document_dois']);
const HOISTED_ALIAS_COLS = new Set(['entity1_aliases', 'entity2_aliases']);

function buildPairRows(cards, ctx) {
    const selectedIds = new Set(selectedColumnIds());
    const activeCols = applicableColumns(ctx).filter((c) => selectedIds.has(c.id));
    const headers = activeCols.map((c) => columnHeaderLabel(c, ctx));
    // For JSON: top-level `documents` array (deduplicated) referenced by
    // index from each row, and a top-level `aliases` map keyed by entity name.
    const wantDocs = selectedIds.has('document_urls') || selectedIds.has('document_dois');
    const includeUrl = selectedIds.has('document_urls');
    const includeDoi = selectedIds.has('document_dois');
    const wantE1Aliases = selectedIds.has('entity1_aliases');
    const wantE2Aliases = selectedIds.has('entity2_aliases');
    const docIndex = new Map();
    const documents = [];
    function indexDoc(d) {
        const key = `${d.title}\\u0000${d.url}\\u0000${d.doi}`;
        if (docIndex.has(key)) return docIndex.get(key);
        const entry = { title: d.title || null };
        if (includeUrl) entry.url = d.url || null;
        if (includeDoi) entry.doi = d.doi || null;
        documents.push(entry);
        docIndex.set(key, documents.length - 1);
        return documents.length - 1;
    }
    const aliases = {};
    const rows = [];
    const jsonRows = [];
    for (const card of cards) {
        const pair = readPairFromCard(card);
        const row = {};
        const jsonRow = {};
        for (const col of activeCols) {
            const label = columnHeaderLabel(col, ctx);
            const value = col.get(pair);
            row[label] = value;
            // Hoisted attributes are omitted from the JSON row and emitted at
            // the top level instead.
            if (!HOISTED_DOC_COLS.has(col.id) && !HOISTED_ALIAS_COLS.has(col.id)) {
                jsonRow[label] = value;
            }
        }
        if (wantDocs) jsonRow.documents = pair.docs.map(indexDoc);
        if (wantE1Aliases && pair.e1Aliases.length) aliases[pair.e1] = pair.e1Aliases;
        if (wantE2Aliases && pair.e2Aliases.length) aliases[pair.e2] = pair.e2Aliases;
        rows.push(row);
        jsonRows.push(jsonRow);
    }
    return { headers, rows, sidecar: { jsonRows, documents, aliases } };
}

// Order entity records per the entity-sort select. `default` preserves
// first-encountered order (the Map's insertion order, i.e. reading down the
// association list); `occurrences` is descending count, name-breaking ties.
function sortEntityRecords(records, mode) {
    if (mode === 'alphabetic') {
        return [...records].sort((a, b) => a.name.localeCompare(b.name));
    }
    if (mode === 'occurrences') {
        return [...records].sort(
            (a, b) => b.occurrences - a.occurrences || a.name.localeCompare(b.name)
        );
    }
    return [...records];  // default: first-seen order, already insertion-ordered
}

function buildEntityRows(cards) {
    const selectedKind = getSelectedEntityKind();
    if (!selectedKind) return { headers: [], rows: [], sidecar: null };
    const includeAliases = document.getElementById('export-entity-include-aliases')?.checked;
    const includeOccurrences = document.getElementById('export-entity-include-occurrences')?.checked;
    const sortMode = document.getElementById('export-entity-sort')?.value || 'default';
    // First-seen order with a per-entity occurrence count (number of cards the
    // entity appears in). Map insertion order == reading down the pair list.
    const seen = new Map();  // canonical name -> { name, aliases, occurrences }
    for (const card of cards) {
        const span1 = card.querySelector('.pair-entities > span:first-child');
        const span2 = card.querySelector('.pair-entities > span:last-child');
        const candidates = [
            { name: card.dataset.e1, kind: span1?.dataset.kind || '',
              aliases: (card.dataset.e1a || '').split(',').filter((s) => s) },
            { name: card.dataset.e2, kind: span2?.dataset.kind || '',
              aliases: (card.dataset.e2a || '').split(',').filter((s) => s) },
        ];
        // Count each entity at most once per card, even if it is both sides.
        const countedInCard = new Set();
        for (const c of candidates) {
            if (c.kind !== selectedKind || !c.name) continue;
            const record = seen.get(c.name)
                || (seen.set(c.name, { name: c.name, aliases: c.aliases, occurrences: 0 }),
                    seen.get(c.name));
            if (!countedInCard.has(c.name)) {
                record.occurrences += 1;
                countedInCard.add(c.name);
            }
        }
    }
    const kindLabel = formatKindForHeader(selectedKind);
    const ordered = sortEntityRecords([...seen.values()], sortMode);
    // Delimited rows: name, optional occurrence count, optional pipe-joined aliases.
    const headers = [kindLabel];
    if (includeOccurrences) headers.push('Occurrences');
    if (includeAliases) headers.push('Aliases');
    const rows = ordered.map((e) => {
        const row = { [kindLabel]: e.name };
        if (includeOccurrences) row.Occurrences = e.occurrences;
        if (includeAliases) row.Aliases = e.aliases;
        return row;
    });
    // JSON sidecar: top-level aliases map (only when requested). Rows are bare
    // names, or {name, occurrences} objects when the count column is on.
    const aliases = {};
    if (includeAliases) {
        for (const e of ordered) {
            if (e.aliases.length) aliases[e.name] = e.aliases;
        }
    }
    const jsonRows = ordered.map((e) =>
        includeOccurrences ? { name: e.name, occurrences: e.occurrences } : e.name);
    return { headers, rows, sidecar: { jsonRows, documents: [], aliases } };
}

// CSV / TSV serialisation.
function csvEscape(value, sep) {
    const str = value === null || value === undefined ? '' : String(value);
    if (str.includes(sep) || str.includes('"') || str.includes('\\n') || str.includes('\\r')) {
        return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
}

const POLARITY_SHORT = { positive: '+', negative: '-', neutral: 'N', irrelevant: 'I' };

function flattenCellForCsv(value) {
    if (value == null) return '';
    if (Array.isArray(value)) return value.join('|');
    if (typeof value === 'object') {
        return Object.entries(value)
            .filter(([, v]) => v > 0)
            .map(([k, v]) => `${POLARITY_SHORT[k] || k}:${v}`)
            .join('|');
    }
    return value;
}

function formatRowsAsDelimited({ headers, rows }, sep) {
    const out = [headers.map((h) => csvEscape(h, sep)).join(sep)];
    for (const row of rows) {
        const line = headers.map((h) => csvEscape(flattenCellForCsv(row[h]), sep)).join(sep);
        out.push(line);
    }
    return out.join('\\n');
}

function formatRowsAsJson({ sidecar }) {
    const jsonRows = sidecar?.jsonRows || [];
    const payload = {
        topic: document.body.dataset.topic || '',
        exported_at: new Date().toISOString(),
        row_count: jsonRows.length,
        filtered: document.getElementById('export-apply-filters').checked,
    };
    if (sidecar?.documents?.length) payload.documents = sidecar.documents;
    if (sidecar && Object.keys(sidecar.aliases || {}).length) payload.aliases = sidecar.aliases;
    payload.rows = jsonRows;
    return JSON.stringify(payload, null, 2);
}

function serialiseExport(data, format) {
    if (format === 'csv') return formatRowsAsDelimited(data, ',');
    if (format === 'tsv') return formatRowsAsDelimited(data, '\\t');
    if (format === 'json') return formatRowsAsJson(data);
    return '';
}

function renderExportPreview() {
    const cards = getExportCards();
    const ctx = buildExportContext(cards);
    // Rebuild controls that depend on context (re-render is idempotent).
    rebuildModeControls(ctx);
    rebuildColumnControls(ctx);
    updateJsonOnlyAvailability();
    updateColumnsSectionVisibility();
    updateScopeCount();
    const data = buildExportRows(cards, ctx);
    const table = document.getElementById('export-preview-table');
    // Render every row -- the preview wrap scrolls (max-height + sticky header),
    // so the table mirrors exactly what will be exported, not a truncated sample.
    let html = '<thead><tr>';
    for (const h of data.headers) {
        html += `<th scope="col">${escapeHtml(h)}</th>`;
    }
    html += '</tr></thead><tbody>';
    if (!data.rows.length) {
        html += `<tr><td colspan="${Math.max(1, data.headers.length)}" style="text-align:center; color:var(--pico-muted-color);">No rows match the current settings</td></tr>`;
    } else {
        for (const row of data.rows) {
            html += '<tr>';
            for (const h of data.headers) {
                const cell = flattenCellForCsv(row[h]);
                html += `<td>${escapeHtml(String(cell))}</td>`;
            }
            html += '</tr>';
        }
    }
    html += '</tbody>';
    table.innerHTML = html;
    const rowCount = document.getElementById('export-row-count');
    rowCount.textContent = `Will export ${data.rows.length} row${data.rows.length === 1 ? '' : 's'}.`;
}

function slugifyTopic(topic) {
    return (topic || 'extraction')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 60) || 'extraction';
}

function exportFilename() {
    const topic = slugifyTopic(document.body.dataset.topic || '');
    const mode = getExportMode();
    const format = getExportFormat();
    const date = new Date().toISOString().slice(0, 10);
    let suffix = 'pairs';
    if (mode === 'entities') {
        const kind = getSelectedEntityKind() || 'entities';
        suffix = `${kind}s`;
    }
    return `${topic}-${suffix}-${date}.${format}`;
}

let exportToastTimer = null;
function showExportToast(message) {
    let toast = document.querySelector('.export-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'export-toast';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    requestAnimationFrame(() => toast.classList.add('visible'));
    clearTimeout(exportToastTimer);
    exportToastTimer = setTimeout(() => toast.classList.remove('visible'), 1800);
}

function getExportPayload() {
    const cards = getExportCards();
    const ctx = buildExportContext(cards);
    const format = getExportFormat();
    const data = buildExportRows(cards, ctx);
    return { text: serialiseExport(data, format), format };
}

function downloadExport() {
    const { text, format } = getExportPayload();
    const mime = format === 'json' ? 'application/json' :
                 format === 'tsv' ? 'text/tab-separated-values' : 'text/csv';
    const blob = new Blob([text], { type: `${mime};charset=utf-8` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = exportFilename();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showExportToast(`Downloaded ${a.download}`);
}

async function copyExportToClipboard() {
    const { text } = getExportPayload();
    try {
        await navigator.clipboard.writeText(text);
        showExportToast('Copied to clipboard');
    } catch (e) {
        // Fallback for older browsers.
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try {
            document.execCommand('copy');
            showExportToast('Copied to clipboard');
        } catch (e2) {
            showExportToast('Copy failed; please download instead');
        }
        document.body.removeChild(ta);
    }
}

// Pico modal lifecycle, adapted from picocss.com's reference implementation.
function openExportDialog() {
    const dialog = document.getElementById('export-dialog');
    const html = document.documentElement;
    const scrollbarWidth = window.innerWidth - html.clientWidth;
    if (scrollbarWidth) {
        html.style.setProperty('--pico-scrollbar-width', `${scrollbarWidth}px`);
    }
    html.classList.add('modal-is-open', 'modal-is-opening');
    setTimeout(() => html.classList.remove('modal-is-opening'), EXPORT_MODAL_ANIMATION_MS);
    dialog.showModal();
    renderExportPreview();
}

function closeExportDialog() {
    const dialog = document.getElementById('export-dialog');
    if (!dialog.open) return;
    const html = document.documentElement;
    html.classList.add('modal-is-closing');
    setTimeout(() => {
        html.classList.remove('modal-is-closing', 'modal-is-open');
        html.style.removeProperty('--pico-scrollbar-width');
        dialog.close();
    }, EXPORT_MODAL_ANIMATION_MS);
}

function initExportDialog() {
    const toggle = document.getElementById('export-toggle');
    const dialog = document.getElementById('export-dialog');
    if (!toggle || !dialog) return;
    toggle.addEventListener('click', openExportDialog);
    dialog.querySelector('button[rel="prev"]').addEventListener('click', closeExportDialog);
    // Backdrop click: <dialog> exposes its backdrop as itself; clicks on the
    // <article> child propagate with target=article, so check the target.
    dialog.addEventListener('click', (e) => {
        if (e.target === dialog) closeExportDialog();
    });
    // ESC: native <dialog> closes itself but bypasses our animation/cleanup.
    dialog.addEventListener('cancel', (e) => {
        e.preventDefault();
        closeExportDialog();
    });
    // Live updates on every form change.
    dialog.addEventListener('change', renderExportPreview);
    document.getElementById('export-download-btn').addEventListener('click', downloadExport);
    document.getElementById('export-copy-btn').addEventListener('click', copyExportToClipboard);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initReport);
"""


def get_css() -> str:
    """Return the CSS stylesheet for reports."""
    return REPORT_CSS


def get_js() -> str:
    """Return the JavaScript code for reports."""
    return REPORT_JS
