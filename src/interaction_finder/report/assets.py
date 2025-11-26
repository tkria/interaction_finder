"""CSS and JavaScript assets for report generation.

Contains inline CSS and JS code as Python strings for embedding
in self-contained HTML reports.
"""

REPORT_CSS = """
/* Report-specific styling using Pico CSS as base */

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

    /* Confidence badge colors */
    --confidence-high-bg: var(--pico-color-green-600);
    --confidence-medium-bg: var(--pico-color-pumpkin-400);
    --confidence-low-bg: var(--pico-color-red-600);

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
    --polarity-supporting-bg: var(--pico-color-green-100);
    --polarity-supporting-text: var(--pico-color-green-700);
    --polarity-refuting-bg: var(--pico-color-red-100);
    --polarity-refuting-text: var(--pico-color-red-700);
    --polarity-neutral-bg: var(--pico-color-zinc-100);
    --polarity-neutral-text: var(--pico-color-slate-600);
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
    --confidence-high-bg: var(--pico-color-green-400);
    --confidence-medium-bg: var(--pico-color-pumpkin-600);
    --confidence-low-bg: var(--pico-color-red-400);
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
    --polarity-supporting-bg: var(--pico-color-green-850);
    --polarity-supporting-text: var(--pico-color-green-300);
    --polarity-refuting-bg: var(--pico-color-red-900);
    --polarity-refuting-text: var(--pico-color-red-300);
    --polarity-neutral-bg: var(--pico-color-slate-900);
    --polarity-neutral-text: var(--pico-color-slate-200);
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
    --confidence-high-bg: var(--pico-color-green-400);
    --confidence-medium-bg: var(--pico-color-pumpkin-600);
    --confidence-low-bg: var(--pico-color-red-400);
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
    --polarity-supporting-bg: var(--pico-color-green-850);
    --polarity-supporting-text: var(--pico-color-green-300);
    --polarity-refuting-bg: var(--pico-color-red-900);
    --polarity-refuting-text: var(--pico-color-red-300);
    --polarity-neutral-bg: var(--pico-color-slate-900);
    --polarity-neutral-text: var(--pico-color-slate-200);
}

body {
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-areas:
        "header header header"
        "sidebar content rightbar";
    grid-template-columns: 320px 1fr 320px;
    grid-template-rows: auto 1fr;
    height: 100vh;
    overflow: hidden;
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
    border-right: 1px solid var(--pico-muted-border-color);
    background: var(--pico-background-color);
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
    border-color: var(--pico-color-amber-500);
    box-shadow: 0 0 0 1px var(--pico-color-amber-200) inset;
}

.pair-entities {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
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
    text-decoration: underline;
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
}

.pair-relationship-only {
    text-align: center;
    margin-bottom: var(--spacing-compact);
}

.pair-evidence {
    margin-top: 0.35rem;
    display: flex;
    gap: 0.35rem;
    flex-wrap: wrap;
}

.polarity-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.polarity-badge.polarity-supporting {
    background: var(--polarity-supporting-bg);
    color: var(--polarity-supporting-text);
}

.polarity-badge.polarity-refuting {
    background: var(--polarity-refuting-bg);
    color: var(--polarity-refuting-text);
}

.polarity-badge.polarity-neutral {
    background: var(--polarity-neutral-bg);
    color: var(--polarity-neutral-text);
}

.pc-chip {
    display: inline-flex;
    border-radius: 1.2em;
    background: transparent;
    gap: 0.2em;
    text-transform: uppercase;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.pc-chip > span {
    padding: 0.1rem 0.45rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.2rem;
    border-radius: 1.2em;
}

.pc-chip > span:last-child {
    padding-left: 0;
}

.pc-chip .pc-count {
    font-size: 0.6rem;
    opacity: 0.8;
}

.pc-chip.polarity-supporting > span:first-child {
    background: var(--polarity-supporting-bg);
    color: var(--polarity-supporting-text);
    margin-left: -1px;
}

.pc-chip.polarity-refuting > span:first-child {
    background: var(--polarity-refuting-bg);
    color: var(--polarity-refuting-text);
    margin-left: -1px;
}

.pc-chip.polarity-neutral > span:first-child,
.pc-chip.polarity-irrelevant > span:first-child {
    background: var(--polarity-neutral-bg);
    color: var(--polarity-neutral-text);
    margin-left: -1px;
}

.pc-chip.confidence-high > span:last-child {
    background: var(--confidence-high-bg);
    color: white;
}

.pc-chip.confidence-medium > span:last-child {
    background: var(--confidence-medium-bg);
    color: white;
}

.pc-chip.confidence-low > span:last-child {
    background: var(--confidence-low-bg);
    color: white;
}

.pair-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
}

.pair-counts {
    color: var(--pico-muted-color);
}

[class^="confidence-"] {
    padding: 0.15rem 0.5rem;
    border-radius: var(--pico-border-radius);
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
}

.confidence-high {
    background: var(--confidence-high-bg);
    color: white;
}

.confidence-medium {
    background: var(--confidence-medium-bg);
    color: white;
}

.confidence-low {
    background: var(--confidence-low-bg);
    color: white;
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
    align-items: baseline;
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

.quote-highlight {
    background: var(--quote-bg);
    padding: 0.25rem 0;
    border-left: 4px solid var(--quote-border);
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
    transition: background 0.5s ease-out;
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
    border-left: 1px solid var(--pico-muted-border-color);
    background: var(--pico-background-color);
}

.reasoning-panel {
    font-size: 0.9rem;
}

.pair-header {
    padding: var(--spacing-card);
    margin-bottom: var(--spacing-card);
    border: 1px solid var(--pico-muted-border-color);
    border-radius: var(--pico-border-radius);
    background: var(--pico-card-background-color);
}

.reasoning-title {
    font-weight: 600;
    margin-bottom: var(--spacing-compact);
    color: var(--pico-primary);
}

.reasoning-subtitle {
    font-size: 0.85rem;
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
    justify-content: space-between;
    align-items: center;
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

.quote-number {
    flex-shrink: 0;
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
}

.quote-preview {
    flex: 1;
    line-height: 1.4;
    color: var(--pico-color);
    font-size: 0.8rem;
}

/* Utility classes */
.hidden {
    display: none !important;
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
};

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

// Initialize report
function initReport() {
    // Set up event listeners
    document.getElementById('search-input').addEventListener('input', handleSearch);
    document.getElementById('show-rejected').addEventListener('change', handleToggleRejected);

    // Add click handlers to pre-rendered pair cards
    const sidebar = document.getElementById('sidebar');
    sidebar.querySelectorAll('.pair-card').forEach((card) => {
        const pairId = getPairIdFromCard(card);
        card.addEventListener('click', () => selectPair(pairId));
    });

    // Initial render
    updateHeaderCounts();
    updatePairListDisplay();
    renderContent();
    renderReasoning();
}

// Search handler
function handleSearch(e) {
    state.searchQuery = e.target.value.toLowerCase();
    updateHeaderCounts();
    updatePairListDisplay();
    renderContent();
    renderReasoning();
}

// Toggle rejected pairs
function handleToggleRejected(e) {
    state.showRejected = e.target.checked;
    updateHeaderCounts();
    updatePairListDisplay();
    renderContent();
    renderReasoning();
}

// Update header counts based on current filter
function updateHeaderCounts() {
    const filtered = getFilteredPairs();

    // Count entities by kind (extract from rendered HTML)
    const entityKinds = {};
    filtered.forEach(card => {
        const kinds = Array.from(card.querySelectorAll('.entity-kind'))
            .map(el => el.textContent.trim());

        const kind1 = kinds[0] || 'unknown';
        const kind2 = kinds.length > 1 ? kinds[1] : 'unknown';

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

    // Update the stats display
    const statsEl = document.querySelector('.header-stats');
    if (statsEl) {
        const statsItems = [];
        statsItems.push(`<span class="stat-item"><span class="stat-label">Pairs:</span> <span>${filtered.length}</span></span>`);

        for (const [kind, names] of Object.entries(entityKinds)) {
            statsItems.push(`<span class="stat-item"><span class="stat-label">${escapeHtml(kind)}:</span> <span>${names.size}</span></span>`);
        }

        statsItems.push(`<span class="stat-item"><span class="stat-label">Documents:</span> <span>${docIndices.size}</span></span>`);

        statsEl.innerHTML = statsItems.join('\\n');
    }
}

// Filter pairs based on search and rejected toggle
function getFilteredPairs() {
    const allPairs = document.querySelectorAll('.pair-card');
    return Array.from(allPairs).filter(card => {
        // Filter rejected
        if (!state.showRejected && card.dataset.accepted === 'false') {
            return false;
        }

        // Search filter
        if (state.searchQuery) {
            const q = state.searchQuery;
            const searchable = [
                card.dataset.e1,
                card.dataset.e1a,
                card.dataset.e2,
                card.dataset.e2a,
                card.dataset.rel
            ].join(' ').toLowerCase();

            if (!searchable.includes(q)) {
                return false;
            }
        }

        return true;
    });
}

// Update pair list visibility and selection based on filters
function updatePairListDisplay() {
    const filtered = getFilteredPairs();
    const sidebar = document.getElementById('sidebar');
    const allCards = sidebar.querySelectorAll('.pair-card');

    const filteredSet = new Set(filtered);

    if (state.selectedPairId !== null) {
        const selectedExists = !!findFilteredPairById(state.selectedPairId, filtered);
        if (!selectedExists) {
            state.selectedPairId = null;
            state.openDocumentIdx = null;
        }
    }

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
}

// Select a pair
function selectPair(pairId) {
    const filtered = getFilteredPairs();
    const match = findFilteredPairById(pairId, filtered);
    if (!match) {
        console.warn(`Pair ${pairId} not available with current filters`);
        return;
    }
    state.selectedPairId = pairId;
    state.openDocumentIdx = null;
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
               relationships, polarity, confidence} = group;

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
                ? `<span class="relationship-label-small">${escapeHtml(relationships[0])}</span>`
                : '';

        // Build polarity/confidence badge
        const polarityMap = {supporting: 'S', refuting: 'R', neutral: 'N', irrelevant: 'I'};
        const badge = polarity && confidence
            ? `<span class="pc-chip polarity-${escapeHtml(polarity)} confidence-${escapeHtml(confidence)}">
                   <span>${escapeHtml(polarityMap[polarity] || polarity)}</span>
                   <span>${escapeHtml(confidence)}</span>
               </span>`
            : confidence ? `<span class="confidence-${escapeHtml(confidence)}">${escapeHtml(confidence)}</span>`
            : polarity ? `<span class="polarity-badge polarity-${escapeHtml(polarity)}">${escapeHtml(polarityMap[polarity] || polarity)}</span>`
            : '';

        const isOpen = state.openDocumentIdx === idx;
        return `
        <div class="document-accordion">
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
    // Capture the clicked header's viewport position before any changes
    const clickedHeader = document.querySelectorAll('.document-header')[idx];
    const headerTopBeforeToggle = clickedHeader ? clickedHeader.getBoundingClientRect().top : null;

    if (state.openDocumentIdx === idx) {
        state.openDocumentIdx = null;
    } else {
        state.openDocumentIdx = idx;
    }
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
            span.onclick = () => {
                const targetPairIdx = parseInt(entityPairs[0]);
                selectPair(targetPairIdx);
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
function renderReasoning() {
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

// Select pair and open specific document by doc_idx
function selectPairAndDocument(pairIdx, docIdx) {
    // Find pairIdx in filtered pairs
    const filtered = getFilteredPairs();
    const match = findFilteredPairById(pairIdx, filtered);

    if (!match) {
        console.warn(`Pair ${pairIdx} not in filtered list`);
        return;
    }

    state.selectedPairId = pairIdx;

    const pairCard = match.card;
    const docIndices = pairCard.dataset.docs.trim().split(' ').map(n => parseInt(n));
    const assessIdx = docIndices.indexOf(docIdx);

    // Open the document (or first if not found)
    state.openDocumentIdx = assessIdx !== -1 ? assessIdx : 0;

    updatePairListDisplay();
    renderContent();
    renderReasoning();
}

// Utility: escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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
