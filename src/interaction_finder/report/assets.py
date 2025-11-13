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
    --entity1-bg-hover: var(--pico-color-violet-300);
    --entity1-text: var(--pico-color-violet-450);
    --entity2-bg: var(--pico-color-lime-100);
    --entity2-bg-hover: var(--pico-color-lime-200);
    --entity2-text: var(--pico-color-lime-450);
    --entity-other-bg: var(--pico-color-slate-150);
    --entity-other-bg-hover: var(--pico-color-slate-250);
    --entity-fallback-bg: var(--pico-color-slate-100);
    --entity-fallback-bg-hover: var(--pico-color-slate-200);

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

    /* Navigation colors */
    --nav-hover-bg: var(--pico-color-cyan-50);
    --nav-active-bg: var(--pico-color-cyan-100);
    --nav-border: var(--pico-color-cyan-600);
    --nav-badge-bg: var(--pico-color-zinc-400);

    /* Connection/decoration colors */
    --connection-stroke: var(--pico-color-azure-600);
}

/* Dark theme colors (prefers-color-scheme: dark without explicit theme) */
@media only screen and (prefers-color-scheme: dark) {
    :root:not([data-theme]) {
        /* Entity highlighting colors (flipped shades: 200->800, 300->700, 450->550) */
        --entity1-bg: var(--pico-color-violet-800);
        --entity1-bg-hover: var(--pico-color-violet-700);
        --entity1-text: var(--pico-color-violet-550);
        --entity2-bg: var(--pico-color-lime-900);
        --entity2-bg-hover: var(--pico-color-lime-800);
        --entity2-text: var(--pico-color-lime-550);
        --entity-other-bg: var(--pico-color-slate-850);
        --entity-other-bg-hover: var(--pico-color-slate-750);
        --entity-fallback-bg: var(--pico-color-slate-900);
        --entity-fallback-bg-hover: var(--pico-color-slate-800);

        /* State colors (flipped: 100->900, 50->950, 600->400) */
        --selected-bg: var(--pico-color-azure-900);
        --selected-hover-bg: var(--pico-color-azure-950);
        --rejected-bg: var(--pico-color-red-950);
        --rejected-border: var(--pico-color-red-400);

        /* Confidence badge colors (flipped: 600->400, 400->600) */
        --confidence-high-bg: var(--pico-color-green-400);
        --confidence-medium-bg: var(--pico-color-pumpkin-600);
        --confidence-low-bg: var(--pico-color-red-400);

        /* Quote colors (flipped: 100->900, 400->600, 200->800, 600->400) */
        --quote-bg: var(--pico-color-zinc-900);
        --quote-border: var(--pico-color-zinc-600);
        --quote-emphasis-bg: var(--pico-color-amber-800);
        --quote-emphasis-border: var(--pico-color-amber-400);

        /* Navigation colors (flipped: 50->950, 100->900, 600->400, 400->600) */
        --nav-hover-bg: var(--pico-color-cyan-950);
        --nav-active-bg: var(--pico-color-cyan-900);
        --nav-border: var(--pico-color-cyan-400);
        --nav-badge-bg: var(--pico-color-zinc-600);

        /* Connection/decoration colors (flipped: 600->400) */
        --connection-stroke: var(--pico-color-azure-400);
    }
}

/* Dark theme colors (explicit [data-theme=dark]) */
[data-theme=dark] {
    /* Entity highlighting colors (flipped shades) */
    --entity1-bg: var(--pico-color-violet-800);
    --entity1-bg-hover: var(--pico-color-violet-700);
    --entity1-text: var(--pico-color-violet-550);
    --entity2-bg: var(--pico-color-lime-900);
    --entity2-bg-hover: var(--pico-color-lime-800);
    --entity2-text: var(--pico-color-lime-550);
    --entity-other-bg: var(--pico-color-slate-850);
    --entity-other-bg-hover: var(--pico-color-slate-750);
    --entity-fallback-bg: var(--pico-color-slate-900);
    --entity-fallback-bg-hover: var(--pico-color-slate-800);

    /* State colors */
    --selected-bg: var(--pico-color-azure-900);
    --selected-hover-bg: var(--pico-color-azure-950);
    --rejected-bg: var(--pico-color-red-950);
    --rejected-border: var(--pico-color-red-400);

    /* Confidence badge colors */
    --confidence-high-bg: var(--pico-color-green-400);
    --confidence-medium-bg: var(--pico-color-pumpkin-600);
    --confidence-low-bg: var(--pico-color-red-400);

    /* Quote colors */
    --quote-bg: var(--pico-color-zinc-900);
    --quote-border: var(--pico-color-zinc-600);
    --quote-emphasis-bg: var(--pico-color-amber-800);
    --quote-emphasis-border: var(--pico-color-amber-400);

    /* Navigation colors */
    --nav-hover-bg: var(--pico-color-cyan-950);
    --nav-active-bg: var(--pico-color-cyan-900);
    --nav-border: var(--pico-color-cyan-400);
    --nav-badge-bg: var(--pico-color-zinc-600);

    /* Connection/decoration colors */
    --connection-stroke: var(--pico-color-azure-400);
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

.header-search {
    max-width: 600px;
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

.pair-entities {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: var(--spacing-compact);
    align-items: center;
    margin-bottom: 0.25rem;
}

.entity-name {
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
}

.entity-name:hover {
    color: var(--pico-primary);
    text-decoration: underline;
}

.entity-name.left {
    text-align: left;
}

.entity-name.right {
    text-align: right;
}

.pair-arrow {
    color: var(--pico-muted-color);
    font-size: 1rem;
}

.pair-kinds {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: var(--spacing-compact);
    font-size: 0.75rem;
    color: var(--pico-muted-color);
    margin-bottom: var(--spacing-compact);
}

.entity-kind.left {
    text-align: left;
}

.entity-kind.right {
    text-align: right;
}

.relationship-label {
    font-style: italic;
    font-size: 0.7rem;
    color: var(--pico-muted-color);
}

.pair-variants {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid var(--pico-muted-border-color);
}

.variant-item {
    padding: 0.25rem 0;
    font-size: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.variant-relationship {
    font-style: italic;
    color: var(--pico-muted-color);
}

.variant-badge {
    font-size: 0.7rem;
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

.confidence-badge {
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
}

.document-stats {
    display: flex;
    gap: 1rem;
    align-items: center;
    font-size: 0.85rem;
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

.document-text {
    white-space: pre-wrap;
    font-family: var(--pico-font-family);
    line-height: 1.6;
    position: relative;
}

/* Entity highlights in document */
.entity-highlight {
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 600;
    cursor: pointer;
    position: relative;
}

/* Entity position-specific colors (entity1 vs entity2 in current pair) */
.entity-highlight.entity1 {
    background: var(--entity1-bg);
}

.entity-highlight.entity1:hover {
    background: var(--entity1-bg-hover);
}

.entity-highlight.entity2 {
    background: var(--entity2-bg);
}

.entity-highlight.entity2:hover {
    background: var(--entity2-bg-hover);
}

/* Entities from other pairs */
.entity-highlight.other {
    background: var(--entity-other-bg);
    font-weight: normal;
}

.entity-highlight.other:hover {
    background: var(--entity-other-bg-hover);
}

.entity-highlight.clickable {
    cursor: pointer;
}

/* Entities in reasoning panel - use color instead of background, lighter weight */
.reasoning-content .entity-highlight {
    background: none;
    font-weight: 500;
}

.reasoning-content .entity-highlight.entity1 {
    color: var(--entity1-text);
}

.reasoning-content .entity-highlight.entity2 {
    color: var(--entity2-text);
}

/* Disable hover effects on entities in reasoning panel */
.reasoning-content .entity-highlight:hover {
    background: none;
}

.reasoning-content .entity-highlight.entity1:hover {
    background: none;
    color: var(--entity1-text);
}

.reasoning-content .entity-highlight.entity2:hover {
    background: none;
    color: var(--entity2-text);
}

/* Fallback for other entity kinds */
.entity-highlight.other-entity {
    background: var(--entity-fallback-bg);
    font-weight: normal;
    cursor: pointer;
}

.entity-highlight.other-entity:hover {
    background: var(--entity-fallback-bg-hover);
}

.quote-highlight {
    background: var(--quote-bg);
    padding: 0.25rem 0;
}

/* First highlighted span in a quote run gets left border */
.quote-highlight:not(.quote-span + .quote-highlight) {
    border-left: 3px solid var(--quote-border);
    padding-left: 0.5rem;
    margin-left: -0.5rem;
}

/* Last highlighted span in a quote run gets right border */
.quote-highlight:not(:has(+ .quote-highlight)) {
    border-right: 3px solid var(--quote-border);
    padding-right: 0.5rem;
    margin-right: -0.5rem;
}

.quote-blink {
    background: var(--quote-emphasis-bg);
    transition: background 0.5s ease-out, border-color 0.5s ease-out;
}

/* First blinked span gets left border */
.quote-blink:not(.quote-span + .quote-blink) {
    border-left: 3px solid var(--quote-emphasis-border);
    padding-left: 0.5rem;
    margin-left: -0.5rem;
}

/* Last blinked span gets right border */
.quote-blink:not(:has(+ .quote-blink)) {
    border-right: 3px solid var(--quote-emphasis-border);
    padding-right: 0.5rem;
    margin-right: -0.5rem;
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

.reasoning-title {
    font-weight: 600;
    margin-bottom: var(--spacing-compact);
    color: var(--pico-primary);
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

/* Connection lines (SVG overlay) */
.connection-lines {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
}

.connection-line {
    stroke: var(--connection-stroke);
    stroke-width: 2;
    fill: none;
    opacity: 0.6;
}
"""

REPORT_JS = """
// Report interactivity

// NOTE: Markdown rendering is done in Python during report generation.
// Pre-rendered document HTML is embedded in <template> tags and cloned on-demand.
// This approach provides fast initial page load and efficient DOM reuse.

// Global state
const state = {
    data: null,
    selectedPairIdx: null,
    openDocumentIdx: null,
    searchQuery: '',
    showRejected: false,
    currentScrolledQuote: null,
};

// Initialize report
function initReport() {
    state.data = window.REPORT_DATA;

    // Set up event listeners
    document.getElementById('search-input').addEventListener('input', handleSearch);
    document.getElementById('show-rejected').addEventListener('change', handleToggleRejected);

    // Initial render
    renderPairList();
    renderContent();
    renderReasoning();
}

// Search handler
function handleSearch(e) {
    state.searchQuery = e.target.value.toLowerCase();
    renderPairList();
}

// Toggle rejected pairs
function handleToggleRejected(e) {
    state.showRejected = e.target.checked;
    updateHeaderCounts();
    renderPairList();
}

// Update header counts based on current filter
function updateHeaderCounts() {
    const filtered = getFilteredPairs();

    // Count entities in filtered pairs
    const entityKinds = {};
    filtered.forEach(pair => {
        if (!entityKinds[pair.entity1.kind]) {
            entityKinds[pair.entity1.kind] = new Set();
        }
        if (!entityKinds[pair.entity2.kind]) {
            entityKinds[pair.entity2.kind] = new Set();
        }
        entityKinds[pair.entity1.kind].add(pair.entity1.name);
        entityKinds[pair.entity2.kind].add(pair.entity2.name);
    });

    // Update the stats display
    const statsEl = document.querySelector('.header-stats');
    if (statsEl) {
        const statsItems = [];
        statsItems.push(`<span class="stat-item"><span class="stat-label">Pairs:</span> <span>${filtered.length}</span></span>`);

        for (const [kind, names] of Object.entries(entityKinds)) {
            statsItems.push(`<span class="stat-item"><span class="stat-label">${escapeHtml(kind)}:</span> <span>${names.size}</span></span>`);
        }

        statsItems.push(`<span class="stat-item"><span class="stat-label">Documents:</span> <span>${state.data.metadata.resource_count}</span></span>`);

        statsEl.innerHTML = statsItems.join('\\n');
    }
}

// Filter pairs based on search and rejected toggle
function getFilteredPairs() {
    return state.data.pairs.filter(pair => {
        // Filter rejected
        if (!state.showRejected && !pair.accepted) {
            return false;
        }

        // Search filter
        if (state.searchQuery) {
            const query = state.searchQuery;
            const matchEntity =
                pair.entity1.name.toLowerCase().includes(query) ||
                pair.entity2.name.toLowerCase().includes(query) ||
                pair.entity1.aliases.some(a => a.toLowerCase().includes(query)) ||
                pair.entity2.aliases.some(a => a.toLowerCase().includes(query));

            const matchRelationship = pair.relationship.toLowerCase().includes(query);

            if (!matchEntity && !matchRelationship) {
                return false;
            }
        }

        return true;
    });
}

// Render pair list in sidebar
function renderPairList() {
    const sidebar = document.getElementById('sidebar');
    const filtered = getFilteredPairs();

    sidebar.innerHTML = filtered.map((pair, idx) => `
        <div class="pair-card ${pair.accepted ? '' : 'rejected'} ${state.selectedPairIdx === idx ? 'selected' : ''}"
             onclick="selectPair(${idx})"
             data-pair-idx="${idx}">
            <div class="pair-entities">
                <span class="entity-name left"
                      onclick="event.stopPropagation(); filterByEntity('${escapeHtml(pair.entity1.name)}')"
                      title="${escapeHtml(pair.entity1.aliases.join(', '))}">
                    ${escapeHtml(pair.entity1.name)}
                </span>
                <span class="pair-arrow">⟷</span>
                <span class="entity-name right"
                      onclick="event.stopPropagation(); filterByEntity('${escapeHtml(pair.entity2.name)}')"
                      title="${escapeHtml(pair.entity2.aliases.join(', '))}">
                    ${escapeHtml(pair.entity2.name)}
                </span>
            </div>
            <div class="pair-kinds">
                <span class="entity-kind left">${escapeHtml(pair.entity1.kind)}</span>
                <span class="relationship-label">${escapeHtml(pair.relationship)}</span>
                <span class="entity-kind right">${escapeHtml(pair.entity2.kind)}</span>
            </div>
            <div class="pair-meta">
                <span class="pair-counts">${pair.doc_count} docs, ${pair.quote_count} quotes</span>
                <span class="confidence-badge confidence-${pair.confidence}">${pair.confidence}</span>
            </div>
            ${pair.variants ? `
                <div class="pair-variants">
                    ${pair.variants.map((v, vIdx) => `
                        <div class="variant-item">
                            <span class="variant-relationship">${escapeHtml(v.relationship)}</span>
                            <span class="confidence-badge variant-badge confidence-${v.confidence}">${v.confidence}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `).join('');
}

// Select a pair
function selectPair(idx) {
    const filtered = getFilteredPairs();
    state.selectedPairIdx = idx;
    state.openDocumentIdx = null;
    renderPairList();
    renderContent();
    renderReasoning();
}

// Filter by entity name
function filterByEntity(entityName) {
    const searchInput = document.getElementById('search-input');
    searchInput.value = entityName;
    state.searchQuery = entityName.toLowerCase();
    renderPairList();
}

// Render content area
function renderContent() {
    const content = document.getElementById('content');

    if (state.selectedPairIdx === null) {
        content.innerHTML = '<div class="content-placeholder">Select a pair to view documents</div>';
        return;
    }

    const filtered = getFilteredPairs();
    const pair = filtered[state.selectedPairIdx];

    // Sort assessments by quote count
    const sortedAssessments = [...pair.assessments].sort((a, b) =>
        b.quotes.length - a.quotes.length
    );

    content.innerHTML = sortedAssessments.map((assess, idx) => `
        <div class="document-accordion">
            <div class="document-header ${state.openDocumentIdx === idx ? 'open' : ''}"
                 onclick="toggleDocument(${idx})">
                <div class="document-title">${escapeHtml(assess.title)}</div>
                <div class="document-stats">
                    <span>${assess.quotes.length} quotes</span>
                    <span class="confidence-badge confidence-${assess.confidence}">${assess.confidence}</span>
                </div>
            </div>
            <div class="document-content ${state.openDocumentIdx === idx ? 'open' : ''}"
                 id="doc-content-${idx}">
                ${state.openDocumentIdx === idx ? renderDocument(assess) : ''}
            </div>
        </div>
    `).join('');
}

// Toggle document accordion
function toggleDocument(idx) {
    if (state.openDocumentIdx === idx) {
        state.openDocumentIdx = null;
    } else {
        state.openDocumentIdx = idx;
    }
    renderContent();
    renderReasoning();
}

// Highlight entities in text (for reasoning panels)
function highlightEntities(text, entity1Terms, entity2Terms, entity1Kind, entity2Kind, entity1Name, entity2Name, escapeFirst = true) {
    if (escapeFirst) {
        text = escapeHtml(text);
    }

    // Create term-to-type (entity1/entity2) and term-to-canonical mapping
    const termTypes = new Map();
    const termCanonical = new Map();
    entity1Terms.forEach(term => {
        termTypes.set(term.toLowerCase(), 'entity1');
        termCanonical.set(term.toLowerCase(), entity1Name);
    });
    entity2Terms.forEach(term => {
        termTypes.set(term.toLowerCase(), 'entity2');
        termCanonical.set(term.toLowerCase(), entity2Name);
    });

    const allTerms = [...entity1Terms, ...entity2Terms];
    // Sort by length (longest first) to avoid partial matches
    allTerms.sort((a, b) => b.length - a.length);

    allTerms.forEach(term => {
        const escapedTerm = escapeHtml(term);
        const type = termTypes.get(term.toLowerCase()) || 'other';
        const canonical = termCanonical.get(term.toLowerCase()) || term;
        const regex = new RegExp(`(${escapedTerm.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')})`, 'gi');
        text = text.replace(regex, `<span class="entity-highlight ${type}" title="${escapeHtml(canonical)}">$1</span>`);
    });

    return text;
}

// Highlight quotes for the current assessment
function highlightQuotesForAssessment(assess) {
    // Remove all existing quote highlights first
    document.querySelectorAll('.quote-highlight, .quote-blink').forEach(el => {
        el.classList.remove('quote-highlight', 'quote-blink');
    });

    // Get quote IDs from this assessment
    const doc = state.data.documents[assess.resource_id];
    if (!doc || !doc.quote_map) return;

    // Get all quote IDs that belong to this assessment
    const assessmentQuoteIds = new Set();
    Object.keys(doc.quote_map).forEach(quoteId => {
        assessmentQuoteIds.add(quoteId);
    });

    // Find all quote spans in the document and highlight those containing our quote IDs
    const docElement = document.querySelector('.document-text');
    if (!docElement) return;

    docElement.querySelectorAll('.quote-span').forEach(spanEl => {
        // Check if this span has any of our quote IDs in its class list
        const hasMatchingQuote = Array.from(spanEl.classList).some(className =>
            assessmentQuoteIds.has(className)
        );

        if (hasMatchingQuote) {
            spanEl.classList.add('quote-highlight');
        }
    });
}

// Update highlighting classes on entity spans based on current pair selection
function updateDocumentHighlights(docId, currentPairIdx) {
    const doc = state.data.documents[docId];
    if (!doc || !doc.entity_map) return;

    const currentPair = state.data.pairs[currentPairIdx];
    if (!currentPair) return;

    // Get current pair's entity names (normalized)
    const entity1Name = currentPair.entity1.name.toLowerCase();
    const entity2Name = currentPair.entity2.name.toLowerCase();

    // Update classes on all entity spans in the document
    Object.entries(doc.entity_map).forEach(([entityId, entityMeta]) => {
        const spanElement = document.getElementById(entityId);
        if (!spanElement) return;

        // Clear existing classes
        spanElement.className = 'entity-highlight';

        // Determine entity type relative to current pair
        const entityName = entityMeta.name.toLowerCase();
        if (entityName === entity1Name) {
            spanElement.classList.add('entity1');
        } else if (entityName === entity2Name) {
            spanElement.classList.add('entity2');
        } else {
            // Entity from a different pair
            spanElement.classList.add('other');

            // Add click handler to navigate to the pair containing this entity
            if (entityMeta.pair_indices && entityMeta.pair_indices.length > 0) {
                spanElement.classList.add('clickable');
                spanElement.onclick = () => {
                    // Navigate to the first pair that contains this entity
                    // Note: pair_indices are indices into the FULL state.data.pairs array
                    const fullArrayIdx = entityMeta.pair_indices[0];

                    // Convert from full array index to filtered array index
                    const filtered = getFilteredPairs();
                    const targetPair = state.data.pairs[fullArrayIdx];
                    const filteredIdx = filtered.findIndex(p => p === targetPair);

                    if (filteredIdx !== -1) {
                        // Found in filtered list - select it and try to keep document open
                        selectPairByEntity(entityMeta.name, filteredIdx);
                    } else {
                        // Not in filtered list (maybe rejected and hidden) - still navigate
                        // but need to ensure the pair is visible first
                        console.warn('Target pair not in filtered list - may be hidden by filters');
                    }
                };
            }
        }
    });
}

// Navigate to pair by entity click
// pairIdx is the index into the FILTERED pairs array
function selectPairByEntity(entityName, pairIdx) {
    // Save current document if open
    const currentDocId = state.openDocumentIdx !== null ?
        getFilteredPairs()[state.selectedPairIdx]?.assessments[state.openDocumentIdx]?.resource_id : null;

    // Select the new pair (using filtered index)
    state.selectedPairIdx = pairIdx;

    // Try to find and re-open the same document in the new pair
    if (currentDocId) {
        const filtered = getFilteredPairs();
        const newPair = filtered[pairIdx];
        const docIdx = newPair.assessments.findIndex(a => a.resource_id === currentDocId);
        state.openDocumentIdx = docIdx !== -1 ? docIdx : null;
    } else {
        state.openDocumentIdx = null;
    }

    renderPairList();
    renderContent();
    renderReasoning();
}

// Navigate to pair, open document, and scroll to specific quote
function selectPairByEntityAndQuote(entityName, pairIdx, docId, quoteIdx) {
    // Select the new pair
    state.selectedPairIdx = pairIdx;

    // Find and open the document in this pair
    const newPair = state.data.pairs[pairIdx];
    const docIdx = newPair.assessments.findIndex(a => a.resource_id === docId);
    state.openDocumentIdx = docIdx !== -1 ? docIdx : 0;

    renderPairList();
    renderContent();
    renderReasoning();

    // After rendering, scroll to the quote
    setTimeout(() => {
        scrollToQuote(quoteIdx);
    }, 100);
}

// Render document with highlights
function renderDocument(assess) {
    const doc = state.data.documents[assess.resource_id];
    if (!doc) {
        return '<p>Document not found</p>';
    }

    // Get the pre-rendered document template
    const templateId = `doc-template-${assess.resource_id}`;
    const template = document.getElementById(templateId);

    if (!template) {
        console.error(`Template not found: ${templateId}`);
        return '<p>Document template not found</p>';
    }

    // Clone the template content to create a new document instance
    const clone = template.content.cloneNode(true);

    // Create a temporary container to get the HTML string
    const tempContainer = document.createElement('div');
    tempContainer.appendChild(clone);
    const html = tempContainer.innerHTML;

    // After rendering, update the highlighting classes based on current pair
    // Use setTimeout to ensure DOM is updated
    setTimeout(() => {
        updateDocumentHighlights(assess.resource_id, state.selectedPairIdx);
        highlightQuotesForAssessment(assess);
    }, 0);

    return html;
}

// Render reasoning sidebar
function renderReasoning() {
    const rightbar = document.getElementById('rightbar');

    if (state.selectedPairIdx === null) {
        rightbar.innerHTML = '<div class="reasoning-panel"><p>Select a pair to view reasoning</p></div>';
        return;
    }

    const filtered = getFilteredPairs();
    const pair = filtered[state.selectedPairIdx];

    if (state.openDocumentIdx === null) {
        // Show overall pair reasoning with entity highlighting
        const entity1Terms = [pair.entity1.name, ...pair.entity1.aliases];
        const entity2Terms = [pair.entity2.name, ...pair.entity2.aliases];
        const highlightedReasoning = highlightEntities(pair.reasoning, entity1Terms, entity2Terms, pair.entity1.kind, pair.entity2.kind, pair.entity1.name, pair.entity2.name, true);

        rightbar.innerHTML = `
            <div class="reasoning-panel">
                <div class="reasoning-title">Overall Assessment</div>
                <div class="reasoning-content">
                    <p><strong>Relationship:</strong> ${escapeHtml(pair.relationship)}</p>
                    <p><strong>Confidence:</strong> <span class="confidence-badge confidence-${pair.confidence}">${pair.confidence}</span></p>
                    <p>${highlightedReasoning}</p>
                </div>
                <div class="entity-details-section">
                    <div class="reasoning-title">Entity Details</div>
                    <div class="entity-detail-item">
                        <strong>${escapeHtml(pair.entity1.name)}</strong> (${escapeHtml(pair.entity1.kind)})
                        ${pair.entity1.aliases.length > 0 ? `
                            <div class="alias-tooltip">Aliases: ${escapeHtml(pair.entity1.aliases.join(', '))}</div>
                        ` : ''}
                    </div>
                    <div class="entity-detail-item">
                        <strong>${escapeHtml(pair.entity2.name)}</strong> (${escapeHtml(pair.entity2.kind)})
                        ${pair.entity2.aliases.length > 0 ? `
                            <div class="alias-tooltip">Aliases: ${escapeHtml(pair.entity2.aliases.join(', '))}</div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    } else {
        // Show document-specific reasoning with entity highlighting
        const sortedAssessments = [...pair.assessments].sort((a, b) =>
            b.quotes.length - a.quotes.length
        );
        const assess = sortedAssessments[state.openDocumentIdx];

        const entity1Terms = [pair.entity1.name, ...pair.entity1.aliases];
        const entity2Terms = [pair.entity2.name, ...pair.entity2.aliases];
        const highlightedReasoning = highlightEntities(assess.reasoning, entity1Terms, entity2Terms, pair.entity1.kind, pair.entity2.kind, true);

        // Build quote navigation with better styling
        const quoteNav = assess.quotes.length > 0 ? `
            <div class="quote-navigation">
                <div class="quote-nav-title">Jump to Quotes (${assess.quotes.length})</div>
                <ul class="quote-nav-list">
                    ${assess.quotes.map((q, idx) => `
                        <li class="quote-nav-item" onclick="scrollToQuote(${idx})" title="${escapeHtml(q.text)}">
                            <span class="quote-number">${idx + 1}</span>
                            <span class="quote-preview">${escapeHtml(q.text.substring(0, 60))}${q.text.length > 60 ? '...' : ''}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        ` : '';

        // Find other pairs that use this document
        const currentDocId = assess.resource_id;
        const otherPairs = state.data.pairs
            .map((p, idx) => ({ pair: p, idx: idx }))
            .filter(({ pair, idx }) =>
                idx !== state.selectedPairIdx &&
                pair.assessments.some(a => a.resource_id === currentDocId)
            );

        const otherPairsNav = otherPairs.length > 0 ? `
            <div class="quote-navigation">
                <div class="quote-nav-title">Other Pairs (${otherPairs.length})</div>
                <ul class="quote-nav-list">
                    ${otherPairs.map(({ pair, idx }) => `
                        <li class="quote-nav-item" onclick="selectPairAndDocument(${idx}, '${escapeHtml(currentDocId)}')">
                            <span class="quote-preview">
                                ${escapeHtml(pair.entity1.name)} ⟷ ${escapeHtml(pair.entity2.name)}
                            </span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        ` : '';

        rightbar.innerHTML = `
            <div class="reasoning-panel">
                <div class="reasoning-title">Document Assessment</div>
                <div class="reasoning-content">
                    <p><strong>Document:</strong> ${escapeHtml(assess.title)}</p>
                    <p><strong>Relationship:</strong> ${escapeHtml(assess.relationship)}</p>
                    <p><strong>Confidence:</strong> <span class="confidence-badge confidence-${assess.confidence}">${assess.confidence}</span></p>
                    <p>${highlightedReasoning}</p>
                </div>
                ${quoteNav}
                ${otherPairsNav}
            </div>
        `;
    }
}

// Scroll to quote with emphasis
function scrollToQuote(quoteIdx) {
    // Find the quote element for the Nth quote in THIS assessment
    // Need to match the quote text from assess.quotes to a quote ID in doc.quote_map

    const filtered = getFilteredPairs();
    const pair = filtered[state.selectedPairIdx];
    if (!pair || state.openDocumentIdx === null) return;

    const sortedAssessments = [...pair.assessments].sort((a, b) =>
        b.quotes.length - a.quotes.length
    );
    const assess = sortedAssessments[state.openDocumentIdx];

    // Get the quote text from this assessment
    if (quoteIdx < 0 || quoteIdx >= assess.quotes.length) return;
    const targetQuote = assess.quotes[quoteIdx];

    const doc = state.data.documents[assess.resource_id];
    if (!doc || !doc.quote_map) return;

    // Find the quote ID in doc.quote_map that matches this quote's text
    // Match by original spans since quote text might have ellipsis
    let matchingQuoteId = null;
    for (const [quoteId, quoteMeta] of Object.entries(doc.quote_map)) {
        // Compare original spans - they should match exactly
        if (quoteMeta.original_spans.length === targetQuote.spans.length) {
            const spansMatch = quoteMeta.original_spans.every((span, i) =>
                span[0] === targetQuote.spans[i][0] && span[1] === targetQuote.spans[i][1]
            );
            if (spansMatch) {
                matchingQuoteId = quoteId;
                break;
            }
        }
    }

    if (!matchingQuoteId) return;

    // Find the first quote span that has this quote ID in its class list
    const docElement = document.querySelector('.document-text');
    if (!docElement) return;

    const quoteEl = Array.from(docElement.querySelectorAll('.quote-span')).find(spanEl =>
        spanEl.classList.contains(matchingQuoteId)
    );

    if (quoteEl) {
        // Scroll into view
        quoteEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Add blink class for emphasis
        quoteEl.classList.add('quote-blink');

        // Remove blink class after animation completes
        setTimeout(() => {
            quoteEl.classList.remove('quote-blink');
        }, 2000);
    }
}

// Select pair and open specific document
function selectPairAndDocument(pairIdx, docId) {
    state.selectedPairIdx = pairIdx;

    // Find the document in this pair's assessments
    const pair = state.data.pairs[pairIdx];
    const docIdx = pair.assessments.findIndex(a => a.resource_id === docId);
    state.openDocumentIdx = docIdx !== -1 ? docIdx : 0;

    renderPairList();
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
