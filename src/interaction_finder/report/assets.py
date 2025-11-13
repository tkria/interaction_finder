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

body {
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-areas:
        "header header header"
        "sidebar content rightbar"
        "footer footer footer";
    grid-template-columns: 320px 1fr 320px;
    grid-template-rows: auto 1fr auto;
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
        flex-shrink: 0;
        width: 400px;
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
    background: var(--pico-color-azure-100);
}

.pair-card.rejected {
    background: var(--pico-color-red-50);
    border-color: var(--pico-color-red-600);
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
    background: var(--pico-color-green-600);
    color: white;
}

.confidence-medium {
    background: var(--pico-color-pumpkin-400);
    color: white;
}

.confidence-low {
    background: var(--pico-color-red-600);
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
    background: var(--pico-color-azure-50);
}

.document-header.open {
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
    border-color: var(--pico-primary);
    background: var(--pico-color-azure-100);
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
    background: var(--pico-color-violet-200);
}

.entity-highlight.entity1:hover {
    background: var(--pico-color-violet-300);
}

.entity-highlight.entity2 {
    background: var(--pico-color-lime-100);
}

.entity-highlight.entity2:hover {
    background: var(--pico-color-lime-200);
}

/* Entities from other pairs */
.entity-highlight.other {
    background: var(--pico-color-slate-150);
    font-weight: normal;
}

.entity-highlight.other:hover {
    background: var(--pico-color-slate-250);
}

/* Entities in reasoning panel - use color instead of background, lighter weight */
.reasoning-content .entity-highlight {
    background: none;
    font-weight: 500;
}

.reasoning-content .entity-highlight.entity1 {
    color: var(--pico-color-violet-450);
}

.reasoning-content .entity-highlight.entity2 {
    color: var(--pico-color-lime-450);
}

/* Disable hover effects on entities in reasoning panel */
.reasoning-content .entity-highlight:hover {
    background: none;
}

.reasoning-content .entity-highlight.entity1:hover {
    background: none;
    color: var(--pico-color-violet-450);
}

.reasoning-content .entity-highlight.entity2:hover {
    background: none;
    color: var(--pico-color-lime-450);
}

/* Fallback for other entity kinds */
.entity-highlight.other-entity {
    background: var(--pico-color-slate-100);
    font-weight: normal;
    cursor: pointer;
}

.entity-highlight.other-entity:hover {
    background: var(--pico-color-slate-200);
}

.quote-highlight {
    background: var(--pico-color-zinc-100);
    border-left: 3px solid var(--pico-color-zinc-400);
    padding-left: 0.5rem;
    margin: 0.5rem 0;
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
    background: var(--pico-color-cyan-50);
    border-color: var(--pico-color-cyan-600);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.quote-nav-item.active {
    background: var(--pico-color-cyan-100);
    border-color: var(--pico-color-cyan-600);
}

.quote-number {
    flex-shrink: 0;
    width: 1.5rem;
    height: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--pico-color-zinc-400);
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

/* Footer */
footer {
    grid-area: footer;
    padding: var(--spacing-compact) var(--spacing-card);
    border-top: 1px solid var(--pico-muted-border-color);
    background: var(--pico-background-color);
    font-size: 0.8rem;
    color: var(--pico-muted-color);
    text-align: center;
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
    stroke: var(--pico-color-azure-600);
    stroke-width: 2;
    fill: none;
    opacity: 0.6;
}
"""

REPORT_JS = """
// Report interactivity

// Simple markdown to HTML converter
function markdownToHtml(text) {
    // Headers (increased by 1 level: h1->h2, h2->h3, h3->h4)
    text = text.replace(/^### (.*$)/gim, '<h4>$1</h4>');
    text = text.replace(/^## (.*$)/gim, '<h3>$1</h3>');
    text = text.replace(/^# (.*$)/gim, '<h2>$1</h2>');

    // Bold
    text = text.replace(/\\*\\*([^\\*]+)\\*\\*/gim, '<strong>$1</strong>');

    // Italic (both * and _ syntax)
    text = text.replace(/\\*([^\\*]+)\\*/gim, '<em>$1</em>');
    text = text.replace(/_([^_]+)_/gim, '<em>$1</em>');

    // Line breaks
    text = text.replace(/\\n\\n/g, '</p><p>');
    text = '<p>' + text + '</p>';

    return text;
}

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

// Highlight entities in quotes (includes current pair + other entities)
function highlightEntitiesInQuote(text, entity1Terms, entity2Terms, entity1Kind, entity2Kind, entity1Name, entity2Name, otherEntities, quoteIdx, docId) {
    // DON'T escape HTML yet - we need to work with plain text first

    // Build list of all entity matches with positions
    const matches = [];

    // Find entity1 matches
    entity1Terms.forEach(term => {
        const termLower = term.toLowerCase();
        const textLower = text.toLowerCase();
        let pos = 0;
        while ((pos = textLower.indexOf(termLower, pos)) !== -1) {
            matches.push({
                start: pos,
                end: pos + term.length,
                type: 'entity1',
                canonical: entity1Name,
                term: text.substring(pos, pos + term.length)
            });
            pos += term.length;
        }
    });

    // Find entity2 matches
    entity2Terms.forEach(term => {
        const termLower = term.toLowerCase();
        const textLower = text.toLowerCase();
        let pos = 0;
        while ((pos = textLower.indexOf(termLower, pos)) !== -1) {
            matches.push({
                start: pos,
                end: pos + term.length,
                type: 'entity2',
                canonical: entity2Name,
                term: text.substring(pos, pos + term.length)
            });
            pos += term.length;
        }
    });

    // Find other entity matches
    otherEntities.forEach(entity => {
        entity.terms.forEach(term => {
            const termLower = term.toLowerCase();
            const textLower = text.toLowerCase();
            let pos = 0;
            while ((pos = textLower.indexOf(termLower, pos)) !== -1) {
                matches.push({
                    start: pos,
                    end: pos + term.length,
                    type: 'other',
                    canonical: entity.canonical,
                    pairIdx: entity.pairIdx,
                    term: text.substring(pos, pos + term.length)
                });
                pos += term.length;
            }
        });
    });

    // Remove overlapping matches (keep first/longest)
    matches.sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));
    const filtered = [];
    for (const match of matches) {
        const overlaps = filtered.some(f =>
            (match.start >= f.start && match.start < f.end) ||
            (match.end > f.start && match.end <= f.end)
        );
        if (!overlaps) {
            filtered.push(match);
        }
    }

    // Build HTML by inserting highlights
    if (filtered.length === 0) {
        return escapeHtml(text);
    }

    // Build result by processing text segments and highlights
    let result = '';
    let lastPos = 0;

    filtered.sort((a, b) => a.start - b.start);

    filtered.forEach(match => {
        // Add escaped text before this match
        if (match.start > lastPos) {
            result += escapeHtml(text.substring(lastPos, match.start));
        }

        // Add the highlight span
        if (match.type === 'other') {
            result += `<span class="entity-highlight other" title="${escapeHtml(match.canonical)}" onclick="selectPairByEntityAndQuote('${escapeHtml(match.canonical)}', ${match.pairIdx}, '${escapeHtml(docId)}', ${quoteIdx})">${escapeHtml(match.term)}</span>`;
        } else {
            result += `<span class="entity-highlight ${match.type}" title="${escapeHtml(match.canonical)}">${escapeHtml(match.term)}</span>`;
        }

        lastPos = match.end;
    });

    // Add any remaining text after the last match
    if (lastPos < text.length) {
        result += escapeHtml(text.substring(lastPos));
    }

    return result;
}

// Navigate to pair by entity click
function selectPairByEntity(entityName, pairIdx) {
    // Save current document if open
    const currentDocId = state.openDocumentIdx !== null ?
        getFilteredPairs()[state.selectedPairIdx]?.assessments[state.openDocumentIdx]?.resource_id : null;

    // Select the new pair
    state.selectedPairIdx = pairIdx;

    // Try to find and re-open the same document in the new pair
    if (currentDocId) {
        const newPair = state.data.pairs[pairIdx];
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

    const filtered = getFilteredPairs();
    const pair = filtered[state.selectedPairIdx];

    // Get all entity aliases to search for (current pair)
    const entity1Terms = [pair.entity1.name, ...pair.entity1.aliases];
    const entity2Terms = [pair.entity2.name, ...pair.entity2.aliases];

    // Collect all OTHER entities from all pairs for highlighting
    const otherEntities = [];
    const currentEntityNames = new Set([pair.entity1.name.toLowerCase(), pair.entity2.name.toLowerCase()]);

    state.data.pairs.forEach((p, pIdx) => {
        if (pIdx !== state.selectedPairIdx) {
            const e1Name = p.entity1.name;
            const e2Name = p.entity2.name;
            if (!currentEntityNames.has(e1Name.toLowerCase())) {
                otherEntities.push({
                    terms: [e1Name, ...p.entity1.aliases],
                    kind: p.entity1.kind,
                    canonical: e1Name,
                    pairIdx: pIdx
                });
            }
            if (!currentEntityNames.has(e2Name.toLowerCase())) {
                otherEntities.push({
                    terms: [e2Name, ...p.entity2.aliases],
                    kind: p.entity2.kind,
                    canonical: e2Name,
                    pairIdx: pIdx
                });
            }
        }
    });

    let html = '';
    let lastEnd = 0;

    // Sort quotes by position
    const sortedQuotes = [...assess.quotes].sort((a, b) => a.spans[0][0] - b.spans[0][0]);

    sortedQuotes.forEach((quote, qIdx) => {
        const quoteStart = quote.spans[0][0];
        const quoteEnd = quote.spans[quote.spans.length - 1][1];

        // Add text before quote (rendered as markdown, no entity highlighting)
        if (quoteStart > lastEnd) {
            const beforeText = doc.text.substring(lastEnd, quoteStart);
            html += markdownToHtml(beforeText);
        }

        // Process quote text with entity highlighting (current pair + other entities)
        let quoteText = doc.text.substring(quoteStart, quoteEnd);
        const highlightedQuote = highlightEntitiesInQuote(quoteText, entity1Terms, entity2Terms, pair.entity1.kind, pair.entity2.kind, pair.entity1.name, pair.entity2.name, otherEntities, qIdx, assess.resource_id);

        html += `<div class="quote-highlight" id="quote-${qIdx}" data-quote-idx="${qIdx}">${highlightedQuote}</div>`;
        lastEnd = quoteEnd;
    });

    // Add remaining text
    if (lastEnd < doc.text.length) {
        const remainingText = doc.text.substring(lastEnd);
        html += markdownToHtml(remainingText);
    }

    return `<div class="document-text">${html}</div>`;
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
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--pico-muted-border-color);">
                    <div class="reasoning-title">Entity Details</div>
                    <div style="margin-bottom: 1rem;">
                        <strong>${escapeHtml(pair.entity1.name)}</strong> (${escapeHtml(pair.entity1.kind)})
                        ${pair.entity1.aliases.length > 0 ? `
                            <div class="alias-tooltip">Aliases: ${escapeHtml(pair.entity1.aliases.join(', '))}</div>
                        ` : ''}
                    </div>
                    <div>
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

// Scroll to quote
function scrollToQuote(quoteIdx) {
    const quoteEl = document.getElementById(`quote-${quoteIdx}`);
    if (quoteEl) {
        quoteEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Highlight temporarily
        quoteEl.style.background = 'var(--pico-color-zinc-250)';
        setTimeout(() => {
            quoteEl.style.background = '';
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
