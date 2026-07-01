"use strict";
// Runner UI: drives POST /runs, then renders the SSE stream into an activity
// feed and three stage tabs. The same render path serves a live run and a
// back-filled checkpoint -- both arrive as the same {scope, description, data}
// events, with meta.cleared marking each stage boundary.

const $ = (id) => document.getElementById(id);
const STAGES = ["keywords", "search", "extraction"];
// Symbol-sheet icon id per stage (declared in ui.html).
const STAGE_ICON = {
    keywords: "icon-keywords",
    search: "icon-search",
    extraction: "icon-extraction",
};
// Which stage owns the run while a given scope is streaming -- this drives the
// active tab and stage status. The keyword stage's own literature searches
// (keywords.search.*) surface in the Search panel but keep Keywords active, so
// they are deliberately absent here (see PANEL_STAGE).
const SCOPE_STAGE = {
    "keywords.started": "keywords",
    "keywords.scored": "keywords",
    "search.queries": "search",
    "search.results": "search",
    "search.selected": "search",
    "extraction.started": "extraction",
    "extraction.pair_judged": "extraction",
};
// Which panel a scope's data populates (independent of the active stage).
const PANEL_STAGE = {
    "keywords.scored": "keywords",
    "keywords.search.queries": "search",
    "keywords.search.selected": "search",
    "search.queries": "search",
    "search.selected": "search",
    "extraction.pair_judged": "extraction",
};

let model = null;   // the focused run's accumulated state (see `runs` registry)

// === Entity-kind chips ===
// Two independent chip editors (new-run form, resume panel), each a {kinds,
// containerId} pair, so the same render/add logic serves both.
const newRunKinds = { kinds: [], containerId: "entity-chips" };
const resumeKinds = { kinds: [], containerId: "resume-chips" };

function renderChips(editor) {
    const box = $(editor.containerId);
    box.innerHTML = "";
    editor.kinds.forEach((kind, i) => {
        const chip = document.createElement("span");
        chip.className = "entity-chip";
        const label = document.createElement("span");
        label.textContent = kind;
        chip.appendChild(label);
        const x = document.createElement("button");
        x.textContent = "×";
        x.title = "remove";
        x.setAttribute("aria-label", `remove ${kind}`);
        x.onclick = () => { editor.kinds.splice(i, 1); renderChips(editor); };
        chip.appendChild(x);
        box.appendChild(chip);
    });
    if (editor === newRunKinds) validateNewRun();
}

function addKind(editor, inputId) {
    const input = $(inputId);
    const kind = input.value.trim().toLowerCase();
    if (kind) { editor.kinds.push(kind); input.value = ""; renderChips(editor); }
}

// Same-kind pairs: the pipeline encodes "allow self-pairs" by repeating a kind,
// so the switch doubles each unique kind.
function withSamePairs(kinds, allow) {
    if (!allow) return kinds.slice();
    return kinds.flatMap((k) => [k, k]);
}

// === New run ===
// Start is enabled only once a topic, at least one entity kind, and an output
// path are all provided. Called on every relevant field/chip change.
function validateNewRun() {
    const ready = !!$("topic").value.trim()
        && newRunKinds.kinds.length > 0
        && !!$("output").value.trim();
    $("start").disabled = !ready;
}

async function startRun() {
    $("form-error").hidden = true;
    const body = {
        checkpoint_or_topic: $("topic").value,
        entity_kinds: withSamePairs(newRunKinds.kinds, $("same-kind").checked),
        backend: null,  // search backend now lives in the config editor
        output_path: $("output").value || null,
        config_overrides: collectConfigOverrides(),
    };
    let resp, data;
    try {
        resp = await fetch("/runs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        data = await resp.json();
    } catch (e) {
        return showFormError("Could not reach the server.");
    }
    if (!resp.ok) return showFormError(data.error || "Failed to start run.");
    beginRun(data);
}

function showFormError(msg) {
    const el = $("form-error");
    el.textContent = msg;
    el.hidden = false;
}

// === Recent history ===
// Fetch and render the recent-checkpoint list. Shown (by updateRecentVisibility)
// only on the Existing-checkpoint tab when nothing is currently loaded.
async function refreshRecent() {
    let entries = [];
    try {
        const resp = await fetch("/recent");
        entries = (await resp.json()).entries || [];
    } catch (e) { /* leave empty on failure */ }
    const list = $("recent-list");
    list.innerHTML = "";
    for (const e of entries) list.appendChild(recentRow(e));
    $("recent-card").dataset.hasEntries = entries.length ? "1" : "";
    updateRecentVisibility();
}

function recentRow(entry) {
    const li = document.createElement("li");
    const meta = entry.pair_count
        ? `${entry.pair_count} pairs · ${entry.opened_at.slice(0, 10)}`
        : entry.opened_at.slice(0, 10);
    // Leading slot: always present (fixed width) so titles align across rows.
    // Holds an open-report link when a report exists; otherwise a partial-run
    // marker for an incomplete checkpoint; otherwise empty.
    const slot = document.createElement("span");
    slot.className = "recent-report-slot";
    if (entry.report_cached) {
        const report = document.createElement("button");
        report.className = "recent-report";
        report.setAttribute("data-tooltip", "Open report");
        report.setAttribute("data-placement", "top");
        report.innerHTML = `<svg class="ui-icon" aria-hidden="true" focusable="false"><use href="#icon-open"/></svg>`;
        report.onclick = (ev) => {
            ev.stopPropagation();
            window.open(`/recent/${entry.id}/report`, "_blank");
        };
        slot.append(report);
    } else if (entry.complete === false) {
        const partial = document.createElement("span");
        partial.className = "recent-partial";
        partial.setAttribute("data-tooltip", "Incomplete — can be resumed");
        partial.setAttribute("data-placement", "top");
        partial.innerHTML = `<svg class="ui-icon" aria-hidden="true" focusable="false"><use href="#icon-partial"/></svg>`;
        slot.append(partial);
    }
    li.append(slot);
    const open = document.createElement("button");
    open.className = "recent-open";
    open.title = entry.path;
    open.innerHTML = `<span class="recent-topic">${escapeHtml(entry.topic)}</span>` +
        `<span class="recent-meta">${escapeHtml(meta)}</span>`;
    open.onclick = () => openRecent(entry.path);
    li.append(open);
    const remove = document.createElement("button");
    remove.className = "recent-remove";
    remove.title = "Remove from recent";
    remove.textContent = "×";
    remove.onclick = (ev) => { ev.stopPropagation(); removeRecent(entry.id); };
    li.append(remove);
    return li;
}

function openRecent(path) {
    // Equivalent to switching to the Existing-checkpoint tab and picking it,
    // even when invoked from the New run tab.
    selectEntry("existing");
    $("checkpoint-path").value = path;
    loadCheckpoint();
}

async function removeRecent(id) {
    try { await fetch(`/recent/${id}`, { method: "DELETE" }); } catch (e) {}
    refreshRecent();
}

async function clearRecent() {
    try { await fetch("/recent", { method: "DELETE" }); } catch (e) {}
    refreshRecent();
}

// The recent card is visible only on the existing-checkpoint tab, with nothing
// Show recent (when there are entries) on the New run tab, or on the Existing
// checkpoint tab while nothing is loaded. A stashed checkpoint only suppresses
// it on the Existing tab -- the New run tab has no loaded view to compete with.
function updateRecentVisibility() {
    const hasEntries = $("recent-card").dataset.hasEntries === "1";
    const onNew = !$("entry-new").hidden;
    const show = hasEntries && (onNew || loadedStatus === null);
    $("recent-card").hidden = !show;
}

// === Existing checkpoint ===
let loadedPath = "";
let loadedStatus = null;

// Load the chosen checkpoint and show how to continue it.
async function loadCheckpoint() {
    const path = $("checkpoint-path").value.trim();
    // Clear any previously-loaded checkpoint's stage info / report card first.
    clearLoadedView();
    if (!path) return showError("load-error", "Choose a checkpoint file.");
    // Reading a large checkpoint can take many seconds; show a busy indicator
    // so the wait after picking is not mistaken for nothing happening.
    $("load-busy").hidden = false;
    let resp, data;
    try {
        resp = await fetch("/load", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
        });
        data = await resp.json();
    } catch (e) {
        $("load-busy").hidden = true;
        return showError("load-error", "Could not reach the server.");
    }
    $("load-busy").hidden = true;
    if (!resp.ok) return showError("load-error", data.error || "Could not load checkpoint.");
    loadedPath = path;
    loadedStatus = data;
    updateRecentVisibility();   // something is now loaded -> hide recent
    refreshRecent();            // this open updated the registry
    renderResume(data);
    showLoaded(data);
}

// Resume controls in the form: entity kinds (when extraction has not run) plus
// the Continue button, shown only for a resumable (incomplete) checkpoint.
function renderResume(status) {
    const resumable = status.resumable;
    $("continue-panel").hidden = !resumable;
    if (!resumable) return;
    const needKinds = status.needs_entity_kinds;
    $("continue-fields").hidden = !needKinds;
    if (needKinds) { resumeKinds.kinds = []; renderChips(resumeKinds); }
}

// The report card (between the form and the stage info): View when cached,
// Generate otherwise; Generate streams progress then flips to View.
function renderReportCard(status) {
    const card = $("report-card");
    card.hidden = false;
    $("report-busy").hidden = true;
    const cached = status.report_cached;
    // No report is available either because the run is incomplete (resume it)
    // or it completed without finding any associations.
    if (status.has_report === false) {
        $("report-summary").textContent = status.complete
            ? "No associations found — nothing to report."
            : "Run incomplete — no report yet.";
        $("report-view").hidden = true;
        $("report-download").hidden = true;
        $("report-generate").hidden = true;
        return;
    }
    $("report-summary").textContent = cached
        ? "Report ready."
        : "No report generated yet.";
    $("report-view").hidden = !cached;
    $("report-download").hidden = !cached;
    $("report-generate").hidden = cached;
    $("report-view").onclick = () => window.open(`/report/${status.run_id}`, "_blank");
    $("report-download").onclick = () => downloadReport(status.run_id);
    $("report-generate").onclick = () => generateReport(status.run_id);
}

// Download the report file. The server sets a Content-Disposition attachment
// header (topic-derived filename) when ?download=1 is present.
function downloadReport(runId) {
    const a = document.createElement("a");
    a.href = `/report/${runId}?download=1`;
    a.download = "";  // hint; the server's filename takes precedence
    document.body.appendChild(a);
    a.click();
    a.remove();
}

// Generate the report, showing per-document progress from a dedicated SSE.
function generateReport(runId) {
    $("report-generate").hidden = true;
    $("report-busy").hidden = false;
    $("report-summary").textContent = "";
    const es = new EventSource(`/report/${runId}/events`);
    es.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "report_progress") {
            // Once every document is rendered, the report still has to be
            // assembled and written -- a step with no per-item progress -- so
            // switch the message rather than sit at N/N looking stuck.
            if (msg.total && msg.completed >= msg.total) {
                $("report-busy").textContent = "Finalising report…";
            } else {
                const n = msg.total ? ` ${msg.completed}/${msg.total}` : "";
                $("report-busy").textContent = `Rendering documents${n}…`;
            }
        } else if (msg.type === "report_done") {
            es.close();
            $("report-busy").hidden = true;
            $("report-summary").textContent = "Report ready.";
            $("report-view").hidden = false;
            $("report-download").hidden = false;
            window.open(`/report/${runId}`, "_blank");
        } else if (msg.type === "report_error") {
            es.close();
            $("report-busy").hidden = true;
            $("report-summary").textContent = msg.error || "Report generation failed.";
            $("report-generate").hidden = false;
        }
    };
    es.onerror = () => {
        es.close();
        $("report-busy").hidden = true;
        $("report-summary").textContent = "Lost connection during report generation.";
        $("report-generate").hidden = false;
    };
}

// Continue (resume) the loaded checkpoint, streaming the remaining stages.
async function continueRun() {
    $("load-error").hidden = true;
    const body = { path: loadedPath, entity_kinds: resumeKinds.kinds, backend: null };
    let resp, data;
    try {
        resp = await fetch("/resume", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        data = await resp.json();
    } catch (e) {
        return showError("load-error", "Could not reach the server.");
    }
    if (!resp.ok) return showError("load-error", data.error || "Could not continue run.");
    beginRun(data);
}

function showError(id, msg) {
    const el = $(id);
    el.textContent = msg;
    el.hidden = false;
}

// Tear down any loaded-checkpoint view (stage info, report card, resume panel)
// and forget the loaded checkpoint. Called when a *new* path is loaded, so a
// fresh load never shows stale data -- not on a mere tab switch (that stashes).
function clearLoadedView() {
    // Drop the previously-focused non-running run's stream, if any (a loaded
    // checkpoint is registered as a run that is never "running").
    const prev = focusedRunId && runs.get(focusedRunId);
    if (prev && prev.status !== "running") {
        if (prev.evtSource) prev.evtSource.close();
        runs.delete(prev.runId);
        focusedRunId = null;
    }
    model = null;
    loadedPath = "";
    loadedStatus = null;
    $("stages").dataset.active = "";
    $("tab-panels").innerHTML = "";
    $("tab-bar").innerHTML = "";
    $("report-card").hidden = true;
    $("continue-panel").hidden = true;
    $("load-error").hidden = true;
    $("load-busy").hidden = true;
}

// Show or hide the loaded-checkpoint sections (stage info + report card)
// without destroying them, so leaving and returning to the tab restores them.
function stashLoadedView(visible) {
    const show = visible && loadedStatus !== null;
    $("stages").dataset.active = show ? "1" : "";
    $("report-card").hidden = !show;
    updateRecentVisibility();
}

// Switch the entry-point tab. Leaving "Existing checkpoint" stashes (hides) any
// loaded checkpoint; returning restores it -- it is only discarded when a new
// path is loaded.
function selectEntry(which) {
    $("entry-tabs").querySelectorAll("button").forEach((b) =>
        b.setAttribute("aria-selected", b.dataset.entry === which ? "true" : "false"));
    $("entry-new").hidden = which !== "new";
    $("entry-existing").hidden = which !== "existing";
    stashLoadedView(which === "existing");
}

// === Run / stage view ===
// Multiple runs can be in progress at once. Each lives in `runs` with its own
// accumulated model, EventSource, and status line, so every active-run card row
// stays live concurrently. The stages/activity DOM renders whichever run is
// `focusedRunId`; `model` always points at that run's model so the render
// helpers below need no per-run plumbing.
const runs = new Map();
let focusedRunId = null;

function newModel(runId, startedAt) {
    const m = { stages: {}, activeStage: null, feed: [], lastIndex: -1, runId, startedAt };
    for (const s of STAGES) {
        m.stages[s] = { status: "pending", hasData: false, terms: [], queries: {}, pairs: [], summary: "" };
    }
    return m;
}

function formatElapsed(ms) {
    const total = Math.floor(ms / 1000);
    const mins = Math.floor(total / 60), secs = total % 60;
    return mins ? `${mins}m ${secs}s` : `${secs}s`;
}

// Start (or resume) a live run: register it, focus it, and stream its events.
function beginRun(info) {
    const startedAt = info.started_at ? info.started_at * 1000 : Date.now();
    const run = {
        runId: info.run_id, topic: info.topic, statusLine: "Starting pipeline…",
        startedAt, status: "running", model: newModel(info.run_id, startedAt),
        evtSource: null, checkpointPath: info.checkpoint_path || null,
    };
    runs.set(run.runId, run);
    connect(run);
    renderActiveRuns();
    focusRun(run.runId);
}

// Bind the stages/activity view to a run and render its accumulated state.
function focusRun(runId) {
    const run = runs.get(runId);
    if (!run) return;
    focusedRunId = runId;
    model = run.model;
    $("new-run").dataset.running = "1";
    $("report-card").hidden = true;
    $("recent-card").hidden = true;
    $("run-summary").dataset.active = "1";
    $("summary-text").textContent = run.topic;
    $("activity").dataset.active = "1";
    $("stages").dataset.active = "1";
    if (run.status === "running") {
        $("activity-done").hidden = true;
        $("activity-running").hidden = false;
        $("activity-status").setAttribute("aria-busy", "true");
        $("activity-status").textContent = run.statusLine || "";
        const cancel = $("cancel-run");
        cancel.hidden = false;
        cancel.disabled = false;
        cancel.removeAttribute("aria-busy");
    } else {
        renderDone(run);
    }
    renderFeed();
    renderTabs();
    selectTab(model.activeStage || currentTab || STAGES[0]);
    renderActiveRuns();   // focus changed: recompute which rows the card shows
    syncFocusedElapsed();
}

// Show a loaded checkpoint: a finished, read-only view (no active-run row).
function showLoaded(info) {
    const run = {
        runId: info.run_id, topic: info.topic, statusLine: "", startedAt: Date.now(),
        status: "loaded", model: newModel(info.run_id, Date.now()), evtSource: null,
    };
    runs.set(run.runId, run);
    focusedRunId = run.runId;
    model = run.model;
    $("stages").dataset.active = "1";
    $("activity").dataset.active = "";     // no live activity for a loaded file
    renderTabs();
    selectTab(STAGES[0]);
    connect(run);
    renderReportCard(info);
}

function connect(run) {
    if (run.evtSource) run.evtSource.close();
    run.evtSource = new EventSource(
        `/runs/${run.runId}/events?since=${run.model.lastIndex + 1}`
    );
    run.evtSource.onmessage = (e) => handleMessage(run, JSON.parse(e.data));
    run.evtSource.onerror = () => { /* browser auto-reconnects; since= resumes */ };
}

function handleMessage(run, msg) {
    const focused = run.runId === focusedRunId;
    if (msg.type === "event") {
        run.model.lastIndex = Math.max(run.model.lastIndex, msg.index);
        applyEvent(run, msg, focused);
    } else if (msg.type === "status") {
        // Ignore empty status snapshots (e.g. the initial catch-up before any
        // stage has set one) so they don't blank the "Starting…" line.
        if (msg.status) {
            run.statusLine = msg.status;
            if (focused) $("activity-status").textContent = run.statusLine;
            renderActiveRuns();
        }
    } else if (msg.type === "counters") {
        applyCounters(run, msg.counters, focused);
    } else if (msg.type === "done") {
        finishRun(run, msg);
    }
}

function applyEvent(run, msg, focused) {
    const m = run.model;
    // Forwarded log warnings/errors carry a severity so the feed can flag them.
    const level = msg.scope === "meta.error" ? "err"
        : msg.scope === "meta.warning" ? "warn" : "";
    pushFeed(m, msg.description, level);
    if (focused) renderFeed();
    if (msg.scope === "meta.cleared") {
        if (m.activeStage) m.stages[m.activeStage].status = "done";
        m.activeStage = null;
        if (focused) renderTabs();
        return;
    }
    // Active-stage changes (tab switch, status) follow the run's owning stage.
    const owner = SCOPE_STAGE[msg.scope];
    if (owner && m.activeStage !== owner) {
        m.activeStage = owner;
        m.stages[owner].status = "active";
        if (focused) selectTab(owner);
    }
    // Data routes to its panel regardless of the active stage -- so the keyword
    // stage's literature searches fill the Search panel while Keywords is active.
    const panel = PANEL_STAGE[msg.scope];
    if (!panel) { if (focused) renderTabs(); return; }
    const st = m.stages[panel];
    if (msg.scope === "keywords.scored") {
        st.terms = msg.data.terms || [];
    } else if (msg.scope === "keywords.search.queries" || msg.scope === "search.queries") {
        st.searchedQueries = (st.searchedQueries || []).concat(msg.data.queries || []);
    } else if (msg.scope === "search.selected" || msg.scope === "keywords.search.selected") {
        st.queries[msg.data.query] = msg.data;
    } else if (msg.scope === "extraction.pair_judged") {
        st.pairs.push(msg.data);
    }
    st.hasData = true;
    if (focused) {
        renderTabs();
        if (panel === currentTab) renderPanel(panel);
    }
}

function applyCounters(run, counters, focused) {
    const m = run.model;
    if (!m.activeStage) return;
    const line = counters
        .filter((c) => c.status !== "unstarted")
        .map((c) => `${c.name} ${c.completed}${c.total ? "/" + c.total : ""}`)
        .join(" · ");
    m.stages[m.activeStage].summary = line;
    if (focused && m.activeStage === currentTab) renderPanel(m.activeStage);
}

function pushFeed(m, text, level) {
    if (!text) return;
    m.feed.unshift({ text, level: level || "" });
    m.feed = m.feed.slice(0, 3);
}

function renderFeed() {
    const ul = $("activity-feed");
    ul.innerHTML = "";
    for (const line of (model ? model.feed : [])) {
        const li = document.createElement("li");
        li.textContent = line.text;
        if (line.level) li.className = `feed-${line.level}`;
        ul.appendChild(li);
    }
}

// On load, re-attach to any runs still in progress server-side (e.g. after a
// page reload) so their card rows reappear and keep streaming.
async function rebuildActiveRuns() {
    let entries = [];
    try {
        entries = (await (await fetch("/runs/active")).json()).runs || [];
    } catch (e) { return; }
    for (const info of entries) {
        if (runs.has(info.run_id)) continue;
        const startedAt = info.started_at ? info.started_at * 1000 : Date.now();
        const run = {
            runId: info.run_id, topic: info.topic, statusLine: info.status || "Working…",
            startedAt, status: "running", model: newModel(info.run_id, startedAt),
            evtSource: null, checkpointPath: null,
        };
        runs.set(run.runId, run);
        connect(run);
    }
    renderActiveRuns();
}

// === Active-run card ===
let activeRunsTimer = null;

// Render the pinned active-run card. The focused run is driven by the activity
// panel instead, so the card only earns its place when *other* runs exist: it
// is hidden when the only running run is the one on screen. When shown, the
// focused run still appears (so the count is honest) but without the redundant
// timer + View/Cancel -- those belong to the activity panel it already owns.
function renderActiveRuns() {
    const running = [...runs.values()].filter((r) => r.status === "running");
    running.sort((a, b) => a.startedAt - b.startedAt);
    const others = running.filter((r) => r.runId !== focusedRunId);
    const card = $("active-runs"), list = $("active-runs-list");
    card.hidden = others.length === 0;
    $("active-runs-plural").textContent = running.length > 1 ? "s" : "";
    list.innerHTML = "";
    if (!card.hidden) {
        for (const run of running) {
            list.appendChild(activeRunRow(run, run.runId === focusedRunId));
        }
    }
    // A single shared ticker advances every row's elapsed (and the focused view).
    if (running.length && !activeRunsTimer) {
        activeRunsTimer = setInterval(tickActiveElapsed, 1000);
    } else if (!running.length && activeRunsTimer) {
        clearInterval(activeRunsTimer); activeRunsTimer = null;
    }
    tickActiveElapsed();
}

// One active-run row. The focused run shows status only (it is being viewed);
// other runs add an elapsed timer and View/Cancel actions.
function activeRunRow(run, isFocused) {
    const li = document.createElement("li");
    li.dataset.runId = run.runId;
    const spinner = `<span class="arun-spinner" aria-busy="true"></span>`;
    const main = `<span class="arun-main"><span class="arun-topic">${escapeHtml(run.topic)}</span>` +
        `<span class="arun-status"> · ${escapeHtml(run.statusLine || "Working…")}</span></span>`;
    li.innerHTML = spinner + main;
    if (isFocused) return li;
    li.insertAdjacentHTML("beforeend",
        `<span class="arun-elapsed"></span>` +
        `<span class="arun-actions">` +
        `<button class="secondary outline arun-view" type="button">` +
        `<svg class="ui-icon" aria-hidden="true" focusable="false"><use href="#icon-eye"/></svg>View</button>` +
        `<button class="secondary outline arun-cancel" type="button">` +
        `<svg class="ui-icon" aria-hidden="true" focusable="false"><use href="#icon-cancel"/></svg>Cancel</button>` +
        `</span>`);
    li.querySelector(".arun-view").onclick = () => focusRun(run.runId);
    li.querySelector(".arun-cancel").onclick = () => promptCancel(run.runId);
    return li;
}

function tickActiveElapsed() {
    const now = Date.now();
    for (const li of $("active-runs-list").children) {
        const run = runs.get(li.dataset.runId);
        const el = li.querySelector(".arun-elapsed");
        if (run && el) el.textContent = formatElapsed(now - run.startedAt);
    }
    syncFocusedElapsed();
}

// Keep the focused run's activity-card elapsed in step with the same clock.
function syncFocusedElapsed() {
    const run = focusedRunId && runs.get(focusedRunId);
    if (run && run.status === "running") {
        $("activity-elapsed").textContent = formatElapsed(Date.now() - run.startedAt);
    }
}

// Open the cancel-confirmation dialog for a specific run.
let pendingCancelRunId = null;
function promptCancel(runId) {
    pendingCancelRunId = runId;
    openModal("confirm-cancel-dialog");
}

async function cancelRun(runId) {
    if (!runId) return;
    if (runId === focusedRunId) {
        const btn = $("cancel-run");
        btn.disabled = true;
        btn.setAttribute("aria-busy", "true");
    }
    try {
        await fetch(`/runs/${runId}/cancel`, { method: "POST" });
    } catch (e) { /* the terminal done still arrives over SSE */ }
}

// Resume an incomplete run from its saved checkpoint, re-entering the live view.
async function resumeRunFromCard(run) {
    const btn = $("resume-run");
    btn.setAttribute("aria-busy", "true");
    let resp, data;
    try {
        resp = await fetch("/resume", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: run.checkpointPath, entity_kinds: [] }),
        });
        data = await resp.json();
    } catch (e) {
        btn.removeAttribute("aria-busy");
        return;
    }
    btn.removeAttribute("aria-busy");
    if (resp.ok) beginRun(data);
}

function finishRun(run, msg) {
    if (run.evtSource) { run.evtSource.close(); run.evtSource = null; }
    run.status = "finished";
    run.result = msg;
    const m = run.model;
    for (const s of STAGES) {
        if (m.stages[s].status === "active") m.stages[s].status = "done";
    }
    m.activeStage = null;
    renderActiveRuns();          // drop this run's row from the pinned card
    if (run.runId === focusedRunId) {
        renderTabs();
        renderDone(run);
    }
}

// Paint the focused run's collapsed "done" block from its terminal result.
function renderDone(run) {
    const msg = run.result || {};
    // A back-filled loaded checkpoint has no live activity block to collapse.
    if ($("activity").dataset.active !== "1") return;
    $("activity-status").removeAttribute("aria-busy");
    $("activity-running").hidden = true;
    $("activity-done").hidden = false;
    const label = $("done-label");
    const view = $("view-report"), dl = $("download-report"), resume = $("resume-run");
    view.hidden = true; dl.hidden = true; resume.hidden = true;
    if (msg.result === "success" && msg.has_report !== false) {
        label.textContent = "Completed";
        label.className = "ok";
        view.hidden = false;
        view.onclick = () => window.open(`/report/${run.runId}`, "_blank");
        dl.hidden = false;
        dl.onclick = () => downloadReport(run.runId);
    } else if (msg.result === "success") {
        label.textContent = "Completed — no associations found";
        label.className = "ok";
    } else if (msg.result === "cancelled") {
        // Completed stages leave a partial, resumable checkpoint on disk.
        label.textContent = "Cancelled — partial progress saved";
        label.className = "cancelled";
        showResume(run, resume);
    } else {
        label.textContent = "Failed" + (msg.error ? ` — ${msg.error}` : "");
        label.className = "err";
        showResume(run, resume);
    }
    $("new-run-again").hidden = false;
}

// Offer Resume for an incomplete run whose checkpoint was saved to a path.
function showResume(run, btn) {
    if (!run.checkpointPath) return;
    btn.hidden = false;
    btn.onclick = () => resumeRunFromCard(run);
}

// === Tabs & panels ===
let currentTab = STAGES[0];

function renderTabs() {
    const bar = $("tab-bar");
    bar.innerHTML = "";
    for (const s of STAGES) {
        const st = model.stages[s];
        const status = st.status;
        const btn = document.createElement("button");
        btn.dataset.status = status;
        btn.setAttribute("aria-selected", s === currentTab ? "true" : "false");
        // A pending stage is reachable once it has any data (e.g. the Search
        // panel populated by the keyword stage's own literature searches).
        if (status === "pending" && !st.hasData) btn.disabled = true;
        // Active stage shows Pico's aria-busy spinner in place of its icon;
        // pending/done stages show the stage glyph (colour conveys the state).
        if (status === "active") btn.setAttribute("aria-busy", "true");
        const icon = status === "active"
            ? ""
            : `<svg class="ui-icon tab-stage-icon" aria-hidden="true" focusable="false"><use href="#${STAGE_ICON[s]}"/></svg>`;
        btn.innerHTML = `${icon}${s[0].toUpperCase() + s.slice(1)}`;
        btn.onclick = () => { if (!btn.disabled) selectTab(s); };
        bar.appendChild(btn);
    }
}

function selectTab(stage) {
    currentTab = stage;
    renderTabs();
    renderPanel(stage);
}

function renderPanel(stage) {
    const panels = $("tab-panels");
    const st = model.stages[stage];
    let html = st.summary ? `<p><small>${escapeHtml(st.summary)}</small></p>` : "";
    if (stage === "keywords") {
        const rows = st.terms.map((t) =>
            `<li><span class="grow">${escapeHtml(t.term)}</span><small class="num">${t.score.toFixed(2)}</small></li>`
        ).join("");
        html += rows ? `<ul class="data-list">${rows}</ul>` : emptyNote("No bridging terms yet.");
    } else if (stage === "search") {
        const items = Object.values(st.queries).map(renderSearchQuery).join("");
        const queries = st.searchedQueries || [];
        if (items) {
            // Once results are selected, show those accordions (the queries are
            // noted as a one-line summary above them).
            if (queries.length) {
                html += `<p><small>Ran ${queries.length} ` +
                    `${queries.length === 1 ? "query" : "queries"}</small></p>`;
            }
            html += items;
        } else if (queries.length) {
            // Searching is underway but nothing is selected yet: list the
            // queries so the panel is not empty.
            const qs = queries.map((q) => `<li>${escapeHtml(q)}</li>`).join("");
            html += `<p><small>Searching…</small></p><ul class="data-list">${qs}</ul>`;
        } else {
            html += emptyNote("No searches yet.");
        }
    } else if (stage === "extraction") {
        const rows = st.pairs.map((p) =>
            `<li><span class="verdict-dot ${p.accepted ? "yes" : "no"}"></span>` +
            `<span class="grow"><strong>${escapeHtml(p.entity1)}</strong> ${escapeHtml(p.relationship || "")} <strong>${escapeHtml(p.entity2)}</strong></span>` +
            `<small class="num">ev ${p.evidence ?? "?"}</small></li>`
        ).join("");
        // A finished extraction with no pairs found nothing; before then it is
        // still pending results.
        const empty = st.status === "done"
            ? emptyNote("No associations found in the fetched documents.")
            : emptyNote("No judged pairs yet.");
        html += rows ? `<ul class="data-list">${rows}</ul>` : empty;
    }
    panels.innerHTML = html;
}

// One query accordion: an icon + query title in the summary, with a
// right-justified "N selected" (or "N / M" when rejected results are known),
// and the picked then rejected results stacked in the body.
function renderSearchQuery(q) {
    const picked = q.picked || [], rejected = q.rejected || [];
    const total = picked.length + rejected.length;
    const count = rejected.length
        ? `${picked.length} / ${total}`
        : `${picked.length} selected`;
    const icon = `<svg class="ui-icon" aria-hidden="true" focusable="false"><use href="#icon-find"/></svg>`;
    const rows = picked.map((r) => searchResult(r, true)).join("")
        + rejected.map((r) => searchResult(r, false)).join("");
    const title = q.query || "Selected results";
    return `<details class="search-q" open><summary>${icon}`
        + `<span class="grow">${escapeHtml(title)}</span>`
        + `<small class="num">${count}</small></summary>`
        + `<ul class="result-list">${rows}</ul></details>`;
}

// One result: title (foreground, underlined link), a breadcrumb URL, and the
// snippet when present. Rejected results are dimmed via the .rejected class.
function searchResult(r, picked) {
    const title = escapeHtml(r.title || r.url || "");
    const titleEl = r.url
        ? `<a class="res-title" href="${escapeAttr(r.url)}" target="_blank">${title}</a>`
        : `<span class="res-title">${title}</span>`;
    const crumb = r.url ? `<div class="res-url">${formatUrl(r.url)}</div>` : "";
    const snippet = r.snippet ? `<div class="res-snippet">${escapeHtml(r.snippet)}</div>` : "";
    return `<li class="${picked ? "" : "rejected"}">${titleEl}${crumb}${snippet}</li>`;
}

// Breadcrumb form of a URL: drop the scheme, bold the domain, and join path
// segments with " > ". Falls back to the raw (escaped) URL if parsing fails.
function formatUrl(url) {
    try {
        const u = new URL(url);
        const domain = `<strong>${escapeHtml(u.hostname)}</strong>`;
        const segments = u.pathname.split("/").filter(Boolean).map(decodeSegment);
        return [domain, ...segments].join(" &rsaquo; ");
    } catch {
        return escapeHtml(url.replace(/^https?:\/\//, ""));
    }
}

function decodeSegment(seg) {
    try { return escapeHtml(decodeURIComponent(seg)); }
    catch { return escapeHtml(seg); }
}

function emptyNote(text) {
    return `<p><small>${escapeHtml(text)}</small></p>`;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;");
}

// Rate-limit the key-status re-fetch while a model string is being typed.
function debounce(fn, ms) {
    let timer;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

// === Config editor ===
// The form is generated server-side from the spec; every control carries its
// dotted config path (name) and the spec default (data-default). The client
// only highlights non-default fields, persists, and collects run overrides.

// A control's current value as a string (checkboxes -> "true"/"false").
function configFieldValue(input) {
    return input.type === "checkbox" ? String(input.checked) : input.value;
}

// True when a control differs from its spec default.
function isNonDefault(input) {
    return configFieldValue(input) !== input.dataset.default;
}

// The env var a "provider:model" needs, or null if no single-key row matches
// its prefix. Read from the key-status rows, which carry prefix->env.
function providerEnvForModel(model) {
    const prefix = (model || "").split(":", 1)[0];
    if (!prefix) return null;
    const row = keyStatus.shown.find((r) => r.prefix === prefix);
    return row ? row.env : null;
}

// The missing env var for an agent llm field whose model can't run, else null.
// Only explicitly-typed models are judged, plus agents._.llm (which every unset
// agent inherits).
function modelKeyMissing(input) {
    const model = input.value.trim();
    if (!model && input.name !== "agents._.llm") return null;
    const effective = model || input.dataset.default || "openai:gpt-4o-mini";
    const env = providerEnvForModel(effective);
    if (!env) return null;
    const set = keySetByEnv();
    // A custom endpoint means a local server ignoring OPENAI_API_KEY.
    if (env === "OPENAI_API_KEY" && set["OPENAI_BASE_URL"]) return null;
    return set[env] ? null : env;
}

// Font Awesome triangle-exclamation; fill via currentColor so CSS tints it.
const NOKEY_ICON =
    '<span class="agent-nokey"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">' +
    '<path d="M256 0c14.7 0 28.2 8.1 35.2 21l216 400c6.7 12.4 6.4 27.4-.8 39.5S486.1 480 472 480L40 480' +
    'c-14.1 0-27.2-7.4-34.4-19.5s-7.5-27.1-.8-39.5l216-400c7-12.9 20.5-21 35.2-21zm0 352a32 32 0 1 0 0 64' +
    ' 32 32 0 1 0 0-64zm0-192c-18.2 0-32.7 15.5-31.4 33.7l7.4 104c.9 12.5 11.4 22.3 23.9 22.3 12.6 0 23-9.7' +
    ' 23.9-22.3l7.4-104c1.3-18.2-13.1-33.7-31.4-33.7z"/></svg></span>';

// Refresh the non-default marks, missing-key warnings, and count badges.
function refreshConfigHighlights() {
    const inputs = $("config-form").querySelectorAll("input[name]");
    inputs.forEach((inp) => {
        // Agent llm inputs sit in a label.agent-head; everything else in a
        // .cfg-field. Highlight whichever wrapper this control has.
        const wrap = inp.closest(".cfg-field, .agent-head");
        if (wrap) wrap.classList.toggle("non-default", isNonDefault(inp));
    });
    $("config-form").querySelectorAll("input.agent-llm").forEach((inp) => {
        const head = inp.closest(".agent-head");
        let icon = head.querySelector(".agent-nokey");
        if (!icon) { head.insertAdjacentHTML("beforeend", NOKEY_ICON); icon = head.querySelector(".agent-nokey"); }
        const missing = modelKeyMissing(inp);
        head.classList.toggle("no-key", Boolean(missing));
        if (missing) {
            icon.setAttribute("data-tooltip", `${missing} is not set`);
            icon.setAttribute("data-placement", "left");
        } else {
            icon.removeAttribute("data-tooltip");
        }
    });
    $("config-form").querySelectorAll(".cfg-section").forEach((sec) => {
        const n = [...sec.querySelectorAll("input[name]")].filter(isNonDefault).length;
        const badge = sec.querySelector(":scope > summary > .cfg-count");
        badge.textContent = n;
        badge.hidden = n === 0;
        const nokey = [...sec.querySelectorAll("input.agent-llm")].filter(modelKeyMissing).length;
        let amber = sec.querySelector(":scope > summary > .cfg-nokey");
        if (!amber && nokey) {
            badge.insertAdjacentHTML("afterend", '<span class="cfg-count cfg-nokey"></span>');
            amber = sec.querySelector(":scope > summary > .cfg-nokey");
        }
        if (amber) { amber.textContent = `⚠ ${nokey}`; amber.hidden = nokey === 0; }
    });
    const total = [...inputs].filter(isNonDefault).length;
    $("config-summary").textContent = total
        ? `${total} field${total === 1 ? "" : "s"} changed from defaults`
        : "All defaults";
    // Mirror both counts on the Configuration button, visible without opening it.
    const badge = $("config-changed-badge");
    badge.textContent = total;
    badge.hidden = total === 0;
    const nokeyTotal = [...$("config-form").querySelectorAll("input.agent-llm")].filter(modelKeyMissing).length;
    const nokeyBadge = $("config-nokey-badge");
    nokeyBadge.textContent = `⚠ ${nokeyTotal}`;
    nokeyBadge.hidden = nokeyTotal === 0;
}

// Restore every control to its spec default.
function resetConfigToDefaults() {
    $("config-form").querySelectorAll("input[name]").forEach((inp) => {
        if (inp.type === "checkbox") inp.checked = inp.dataset.default === "true";
        else inp.value = inp.dataset.default;
    });
    refreshConfigHighlights();
}

// Dotted-key dict of only the fields that differ from defaults -- sent as a
// run's config_overrides (defaults are left implicit).
function collectConfigOverrides() {
    const overrides = {};
    $("config-form").querySelectorAll("input[name]").forEach((inp) => {
        if (isNonDefault(inp)) overrides[inp.name] = configFieldValue(inp);
    });
    return overrides;
}

// The last *committed* form state (path -> string value). Runs read the live
// form, but dismissing the modal restores this snapshot, so only Apply (or
// Save) commits edits. Starts from the server-rendered initial values.
let appliedConfig = null;

function snapshotConfigForm() {
    const snap = {};
    $("config-form").querySelectorAll("input[name]").forEach((inp) => {
        snap[inp.name] = configFieldValue(inp);
    });
    return snap;
}

// Write a snapshot back into the form (used to revert on dismiss).
function restoreConfigForm(snap) {
    $("config-form").querySelectorAll("input[name]").forEach((inp) => {
        const value = snap[inp.name];
        if (value === undefined) return;
        if (inp.type === "checkbox") inp.checked = value === "true";
        else inp.value = value;
    });
    refreshConfigHighlights();
}

// Commit the current form as the applied state (Apply / Save) and close.
function commitConfig() {
    appliedConfig = snapshotConfigForm();
    closeModal("config-dialog");
}

// Revert to the last committed state and close (× / Esc / backdrop).
function dismissConfig() {
    if (appliedConfig) restoreConfigForm(appliedConfig);
    closeModal("config-dialog");
}

async function saveConfigDefault() {
    const btn = $("config-save");
    btn.setAttribute("aria-busy", "true");
    try {
        const resp = await fetch("/config/default", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ overrides: collectConfigOverrides() }),
        });
        if (!resp.ok) throw new Error();
        // Persisting also commits: these values now stand even if the modal is
        // then dismissed rather than applied.
        appliedConfig = snapshotConfigForm();
        $("config-summary").textContent = "Saved as default.";
    } catch (e) {
        $("config-summary").textContent = "Could not save the default config.";
    } finally {
        btn.removeAttribute("aria-busy");
    }
}

// === API keys ===
// Provider values live in the server process for the session only. The modal
// shows the providers that are set, that the current config's models implicate,
// and NCBI (always); a picker adds the rest. Secrets report only set/unset and
// are masked once set; a plain value (a custom endpoint) is shown and editable.

// Last /keys/status response ({shown, available}), shared by the modal and the
// config form's missing-key badge. Refreshed against the live config overrides.
let keyStatus = { shown: [], available: [] };

async function refreshKeyStatus() {
    try {
        const resp = await fetch("/keys/status", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ overrides: collectConfigOverrides() }),
        });
        keyStatus = await resp.json();
    } catch (e) { /* keep the previous status on failure */ }
    return keyStatus;
}

function keySetByEnv() {
    const map = {};
    for (const r of keyStatus.shown) map[r.env] = r.set;
    return map;
}

async function openKeys() {
    await refreshKeyStatus();
    renderProviderRows();
    openModal("keys-dialog");
}

// Clear on a set secret only re-enables the field; blank on save keeps the
// stored key (whereas a blank plain value clears it).
function providerRow(p) {
    const row = document.createElement("div");
    row.className = "key-row" + (p.set ? " is-set" : "");
    const label =
        `<span class="key-label"><span class="key-dot"></span>${escapeHtml(p.label)}</span>`;
    if (!p.secret) {
        row.innerHTML = label +
            `<label class="key-field"><input type="text" autocomplete="off" ` +
            `data-env="${p.env}" value="${escapeAttr(p.value || "")}" ` +
            `placeholder="${escapeAttr(p.env)}"></label>`;
        return row;
    }
    if (p.set) {
        row.innerHTML = label +
            `<label class="key-field"><input type="password" autocomplete="off" ` +
            `data-env="${p.env}" value="••••••••" disabled>` +
            `<button type="button" class="key-clear secondary outline">Clear</button></label>`;
        const [input, clear] = [row.querySelector("input"), row.querySelector(".key-clear")];
        clear.onclick = () => { input.value = ""; input.disabled = false; clear.remove(); input.focus(); };
        return row;
    }
    row.innerHTML = label +
        `<label class="key-field"><input type="password" autocomplete="off" ` +
        `data-env="${p.env}" placeholder="${escapeHtml(p.env)}"></label>`;
    return row;
}

function renderProviderRows() {
    const box = $("keys-providers");
    box.innerHTML = "";
    for (const p of keyStatus.shown) box.appendChild(providerRow(p));
    const select = $("keys-add-select");
    select.innerHTML = keyStatus.available
        .map((p) => `<option value="${p.env}">${escapeHtml(p.label)}</option>`)
        .join("");
    $("keys-add").disabled = keyStatus.available.length === 0;
}

function addProviderRow() {
    const env = $("keys-add-select").value;
    const idx = keyStatus.available.findIndex((p) => p.env === env);
    if (idx < 0) return;
    const [picked] = keyStatus.available.splice(idx, 1);
    const full = { ...picked, set: false, secret: !env.endsWith("_BASE_URL") };
    keyStatus.shown.push(full);
    renderProviderRows();
}

async function applyKeys() {
    // Skip disabled inputs so an untouched masked secret is left as-is.
    const keys = {};
    $("keys-providers").querySelectorAll("input[data-env]:not([disabled])").forEach((inp) => {
        keys[inp.dataset.env] = inp.value.trim();
    });
    const btn = $("keys-apply");
    btn.setAttribute("aria-busy", "true");
    try {
        const resp = await fetch("/keys", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ keys, overrides: collectConfigOverrides() }),
        });
        keyStatus = await resp.json();
        refreshConfigHighlights();
        closeModal("keys-dialog");
    } catch (e) {
        $("keys-summary").textContent = "Could not set keys.";
    } finally {
        btn.removeAttribute("aria-busy");
    }
}

// === File picker (server-side browse; modal lifecycle mirrors the report) ===
const MODAL_ANIM_MS = 200;

function openModal(id) {
    const dialog = $(id), html = document.documentElement;
    const sb = window.innerWidth - html.clientWidth;
    if (sb) html.style.setProperty("--pico-scrollbar-width", `${sb}px`);
    html.classList.add("modal-is-open", "modal-is-opening");
    setTimeout(() => html.classList.remove("modal-is-opening"), MODAL_ANIM_MS);
    dialog.showModal();
}

function closeModal(id) {
    const dialog = $(id), html = document.documentElement;
    if (!dialog.open) return;
    html.classList.add("modal-is-closing");
    setTimeout(() => {
        html.classList.remove("modal-is-closing", "modal-is-open");
        html.style.removeProperty("--pico-scrollbar-width");
        dialog.close();
    }, MODAL_ANIM_MS);
}

// Picker state. mode "open" selects an existing .json; "save" chooses a
// directory + filename to write. targetId is the input the result fills;
// onPicked (optional) runs after an open-mode selection.
const picker = { mode: "open", targetId: null, dir: "", onPicked: null };

function openPicker(mode, targetId, title, onPicked) {
    picker.mode = mode;
    picker.targetId = targetId;
    picker.onPicked = onPicked || null;
    $("browse-title").textContent = title;
    $("browse-save-footer").hidden = mode !== "save";
    if (mode === "save") {
        const cur = $(targetId).value.trim();
        $("browse-filename").value = cur.replace(/^.*\//, "");  // basename, if any
    }
    openModal("browse-dialog");
    const cur = $(targetId).value.trim();
    browseTo(cur ? cur.replace(/[^/]*$/, "") : "");
}

async function browseTo(dir) {
    const err = $("browse-error");
    err.hidden = true;
    let data;
    try {
        const resp = await fetch(`/browse?dir=${encodeURIComponent(dir || "")}`);
        data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Could not list directory.");
    } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
        return;
    }
    picker.dir = data.dir;
    const list = $("browse-entries");
    list.innerHTML = "";
    renderBreadcrumbs(data.dir, data.home);
    for (const e of data.entries) list.appendChild(entryRow(e.name, e.path, e.is_dir));
}

// Confirm a save: join the current directory with the typed filename.
function confirmSave() {
    const name = $("browse-filename").value.trim();
    if (!name) return;
    $(picker.targetId).value = `${picker.dir}/${name}`;
    if (picker.targetId === "output") validateNewRun();
    closeModal("browse-dialog");
}

// A clickable path trail: each ancestor segment navigates to that directory.
function renderBreadcrumbs(dir, home) {
    const nav = $("browse-path");
    nav.innerHTML = "";
    // When the directory sits inside the user's home, collapse the home prefix
    // to a single home-icon crumb and show only the segments beneath it.
    // Otherwise fall back to a filesystem-root ("/") crumb.
    const homeIcon = `<svg class="ui-icon" aria-hidden="true" focusable="false"><use href="#icon-home"/></svg>`;
    const driveIcon = `<svg class="ui-icon" aria-hidden="true" focusable="false"><use href="#icon-drive"/></svg>`;
    let acc, segments;
    if (home && (dir === home || dir.startsWith(home + "/"))) {
        acc = home;
        segments = dir.slice(home.length).split("/").filter(Boolean);
        appendCrumb(nav, homeIcon, home, segments.length === 0, true);
    } else {
        acc = "";
        segments = dir.split("/").filter(Boolean);
        appendCrumb(nav, driveIcon, "/", segments.length === 0, true);
    }
    segments.forEach((seg, i) => {
        acc += "/" + seg;
        appendCrumb(nav, seg, acc, i === segments.length - 1);
    });
}

// label is plain text unless isHtml is set (e.g. the home glyph).
function appendCrumb(nav, label, path, isLast, isHtml) {
    if (nav.children.length) {
        const sep = document.createElement("span");
        sep.className = "crumb-sep";
        sep.textContent = "/";
        nav.appendChild(sep);
    }
    const el = document.createElement(isLast ? "strong" : "a");
    if (isHtml) el.innerHTML = label; else el.textContent = label;
    if (!isLast) {
        el.href = "#";
        el.onclick = (e) => { e.preventDefault(); browseTo(path); };
    }
    nav.appendChild(el);
}

function entryRow(name, path, isDir) {
    const li = document.createElement("li");
    const glyph = isDir ? "icon-folder" : "icon-json";
    li.innerHTML = `<span class="browse-icon"><svg class="ui-icon" aria-hidden="true" focusable="false"><use href="#${glyph}"/></svg></span><span class="grow">${escapeHtml(name)}</span>`;
    if (isDir) {
        li.onclick = () => browseTo(path);
    } else if (picker.mode === "save") {
        // Clicking an existing file in save mode prefills it (to overwrite).
        li.onclick = () => { $("browse-filename").value = name; };
    } else {
        li.onclick = () => {
            $(picker.targetId).value = path;
            closeModal("browse-dialog");
            if (picker.onPicked) picker.onPicked();
        };
    }
    return li;
}

// === Wiring ===
function init() {
    renderChips(newRunKinds);
    // New-run form.
    $("add-kind").onclick = () => addKind(newRunKinds, "kind-input");
    $("kind-input").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addKind(newRunKinds, "kind-input"); } });
    $("start").onclick = startRun;
    // Cancel goes through a confirmation dialog, both from the activity panel
    // (focused run) and from each active-run row.
    $("cancel-run").onclick = () => { if (focusedRunId) promptCancel(focusedRunId); };
    const cancelDlg = $("confirm-cancel-dialog");
    $("confirm-cancel-ok").onclick = () => {
        closeModal("confirm-cancel-dialog");
        cancelRun(pendingCancelRunId);
        pendingCancelRunId = null;
    };
    $("confirm-cancel-keep").onclick = () => closeModal("confirm-cancel-dialog");
    cancelDlg.querySelector('button[rel="prev"]').onclick = () => closeModal("confirm-cancel-dialog");
    cancelDlg.addEventListener("click", (e) => { if (e.target === cancelDlg) closeModal("confirm-cancel-dialog"); });
    cancelDlg.addEventListener("cancel", (e) => { e.preventDefault(); closeModal("confirm-cancel-dialog"); });
    $("new-run-again").onclick = () => {
        $("new-run").dataset.running = "";
        $("run-summary").dataset.active = "";
        $("activity").dataset.active = "";
        $("stages").dataset.active = "";
        focusedRunId = null;
        refreshRecent();
    };
    rebuildActiveRuns();
    // Resume panel chip editor.
    $("resume-add-kind").onclick = () => addKind(resumeKinds, "resume-kind-input");
    $("resume-kind-input").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addKind(resumeKinds, "resume-kind-input"); } });
    $("continue").onclick = continueRun;
    // Recent history: clearing all goes through a confirmation dialog.
    const confirmDlg = $("confirm-clear-dialog");
    $("recent-clear").onclick = (e) => { e.preventDefault(); openModal("confirm-clear-dialog"); };
    $("confirm-clear-ok").onclick = () => { closeModal("confirm-clear-dialog"); clearRecent(); };
    confirmDlg.querySelector('button[rel="prev"]').onclick = () => closeModal("confirm-clear-dialog");
    $("confirm-clear-cancel").onclick = () => closeModal("confirm-clear-dialog");
    confirmDlg.addEventListener("click", (e) => { if (e.target === confirmDlg) closeModal("confirm-clear-dialog"); });
    confirmDlg.addEventListener("cancel", (e) => { e.preventDefault(); closeModal("confirm-clear-dialog"); });
    refreshRecent();
    // Entry tabs: New run vs Existing checkpoint.
    $("entry-tabs").querySelectorAll("button").forEach((btn) =>
        btn.onclick = () => selectEntry(btn.dataset.entry));
    // Existing-checkpoint load: the picker (onPicked) loads on selection;
    // a typed path loads on Enter. We avoid the `change` event because it also
    // fires on programmatic fills (e.g. seeding the picker's start directory),
    // which would race a real selection with a stray directory-path request.
    $("checkpoint-path").addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); loadCheckpoint(); }
    });
    // Suggest a kebab-case output filename from the topic (as a placeholder, so
    // it never clobbers a path the user typed or picked themselves).
    $("topic").addEventListener("input", () => {
        const slug = kebab($("topic").value);
        $("output").placeholder = slug ? `${slug}.json` : "results.json";
        validateNewRun();
    });
    $("output").addEventListener("input", validateNewRun);
    // File-picker modal: open mode for the checkpoint, save mode for output.
    const dialog = $("browse-dialog");
    $("browse").onclick = () =>
        openPicker("open", "checkpoint-path", "Choose a checkpoint", loadCheckpoint);
    $("browse-save").onclick = () => {
        // Default the filename to a kebab-case of the topic when none is set.
        if (!$("output").value.trim()) {
            const slug = kebab($("topic").value);
            if (slug) { $("output").value = slug + ".json"; validateNewRun(); }
        }
        openPicker("save", "output", "Choose where to save");
    };
    $("browse-save-confirm").onclick = confirmSave;
    $("browse-filename").addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); confirmSave(); }
    });
    dialog.querySelector('button[rel="prev"]').onclick = () => closeModal("browse-dialog");
    dialog.addEventListener("click", (e) => { if (e.target === dialog) closeModal("browse-dialog"); });
    dialog.addEventListener("cancel", (e) => { e.preventDefault(); closeModal("browse-dialog"); });
    // Config editor modal.
    const cfgDialog = $("config-dialog");
    // The server-rendered form is the initial committed state.
    appliedConfig = snapshotConfigForm();
    // Re-fetch key status on open and (debounced) on input, since typing a
    // model changes which providers the config implicates.
    $("open-config").onclick = async () => {
        openModal("config-dialog");
        refreshConfigHighlights();
        await refreshKeyStatus();
        refreshConfigHighlights();
    };
    const refetchKeys = debounce(async () => { await refreshKeyStatus(); refreshConfigHighlights(); }, 400);
    $("config-form").addEventListener("input", () => { refreshConfigHighlights(); refetchKeys(); });
    $("config-reset").onclick = resetConfigToDefaults;
    $("config-save").onclick = saveConfigDefault;
    // Apply commits the edits; dismissing (×, backdrop, Esc) reverts to the
    // last committed state so an abandoned edit never affects a run.
    $("config-apply").onclick = commitConfig;
    cfgDialog.querySelector('button[rel="prev"]').onclick = dismissConfig;
    cfgDialog.addEventListener("click", (e) => { if (e.target === cfgDialog) dismissConfig(); });
    cfgDialog.addEventListener("cancel", (e) => { e.preventDefault(); dismissConfig(); });
    refreshConfigHighlights();
    // Warm the status so the button's badge is correct before any open.
    refreshKeyStatus().then(refreshConfigHighlights);
    // API-keys modal.
    const keysDialog = $("keys-dialog");
    $("open-keys").onclick = openKeys;
    $("keys-apply").onclick = applyKeys;
    $("keys-add").onclick = addProviderRow;
    keysDialog.querySelector('button[rel="prev"]').onclick = () => closeModal("keys-dialog");
    keysDialog.addEventListener("click", (e) => { if (e.target === keysDialog) closeModal("keys-dialog"); });
    keysDialog.addEventListener("cancel", (e) => { e.preventDefault(); closeModal("keys-dialog"); });
}

// kebab-case a topic for a default output filename.
function kebab(text) {
    return (text || "").toLowerCase().trim()
        .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

document.addEventListener("DOMContentLoaded", init);
