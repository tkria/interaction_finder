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
// Which stage a given event scope belongs to.
const SCOPE_STAGE = {
    "keywords.scored": "keywords",
    "search.queries": "search",
    "search.results": "search",
    "search.selected": "search",
    "extraction.pair_judged": "extraction",
};

let evtSource = null;
let model = null;   // per-run accumulated state

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
    if (evtSource) { evtSource.close(); evtSource = null; }
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
function initModel(runId) {
    model = { stages: {}, activeStage: null, feed: [], lastIndex: -1, runId };
    for (const s of STAGES) {
        model.stages[s] = { status: "pending", terms: [], queries: {}, pairs: [], summary: "" };
    }
}

// Start a live run: hide the form, show the activity card + stage tabs, stream.
function beginRun(info) {
    initModel(info.run_id);
    $("new-run").dataset.running = "1";
    $("report-card").hidden = true;
    $("run-summary").dataset.active = "1";
    $("activity").dataset.active = "1";
    $("stages").dataset.active = "1";
    $("summary-text").textContent = info.topic;
    // Reset the activity card to its running state (spinner + feed).
    $("activity-done").hidden = true;
    $("activity-running").hidden = false;
    $("activity-status").setAttribute("aria-busy", "true");
    $("activity-status").textContent = "Starting…";
    $("activity-feed").innerHTML = "";
    const cancel = $("cancel-run");
    cancel.hidden = false;
    cancel.disabled = false;
    cancel.removeAttribute("aria-busy");
    renderTabs();
    selectTab(STAGES[0]);
    connect(info.run_id);
}

// Show a loaded checkpoint: keep the form, render its back-filled stage info
// (no activity card), and surface the report card. The SSE stream replays the
// back-filled events into the stage tabs, then immediately reports done.
function showLoaded(info) {
    initModel(info.run_id);
    $("stages").dataset.active = "1";
    $("activity").dataset.active = "";     // no live activity for a loaded file
    $("run-summary").dataset.active = "";
    renderTabs();
    selectTab(STAGES[0]);
    connect(info.run_id);
    renderReportCard(info);
}

function connect(runId) {
    if (evtSource) evtSource.close();
    evtSource = new EventSource(`/runs/${runId}/events?since=${model.lastIndex + 1}`);
    evtSource.onmessage = (e) => handleMessage(JSON.parse(e.data));
    evtSource.onerror = () => { /* browser auto-reconnects; since= resumes */ };
}

function handleMessage(msg) {
    if (msg.type === "event") {
        model.lastIndex = Math.max(model.lastIndex, msg.index);
        applyEvent(msg);
    } else if (msg.type === "status") {
        $("activity-status").textContent = msg.status || "";
        $("activity-elapsed").textContent = msg.elapsed || "";
    } else if (msg.type === "counters") {
        applyCounters(msg.counters);
    } else if (msg.type === "done") {
        finishRun(msg);
    }
}

function applyEvent(msg) {
    pushFeed(msg.description);
    if (msg.scope === "meta.cleared") {
        // Current stage ended; finalize it. Next stage is identified by the
        // next non-meta event's scope.
        if (model.activeStage) { model.stages[model.activeStage].status = "done"; }
        model.activeStage = null;
        renderTabs();
        return;
    }
    const stage = SCOPE_STAGE[msg.scope];
    if (!stage) return;
    if (model.activeStage !== stage) {
        model.activeStage = stage;
        model.stages[stage].status = "active";
        selectTab(stage);
        renderTabs();
    }
    const st = model.stages[stage];
    if (msg.scope === "keywords.scored") {
        st.terms = msg.data.terms || [];
    } else if (msg.scope === "search.selected") {
        st.queries[msg.data.query] = msg.data;
    } else if (msg.scope === "extraction.pair_judged") {
        st.pairs.push(msg.data);
    }
    if (stage === currentTab) renderPanel(stage);
}

function applyCounters(counters) {
    // Summarise the active stage's counters as a one-line header.
    if (!model.activeStage) return;
    const line = counters
        .filter((c) => c.status !== "unstarted")
        .map((c) => `${c.name} ${c.completed}${c.total ? "/" + c.total : ""}`)
        .join(" · ");
    model.stages[model.activeStage].summary = line;
    if (model.activeStage === currentTab) renderPanel(model.activeStage);
}

function pushFeed(text) {
    if (!text) return;
    model.feed.unshift(text);
    model.feed = model.feed.slice(0, 3);
    const ul = $("activity-feed");
    ul.innerHTML = "";
    for (const line of model.feed) {
        const li = document.createElement("li");
        li.textContent = line;
        ul.appendChild(li);
    }
}

// Ask the server to cancel the in-flight run. The terminal "done" (result
// "cancelled") arrives over the SSE stream and is handled by finishRun.
async function cancelRun() {
    if (!model || !model.runId) return;
    const btn = $("cancel-run");
    btn.disabled = true;
    btn.setAttribute("aria-busy", "true");
    try {
        await fetch(`/runs/${model.runId}/cancel`, { method: "POST" });
    } catch (e) {
        btn.disabled = false;
        btn.removeAttribute("aria-busy");
    }
}

function finishRun(msg) {
    if (evtSource) { evtSource.close(); evtSource = null; }
    // Any still-active stage is complete now.
    for (const s of STAGES) {
        if (model.stages[s].status === "active") model.stages[s].status = "done";
    }
    model.activeStage = null;
    renderTabs();
    // A loaded checkpoint renders its stage info with no activity card and no
    // run chrome -- the back-filled stream just ends; nothing more to collapse.
    if ($("activity").dataset.active !== "1") return;
    // Collapse the live progress block; the card becomes a single flush row.
    $("activity-status").removeAttribute("aria-busy");
    $("activity-running").hidden = true;
    const done = $("activity-done");
    const label = $("done-label");
    const btn = $("view-report");
    done.hidden = false;
    if (msg.result === "success") {
        label.textContent = "Completed";
        label.className = "ok";
        btn.hidden = false;
        btn.onclick = () => window.open(`/report/${model.runId}`, "_blank");
        const dl = $("download-report");
        dl.hidden = false;
        dl.onclick = () => downloadReport(model.runId);
    } else if (msg.result === "cancelled") {
        // Stages already finished keep a partial, resumable checkpoint on disk.
        label.textContent = "Cancelled — partial progress saved";
        label.className = "cancelled";
        btn.hidden = true;
    } else {
        label.textContent = "Failed" + (msg.error ? ` — ${msg.error}` : "");
        label.className = "err";
        btn.hidden = true;
    }
    $("new-run-again").hidden = false;
}

// === Tabs & panels ===
let currentTab = STAGES[0];

function renderTabs() {
    const bar = $("tab-bar");
    bar.innerHTML = "";
    for (const s of STAGES) {
        const status = model.stages[s].status;
        const btn = document.createElement("button");
        btn.dataset.status = status;
        btn.setAttribute("aria-selected", s === currentTab ? "true" : "false");
        if (status === "pending") btn.disabled = true;
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
        html += items || emptyNote("No search selections yet.");
    } else if (stage === "extraction") {
        const rows = st.pairs.map((p) =>
            `<li><span class="verdict-dot ${p.accepted ? "yes" : "no"}"></span>` +
            `<span class="grow"><strong>${escapeHtml(p.entity1)}</strong> ${escapeHtml(p.relationship || "")} <strong>${escapeHtml(p.entity2)}</strong></span>` +
            `<small class="num">ev ${p.evidence ?? "?"}</small></li>`
        ).join("");
        html += rows ? `<ul class="data-list">${rows}</ul>` : emptyNote("No judged pairs yet.");
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
    return `<details class="search-q"><summary>${icon}`
        + `<span class="grow">${escapeHtml(q.query)}</span>`
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

// Refresh the .non-default marks and per-section count badges. Called on load
// and on any input within the form.
function refreshConfigHighlights() {
    const inputs = $("config-form").querySelectorAll("input[name]");
    inputs.forEach((inp) => {
        // Agent llm inputs sit in a label.agent-head; everything else in a
        // .cfg-field. Highlight whichever wrapper this control has.
        const wrap = inp.closest(".cfg-field, .agent-head");
        if (wrap) wrap.classList.toggle("non-default", isNonDefault(inp));
    });
    // Each section's badge counts non-default fields anywhere beneath it.
    $("config-form").querySelectorAll(".cfg-section").forEach((sec) => {
        const n = [...sec.querySelectorAll("input[name]")].filter(isNonDefault).length;
        const badge = sec.querySelector(":scope > summary > .cfg-count");
        badge.textContent = n;
        badge.hidden = n === 0;
    });
    const total = [...inputs].filter(isNonDefault).length;
    $("config-summary").textContent = total
        ? `${total} field${total === 1 ? "" : "s"} changed from defaults`
        : "All defaults";
    // Mirror the count on the form's Configuration button so divergence from
    // defaults is visible without opening the modal.
    const badge = $("config-changed-badge");
    badge.textContent = total;
    badge.hidden = total === 0;
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
// Provider keys live in the server process for the session only. We fetch
// set/unset status (never the value) to render the rows, and POST any typed
// values. A set provider shows a jade dot; its input stays empty (typing a
// new value replaces it, leaving it blank keeps the existing key).
async function openKeys() {
    await renderProviderRows();
    openModal("keys-dialog");
}

async function renderProviderRows() {
    const box = $("keys-providers");
    box.innerHTML = "";
    let providers = [];
    try {
        providers = (await (await fetch("/keys")).json()).providers || [];
    } catch (e) { /* leave empty on failure */ }
    for (const p of providers) {
        const row = document.createElement("div");
        row.className = "key-row" + (p.set ? " is-set" : "");
        row.innerHTML =
            `<span class="key-label"><span class="key-dot"></span>${escapeHtml(p.label)}</span>` +
            `<input type="password" autocomplete="off" data-env="${p.env}" ` +
            `placeholder="${p.set ? "set — type to replace" : "not set"}">`;
        box.appendChild(row);
    }
}

async function applyKeys() {
    const keys = {};
    $("keys-providers").querySelectorAll("input[data-env]").forEach((inp) => {
        if (inp.value.trim()) keys[inp.dataset.env] = inp.value.trim();
    });
    const btn = $("keys-apply");
    btn.setAttribute("aria-busy", "true");
    try {
        await fetch("/keys", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ keys }),
        });
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
    $("cancel-run").onclick = cancelRun;
    $("new-run-again").onclick = () => {
        $("new-run").dataset.running = "";
        $("run-summary").dataset.active = "";
        $("activity").dataset.active = "";
        $("stages").dataset.active = "";
    };
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
    $("open-config").onclick = () => { refreshConfigHighlights(); openModal("config-dialog"); };
    $("config-form").addEventListener("input", refreshConfigHighlights);
    $("config-reset").onclick = resetConfigToDefaults;
    $("config-save").onclick = saveConfigDefault;
    // Apply commits the edits; dismissing (×, backdrop, Esc) reverts to the
    // last committed state so an abandoned edit never affects a run.
    $("config-apply").onclick = commitConfig;
    cfgDialog.querySelector('button[rel="prev"]').onclick = dismissConfig;
    cfgDialog.addEventListener("click", (e) => { if (e.target === cfgDialog) dismissConfig(); });
    cfgDialog.addEventListener("cancel", (e) => { e.preventDefault(); dismissConfig(); });
    refreshConfigHighlights();
    // API-keys modal.
    const keysDialog = $("keys-dialog");
    $("open-keys").onclick = openKeys;
    $("keys-apply").onclick = applyKeys;
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
