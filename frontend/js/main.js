/**
 * main.js — Jatayu AFDRN application entry point.
 *
 * State machine:  upload  →  processing  →  dashboard
 */

import { uploadCSV, fetchSummary, fetchAudit, fetchAgentOutputs, fetchIntelligence, streamIntelligence, openProgressSocket } from "./api.js";
import { initNav, setActiveTab } from "./nav.js";
import { initGraph } from "./graph.js";
import { renderCharts } from "./charts.js";

// ── App state ─────────────────────────────────────────────────────────────────
const state = {
  view: "upload",
  taskId: null,
  summaryData: null,
  // Persistent intelligence output — survive tab switches within a session
  intelligenceResult: null,      // final structured payload
  intelligenceStreaming: false,  // true while SSE is active
  intelligenceAbort: null,       // close() fn from streamIntelligence
};


// ── DOM refs ──────────────────────────────────────────────────────────────────
const views = {
  upload: document.getElementById("view-upload"),
  processing: document.getElementById("view-processing"),
  dashboard: document.getElementById("view-dashboard"),
};
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const progressPct = document.getElementById("progress-pct");
const progressMsg = document.getElementById("progress-msg");
const progressBar = document.getElementById("progress-bar");
const progressRing = document.getElementById("progress-ring-circle");
const toastContainer = document.getElementById("toast-container");

// ── View transitions ──────────────────────────────────────────────────────────
function showView(name) {
  state.view = name;
  Object.entries(views).forEach(([k, el]) => {
    if (!el) return;
    el.classList.toggle("hidden", k !== name);
    el.classList.toggle("opacity-0", k !== name);
  });
}

// ── Toast notifications ───────────────────────────────────────────────────────
function showToast(message, type = "error") {
  const colors = {
    error: "border-red-500/40 bg-red-500/10 text-red-300",
    success: "border-green-500/40 bg-green-500/10 text-green-300",
    info: "border-indigo-500/40 bg-indigo-500/10 text-indigo-300",
  };
  const toast = document.createElement("div");
  toast.className =
    `toast glass px-4 py-3 text-sm font-medium ${colors[type] || colors.error} ` +
    "flex items-center gap-2 max-w-sm";
  toast.innerHTML = `<i class="ph ph-${type === "error" ? "warning-circle" : "check-circle"} text-lg"></i>${message}`;
  toastContainer.appendChild(toast);
  setTimeout(() => { toast.classList.add("hide"); setTimeout(() => toast.remove(), 300); }, 4000);
}

// ── Progress ring helper ──────────────────────────────────────────────────────
function updateRing(pct) {
  if (!progressRing) return;
  const r = 54;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (pct / 100) * circumference;
  progressRing.style.strokeDasharray = circumference;
  progressRing.style.strokeDashoffset = offset;
}

// ── Upload handling ───────────────────────────────────────────────────────────
async function handleFile(file) {
  if (!file || !file.name.endsWith(".csv")) {
    showToast("Please upload a CSV file.", "error");
    return;
  }

  showView("processing");
  progressMsg.textContent = "Uploading file…";
  updateRing(0);

  let taskId;
  try {
    const res = await uploadCSV(file);
    taskId = res.task_id;
    state.taskId = taskId;
  } catch (err) {
    showView("upload");
    showToast(err.message || "Upload failed", "error");
    return;
  }

  openProgressSocket(taskId, {
    onProgress({ progress, message, processed, total }) {
      const pct = Math.round(progress || 0);
      progressPct.textContent = `${pct}%`;
      progressMsg.textContent = message || "Processing…";
      if (progressBar) progressBar.style.width = `${pct}%`;
      updateRing(pct);
      if (processed != null && total != null) {
        const counter = document.getElementById("progress-counter");
        if (counter) counter.textContent = `${processed} / ${total} transactions`;
      }
    },
    async onComplete() {
      updateRing(100);
      progressPct.textContent = "100%";
      progressMsg.textContent = "Analysis complete!";
      await new Promise((r) => setTimeout(r, 800));
      await loadDashboard(taskId);
    },
    onError(msg) {
      showView("upload");
      showToast(msg || "Processing error", "error");
    },
  });
}

// Dropzone events
dropzone?.addEventListener("click", () => fileInput?.click());
browseBtn?.addEventListener("click", (e) => { e.stopPropagation(); fileInput?.click(); });
fileInput?.addEventListener("change", (e) => handleFile(e.target.files?.[0]));
dropzone?.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone?.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  handleFile(e.dataTransfer.files?.[0]);
});

// ── Dashboard population ──────────────────────────────────────────────────────
async function loadDashboard(taskId) {
  try {
    const data = await fetchSummary(taskId);
    state.summaryData = data;
    populateDashboard(data);
    showView("dashboard");
    initNav((tab) => onTabActivated(tab, data));
    onTabActivated("dashboard", data);
    showToast("Analysis complete!", "success");

    // Wire refresh button for intelligence tab
    const refreshBtn = document.getElementById("intel-refresh-btn");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", () => {
        renderIntelligence(state.taskId, true);
      });
    }

    // Fetch agent outputs in background for accordion
    fetchAgentOutputs(taskId)
      .then((ao) => populateAgentAccordion(ao))
      .catch((err) => console.warn("Agent outputs unavailable:", err));
  } catch (err) {
    showToast("Failed to load dashboard: " + err.message, "error");
    showView("upload");
  }
}


function onTabActivated(tab, data) {
  if (tab === "cosmos") initGraph("graph-container", data.graph || { nodes: [], links: [] });
  if (tab === "analytics") renderCharts(data.metrics || {});
  if (tab === "intelligence") renderIntelligence(state.taskId);
  if (tab === "audit") loadAuditTab(state.taskId);
}

// ── Dashboard — metric cards ──────────────────────────────────────────────────
function populateDashboard(data) {
  const m = data.metrics || {};
  _setText("metric-total", (m.total_transactions || 0).toLocaleString());
  _setText("metric-fraud", (m.fraud_count || 0).toLocaleString());
  _setText("metric-rate", `${m.fraud_rate || 0}%`);
  _setText("metric-amount", `$${(m.fraud_amount || 0).toLocaleString()}`);
  _setText("metric-risk", m.avg_risk_score?.toFixed(3) || "0.000");
  _setText("metric-volume", `$${(m.total_amount || 0).toLocaleString()}`);
}

// ── Agent Pipeline Accordion ──────────────────────────────────────────────────

/**
 * Config for each accordion panel.
 * renderer(rows) → HTML string for the scrollable table body.
 */
const AGENT_CONFIG = [
  {
    key: "agent1",
    num: "1",
    label: "Transaction Monitoring",
    subtitle: "XGBoost fraud scoring",
    icon: "magnifying-glass",
    accent: "indigo",
    renderer: renderAgent1,
  },
  {
    key: "agent2",
    num: "2",
    label: "Pattern Detection",
    subtitle: "Mule / ATO / Velocity",
    icon: "brain",
    accent: "purple",
    renderer: renderAgent2,
  },
  {
    key: "agent3",
    num: "3",
    label: "Risk Assessment",
    subtitle: "Rules · PPO-ready",
    icon: "scales",
    accent: "blue",
    renderer: renderAgent3,
  },
  {
    key: "agent4",
    num: "4",
    label: "Alert & Block",
    subtitle: "Enforcement execution",
    icon: "shield-warning",
    accent: "orange",
    renderer: renderAgent4,
  },
  {
    key: "agent5",
    num: "5",
    label: "Compliance Logging",
    subtitle: "Immutable audit trail",
    icon: "clipboard-text",
    accent: "green",
    renderer: renderAgent5,
  },
];

function populateAgentAccordion(agentOutputs) {
  const container = document.getElementById("agent-accordion");
  if (!container) return;

  // Show loading skeleton while data arrives
  const loader = document.getElementById("pipeline-loading");
  if (loader) loader.classList.add("hidden");

  const totalTxns = agentOutputs.total || 0;
  container.innerHTML = "";

  AGENT_CONFIG.forEach((cfg, idx) => {
    const rows = agentOutputs[cfg.key] || [];
    const panelId = `accordion-body-${cfg.key}`;
    const accentMap = {
      indigo: { ring: "bg-indigo-500/20", text: "text-indigo-400", dot: "bg-indigo-400" },
      purple: { ring: "bg-purple-500/20", text: "text-purple-400", dot: "bg-purple-400" },
      blue: { ring: "bg-blue-500/20", text: "text-blue-400", dot: "bg-blue-400" },
      orange: { ring: "bg-orange-500/20", text: "text-orange-400", dot: "bg-orange-400" },
      green: { ring: "bg-green-500/20", text: "text-green-400", dot: "bg-green-400" },
    };
    const ac = accentMap[cfg.accent] || accentMap.indigo;

    const item = document.createElement("div");
    item.className = "rounded-xl overflow-hidden border border-white/[0.06]";
    item.innerHTML = `
      <button
        class="agent-accordion-btn w-full flex items-center gap-3 px-3 py-2.5 rounded-xl"
        data-panel="${panelId}"
        aria-expanded="false"
      >
        <div class="w-7 h-7 rounded-lg ${ac.ring} flex items-center justify-center flex-shrink-0">
          <i class="ph ph-${cfg.icon} ${ac.text} text-base"></i>
        </div>
        <div class="flex-1 text-left min-w-0">
          <p class="text-slate-200 text-xs font-semibold leading-tight">${_esc(cfg.label)}</p>
          <p class="text-slate-500 text-[10px] leading-tight">${_esc(cfg.subtitle)}</p>
        </div>
        <div class="flex items-center gap-2 flex-shrink-0">
          <span class="${ac.text} text-[10px] font-medium">${rows.length} txns</span>
          <i id="chevron-${cfg.key}" class="ph ph-caret-down ${ac.text} text-xs transition-transform duration-300"></i>
        </div>
      </button>
      <div id="${panelId}" class="agent-accordion-body">
        <div class="agent-accordion-scroll px-1 pb-2">
          ${rows.length ? cfg.renderer(rows) : `<p class="text-slate-500 text-xs text-center py-4">No data available.</p>`}
        </div>
      </div>
    `;
    container.appendChild(item);

    // Toggle open/close
    const btn = item.querySelector("button");
    btn.addEventListener("click", () => {
      const body = document.getElementById(panelId);
      const chevron = document.getElementById(`chevron-${cfg.key}`);
      const isOpen = body.classList.contains("open");

      // Close all others
      container.querySelectorAll(".agent-accordion-body").forEach((b) => b.classList.remove("open"));
      container.querySelectorAll("[id^='chevron-']").forEach((c) => c.classList.remove("rotate-180"));
      container.querySelectorAll(".agent-accordion-btn").forEach((b) => b.setAttribute("aria-expanded", "false"));

      if (!isOpen) {
        body.classList.add("open");
        if (chevron) chevron.classList.add("rotate-180");
        btn.setAttribute("aria-expanded", "true");
      }
    });
  });
}

// ── Per-agent row renderers ────────────────────────────────────────────────────

function _th(...cols) {
  return `<thead><tr class="border-b border-white/[0.06] text-[10px] text-slate-500 uppercase tracking-wide">
    ${cols.map(c => `<th class="py-1.5 px-2 text-left font-medium">${c}</th>`).join("")}
  </tr></thead>`;
}

function _table(head, body) {
  return `<table class="w-full text-[11px] border-collapse">${head}<tbody>${body}</tbody></table>`;
}

function _scoreBar(score) {
  if (score == null) return "—";
  const pct = Math.round(score * 100);
  const color = score >= 0.5 ? "#ef4444" : score >= 0.2 ? "#f97316" : "#6366f1";
  return `
    <div class="flex items-center gap-1.5">
      <div class="score-bar-track flex-1">
        <div class="score-bar-fill" style="width:${pct}%; background:${color};"></div>
      </div>
      <span style="color:${color}; min-width:28px; text-align:right;">${pct}%</span>
    </div>`;
}

function _datasetSignal(influence = {}) {
  const ratio = influence.threshold_ratio;
  const mode = influence.artifact_mode || "unknown";
  const gnn = influence.gnn_embedding || {};
  const label = ratio != null ? `${Number(ratio).toFixed(1)}x` : mode;
  const title = [
    `Artifact: ${mode}`,
    influence.model_version ? `Version: ${influence.model_version}` : "",
    influence.decision_threshold != null ? `Threshold: ${Number(influence.decision_threshold).toFixed(4)}` : "",
    `GNN embedding: ${gnn.used ? "matched" : "not matched"}`,
  ].filter(Boolean).join(" | ");
  return `<span title="${_esc(title)}" class="text-[10px] text-indigo-300 font-medium whitespace-nowrap">${_esc(label)}</span>`;
}

function _badge(val, type = "risk") {
  if (!val) return "—";
  const v = String(val).toLowerCase();
  const classMap = {
    critical: "badge-critical", high: "badge-high", medium: "badge-medium", low: "badge-low",
    block: "badge-block", hold: "badge-hold", silent_flag: "badge-flag", pass: "badge-pass",
    none: "badge-none", mule_network: "badge-block", account_takeover: "badge-high",
    velocity_spike: "badge-hold",
  };
  const cls = classMap[v] || "badge-none";
  return `<span class="badge ${cls}">${_esc(val)}</span>`;
}

function renderAgent1(rows) {
  const head = _th("Txn", "Origin", "Type", "$", "Score", "Dataset", "Flag");
  const body = rows.map(r => `
    <tr class="border-b border-white/[0.04] hover:bg-white/[0.025]">
      <td class="py-1 px-2 font-mono text-slate-400">${_esc(r.txn_id)}…</td>
      <td class="py-1 px-2 text-slate-300 max-w-[82px] truncate">${_esc(r.nameOrig)}</td>
      <td class="py-1 px-2 text-slate-400">${_esc(r.type || "—")}</td>
      <td class="py-1 px-2 text-slate-300">$${Number(r.amount || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
      <td class="py-1 px-2 min-w-[90px]">${_scoreBar(r.fraud_score)}</td>
      <td class="py-1 px-2">${_datasetSignal(r.dataset_influence)}</td>
      <td class="py-1 px-2">${r.fraud_label ? '<span class="text-red-400 font-medium">FRAUD</span>' : '<span class="text-green-400">CLEAN</span>'}</td>
    </tr>
  `).join("");
  return _table(head, body);
}

function renderAgent2(rows) {
  const head = _th("Txn", "Origin", "Pattern", "Conf", "Reasoning");
  const body = rows.map(r => `
    <tr class="border-b border-white/[0.04] hover:bg-white/[0.025]">
      <td class="py-1 px-2 font-mono text-slate-400">${_esc(r.txn_id)}…</td>
      <td class="py-1 px-2 text-slate-300 max-w-[80px] truncate">${_esc(r.nameOrig)}</td>
      <td class="py-1 px-2">${_badge(r.pattern_type)}</td>
      <td class="py-1 px-2 text-slate-400">${r.pattern_confidence != null ? (r.pattern_confidence * 100).toFixed(0) + "%" : "—"}</td>
      <td class="py-1 px-2 text-slate-400 max-w-[180px]">
        ${r.pattern_reasoning
      ? `<span title="${_esc(r.pattern_reasoning)}" class="truncate block italic">"${_esc(r.pattern_reasoning.slice(0, 60))}${r.pattern_reasoning.length > 60 ? "…" : ""}"</span>`
      : '<span class="text-slate-600">—</span>'}
      </td>
    </tr>
  `).join("");
  return _table(head, body);
}

function renderAgent3(rows) {
  const head = _th("Txn", "Origin", "Score", "Model x", "Pattern", "Risk", "Action");
  const body = rows.map(r => `
    <tr class="border-b border-white/[0.04] hover:bg-white/[0.025]">
      <td class="py-1 px-2 font-mono text-slate-400">${_esc(r.txn_id)}…</td>
      <td class="py-1 px-2 text-slate-300 max-w-[80px] truncate">${_esc(r.nameOrig)}</td>
      <td class="py-1 px-2 min-w-[90px]">${_scoreBar(r.fraud_score)}</td>
      <td class="py-1 px-2 text-indigo-300 text-[10px]">${r.account_context?.model_score_ratio != null ? Number(r.account_context.model_score_ratio).toFixed(1) + "x" : "—"}</td>
      <td class="py-1 px-2">${_badge(r.pattern_type)}</td>
      <td class="py-1 px-2">${_badge(r.risk_level)}</td>
      <td class="py-1 px-2">${_badge(r.recommended_action)}</td>
    </tr>
  `).join("");
  return _table(head, body);
}

function renderAgent4(rows) {
  const head = _th("Txn", "Origin", "Amount", "Risk", "Action", "Explanation");
  const body = rows.map(r => `
    <tr class="border-b border-white/[0.04] hover:bg-white/[0.025]">
      <td class="py-1 px-2 font-mono text-slate-400">${_esc(r.txn_id)}…</td>
      <td class="py-1 px-2 text-slate-300 max-w-[80px] truncate">${_esc(r.nameOrig)}</td>
      <td class="py-1 px-2 text-slate-300">$${Number(r.amount || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
      <td class="py-1 px-2">${_badge(r.risk_level)}</td>
      <td class="py-1 px-2">${_badge(r.action_taken)}</td>
      <td class="py-1 px-2 text-slate-400 max-w-[200px]">
        <span title="${_esc(r.explanation)}" class="truncate block">${_esc((r.explanation || "—").slice(0, 55))}${(r.explanation || "").length > 55 ? "…" : ""}</span>
      </td>
    </tr>
  `).join("");
  return _table(head, body);
}

function renderAgent5(rows) {
  const head = _th("Txn", "Origin", "Risk", "Action", "Latency", "Agents");
  const body = rows.map(r => {
    const latMs = r.decision_latency_ms != null ? `${r.decision_latency_ms.toFixed(1)} ms` : "—";
    const agentStatuses = (r.pipeline_metadata || []).map(m => {
      const statusColor = m.status === "ok" ? "text-green-400" : m.status === "fallback" ? "text-yellow-400" : "text-red-400";
      const short = (m.agent_name || "").replace("Agent", "A").replace("Transaction Monitoring", "TxnMon").replace("PatternDetection", "PatDet").replace("RiskAssessment", "Risk").replace("AlertBlock", "Alert").replace("ComplianceLogging", "Comply");
      return `<span class="${statusColor}" title="${_esc(m.agent_name)}: ${_esc(m.status)} (${m.latency_ms ? m.latency_ms.toFixed(1) : "—"}ms)">${_esc(short.slice(0, 4))}</span>`;
    }).join(" · ");
    return `
      <tr class="border-b border-white/[0.04] hover:bg-white/[0.025]">
        <td class="py-1 px-2 font-mono text-slate-400">${_esc(r.txn_id)}…</td>
        <td class="py-1 px-2 text-slate-300 max-w-[80px] truncate">${_esc(r.nameOrig)}</td>
        <td class="py-1 px-2">${_badge(r.risk_level)}</td>
        <td class="py-1 px-2">${_badge(r.action_taken)}</td>
        <td class="py-1 px-2 text-slate-400 whitespace-nowrap">${latMs}</td>
        <td class="py-1 px-2 text-[10px] space-x-0.5">${agentStatuses || "<span class='text-slate-600'>—</span>"}</td>
      </tr>
    `;
  }).join("");
  return _table(head, body);
}

// ── Intelligence tab — LLM Reasoning (streaming + persistent) ─────────────────────
function renderIntelligence(taskId, forceRefresh = false) {
  const container = document.getElementById("intelligence-content");
  if (!container) return;

  // ── Already have a cached result ───────────────────────────────────────────
  if (state.intelligenceResult && !forceRefresh) {
    _renderIntelligenceResult(container, state.intelligenceResult);
    return;
  }

  // ── Already streaming (tab switched away & back) ────────────────────────────
  if (state.intelligenceStreaming && !forceRefresh) return;

  // ── Force refresh — abort any in-flight stream ───────────────────────────────
  if (forceRefresh) {
    state.intelligenceAbort?.();
    state.intelligenceResult = null;
    state.intelligenceStreaming = false;
  }

  // ── Show streaming skeleton ─────────────────────────────────────────────────────
  container.innerHTML = `
    <div class="col-span-full glass p-5 border border-indigo-500/20 bg-indigo-500/5">
      <div class="flex items-center gap-3 mb-3">
        <div class="relative w-8 h-8">
          <div class="absolute inset-0 rounded-full border-2 border-indigo-500/40 animate-ping"></div>
          <div class="relative w-8 h-8 glass rounded-full flex items-center justify-center">
            <i class="ph ph-brain text-indigo-400 text-sm"></i>
          </div>
        </div>
        <div>
          <p class="text-indigo-300 text-xs font-semibold uppercase tracking-wider">LLM Intelligence Report</p>
          <p class="text-slate-500 text-[11px]">LFM2.5-1.2B-Instruct · Streaming from localhost:8080…</p>
        </div>
        <span id="intel-stream-badge" class="ml-auto px-2 py-0.5 rounded-full text-[10px] border border-indigo-500/30 bg-indigo-500/10 text-indigo-300 animate-pulse">Generating…</span>
      </div>
      <div id="intel-stream-output"
           class="font-mono text-sm text-slate-200 leading-relaxed whitespace-pre-wrap break-words
                  max-h-64 overflow-y-auto rounded-lg bg-black/30 p-3 border border-white/[0.06]"
      ></div>
    </div>
  `;

  const streamOutput = document.getElementById("intel-stream-output");
  state.intelligenceStreaming = true;

  state.intelligenceAbort = streamIntelligence(taskId, {
    onToken(text) {
      if (streamOutput) {
        streamOutput.textContent += text;
        // Auto-scroll to bottom
        streamOutput.scrollTop = streamOutput.scrollHeight;
      }
    },
    onComplete(data) {
      state.intelligenceStreaming = false;
      state.intelligenceResult = data;
      state.intelligenceAbort = null;
      _renderIntelligenceResult(container, data);
    },
    onCached(data) {
      state.intelligenceStreaming = false;
      state.intelligenceResult = data;
      state.intelligenceAbort = null;
      _renderIntelligenceResult(container, data);
    },
    onError(msg, fallbackData) {
      state.intelligenceStreaming = false;
      state.intelligenceAbort = null;
      if (fallbackData) {
        state.intelligenceResult = fallbackData;
        _renderIntelligenceResult(container, fallbackData);
      } else {
        container.innerHTML = `
          <div class="col-span-full glass p-6 flex flex-col items-center gap-3">
            <i class="ph ph-warning-circle text-red-400 text-3xl"></i>
            <p class="text-red-300 font-medium">${_esc(msg)}</p>
          </div>`;
      }
    },
  });
}

function _renderIntelligenceResult(container, data) {
  const analysis = data.analysis;
  const meta = data.meta || {};
  const samples = data.high_risk_samples || [];
  const error = data.error;

  const confidenceColor = {
    HIGH: "text-green-400 bg-green-500/10 border-green-500/30",
    MEDIUM: "text-yellow-400 bg-yellow-500/10 border-yellow-500/30",
    LOW: "text-orange-400 bg-orange-500/10 border-orange-500/30",
  };

  if (analysis) {
    const confClass = confidenceColor[analysis.confidence] || confidenceColor.LOW;
    container.innerHTML = `
      <div class="col-span-full glass p-4 border border-indigo-500/20 bg-indigo-500/5 flex items-start gap-3">
        <div class="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
          <i class="ph ph-robot text-indigo-400"></i>
        </div>
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 flex-wrap mb-1">
            <span class="text-indigo-300 text-xs font-semibold uppercase tracking-wider">LLM Intelligence Report</span>
            <span class="px-2 py-0.5 rounded-full text-xs border font-medium ${confClass}">${analysis.confidence || "UNKNOWN"} Confidence</span>
            <span class="text-slate-500 text-xs">LFM2.5-1.2B-Instruct · localhost:8080</span>
            <span class="ml-auto text-green-400 text-xs flex items-center gap-1"><i class="ph ph-check-circle"></i> Cached for session</span>
          </div>
          <p class="text-slate-200 text-sm leading-relaxed">${_esc(analysis.headline || "")}</p>
        </div>
      </div>

      <div class="col-span-full md:col-span-1 glass p-5">
        <h3 class="text-white font-semibold text-sm mb-3 flex items-center gap-2">
          <i class="ph ph-magnifying-glass-plus text-purple-400"></i> Key Findings
        </h3>
        <ul class="space-y-2">
          ${(analysis.key_findings || []).map((f, i) => `
            <li class="flex items-start gap-2.5">
              <span class="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">${i + 1}</span>
              <span class="text-slate-300 text-sm leading-snug">${_esc(f)}</span>
            </li>
          `).join("")}
        </ul>
      </div>

      <div class="col-span-full md:col-span-1 glass p-5">
        <h3 class="text-white font-semibold text-sm mb-3 flex items-center gap-2">
          <i class="ph ph-shield-warning text-red-400"></i> Threat Assessment
        </h3>
        <p class="text-slate-300 text-sm leading-relaxed">${_esc(analysis.threat_assessment || "No threat assessment available.")}</p>
        <div class="mt-4 grid grid-cols-2 gap-2">
          <div class="glass p-3 bg-red-500/5 border-red-500/20">
            <p class="text-red-400 text-xs font-medium">Fraud Rate</p>
            <p class="text-white font-bold text-lg">${meta.fraud_rate || 0}%</p>
          </div>
          <div class="glass p-3 bg-orange-500/5 border-orange-500/20">
            <p class="text-orange-400 text-xs font-medium">High Risk</p>
            <p class="text-white font-bold text-lg">${(meta.risk_counts?.HIGH || 0) + (meta.risk_counts?.CRITICAL || 0)}</p>
          </div>
        </div>
      </div>

      <div class="col-span-full md:col-span-1 glass p-5">
        <h3 class="text-white font-semibold text-sm mb-3 flex items-center gap-2">
          <i class="ph ph-lightbulb text-yellow-400"></i> Recommendations
        </h3>
        <ul class="space-y-2">
          ${(analysis.recommendations || []).map(r => `
            <li class="flex items-start gap-2">
              <i class="ph ph-arrow-right text-yellow-400 text-sm mt-0.5 flex-shrink-0"></i>
              <span class="text-slate-300 text-sm leading-snug">${_esc(r)}</span>
            </li>
          `).join("")}
        </ul>
      </div>

      ${samples.length ? `
      <div class="col-span-full glass p-5">
        <h3 class="text-white font-semibold text-sm mb-3 flex items-center gap-2">
          <i class="ph ph-fire text-red-400"></i> High-Risk Transactions Analysed
        </h3>
        <div class="space-y-2">
          ${samples.map(s => `
            <div class="glass p-3 bg-red-500/5 border border-red-500/10 flex flex-wrap items-start gap-3">
              <span class="font-mono text-xs text-slate-400">${_esc(s.transaction_id)}…</span>
              <span class="badge badge-${(s.risk_level || "low").toLowerCase()}">${s.risk_level || "—"}</span>
              <span class="text-slate-300 text-xs">$${(s.amount || 0).toLocaleString()}</span>
              <span class="text-slate-400 text-xs">${s.pattern_type || "NONE"}</span>
              <span class="text-slate-400 text-xs">Score: ${(s.fraud_score || 0).toFixed(3)}</span>
              ${s.explanation ? `<p class="w-full text-slate-400 text-xs italic mt-1">"${_esc(s.explanation)}"</p>` : ""}
            </div>
          `).join("")}
        </div>
      </div>` : ""}
    `;
  } else {
    // LLM unavailable — show pattern metrics fallback
    const pc = meta.pattern_counts || {};
    container.innerHTML = `
      <div class="col-span-full glass p-4 border border-orange-500/20 bg-orange-500/5 flex items-start gap-3 mb-2">
        <i class="ph ph-warning text-orange-400 text-xl flex-shrink-0 mt-0.5"></i>
        <div>
          <p class="text-orange-300 text-sm font-semibold">LLM Service Unavailable</p>
          <p class="text-slate-400 text-xs mt-0.5">${_esc(error || "Could not connect to llama-server on port 8080. Showing pattern metrics instead.")}</p>
        </div>
      </div>
      ${["MULE_NETWORK", "ACCOUNT_TAKEOVER", "VELOCITY_SPIKE", "NONE"].map(key => {
      const cfg = {
        MULE_NETWORK: { label: "Mule Network", icon: "graph", color: "text-purple-400", bg: "bg-purple-500/10 border-purple-500/20" },
        ACCOUNT_TAKEOVER: { label: "Account Takeover", icon: "user-circle-gear", color: "text-red-400", bg: "bg-red-500/10 border-red-500/20" },
        VELOCITY_SPIKE: { label: "Velocity Spike", icon: "lightning", color: "text-orange-400", bg: "bg-orange-500/10 border-orange-500/20" },
        NONE: { label: "Clean", icon: "check-circle", color: "text-green-400", bg: "bg-green-500/10 border-green-500/20" },
      }[key];
      return `
          <div class="glass p-5 border ${cfg.bg}">
            <div class="flex items-start gap-3">
              <i class="ph ph-${cfg.icon} text-2xl ${cfg.color} mt-0.5"></i>
              <div class="flex-1">
                <h3 class="text-white font-semibold text-sm">${cfg.label}</h3>
                <p class="text-3xl font-bold ${cfg.color} mt-1">${(pc[key] || 0).toLocaleString()}</p>
                <p class="text-slate-400 text-xs mt-1">${key === "NONE" ? "Clean transactions" : "Detected incidents"}</p>
              </div>
            </div>
          </div>`;
    }).join("")}
    `;
  }
}

// ── Audit tab ──────────────────────────────────────────────────────────────────
async function loadAuditTab(taskId) {
  const container = document.getElementById("audit-timeline");
  if (!container) return;
  container.innerHTML = '<p class="text-slate-500 text-sm">Loading…</p>';
  try {
    const { records } = await fetchAudit(taskId);
    if (!records.length) {
      container.innerHTML = '<p class="text-slate-500 text-sm">No audit records yet.</p>';
      return;
    }
    container.innerHTML = records.slice(0, 100).map((entry) => {
      const record = entry.record || {};
      return `
        <div class="flex gap-4 group">
          <div class="flex flex-col items-center">
            <div class="w-2.5 h-2.5 rounded-full bg-indigo-500 mt-1 flex-shrink-0 group-hover:bg-indigo-300 transition-colors"></div>
            <div class="w-px flex-1 bg-white/10 mt-1"></div>
          </div>
          <div class="pb-5 flex-1">
            <p class="text-xs text-slate-500">${entry.created_at}</p>
            <p class="text-slate-200 text-sm font-medium mt-0.5">${record.transaction_id || "Unknown transaction"}</p>
            <p class="text-slate-400 text-xs mt-1">${record.action_taken || record.action || "—"} · ${record.risk_level || "—"}</p>
          </div>
        </div>`;
    }).join("");
  } catch (err) {
    container.innerHTML = `<p class="text-red-400 text-sm">${err.message}</p>`;
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────
function _setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function _esc(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Init ──────────────────────────────────────────────────────────────────────
showView("upload");
