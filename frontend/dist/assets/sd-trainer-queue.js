/**
 * 训练队列 / Training queue panel.
 *
 * Injected on every page via sd-trainer-brand.js (both served no-cache).
 * Adds a 「队列」 sidebar group after 「训练」; clicking 「训练队列」 opens a
 * same-style fullscreen overlay — no route change, no page flicker.
 * Backend: /api/queue* (mikazuki/train_queue.py). While a task runs or an
 * entry is being edited, the page's 「开始训练」 button is relabeled because
 * POST /api/run then queues / saves instead of starting (server-side logic).
 */
(function () {
  if (window.__sdTrainerQueueLoaded) return;
  window.__sdTrainerQueueLoaded = true;

  const OVERLAY_ID = "sd-queue-overlay";
  const STYLE_ID = "sd-queue-style";
  const NAV_ID = "sd-queue-nav";
  const TOAST_ID = "sd-queue-toast";
  const POLL_MS = 3000;

  // model_train_type → training page that can edit it. The entry config is
  // handed over via sessionStorage["mikazuki-pending-import"], which the page
  // layout applies on mount through /api/config/validate-import (branch-const
  // stamping, network_args → UI field hydration, schema-default merge).
  const PAGE_MAP = {
    "anima-lora": { path: "/lora/sd3.html" },
    "sd3-lora": { path: "/lora/sd3.html" },
    "anima-finetune": { path: "/lora/anima-finetune.html" },
    "anima-lora-fast": { path: "/lora/anima-fast.html" },
    "flux-lora": { path: "/lora/flux.html" },
    "flux-finetune": { path: "/lora/flux.html" },
    "sd-lora": { path: "/lora/master.html" },
    "sdxl-lora": { path: "/lora/master.html" },
    "sdxl-finetune": { path: "/lora/master.html" },
    "sd-dreambooth": { path: "/dreambooth/" },
  };

  const STATUS_META = {
    queued: { label: "排队中", cls: "queued" },
    paused: { label: "已暂停", cls: "paused" },
    editing: { label: "编辑中", cls: "editing" },
    running: { label: "训练中", cls: "running" },
    done: { label: "已完成", cls: "done" },
    failed: { label: "失败", cls: "failed" },
  };

  const START_LABELS = ["开始训练", "Start training", "加入训练队列", "Add to training queue", "保存修改到队列", "Save to queue"];

  let state = null; // last /api/queue snapshot
  let overlayOpen = false;
  let dragging = false;
  let pollTimer = null;

  function isEnglish() {
    return (document.documentElement.dataset.sdUiLocale || "") === "en-US";
  }

  function normalize(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function escapeHtml(text) {
    return String(text == null ? "" : text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function api(method, url, body) {
    const res = await fetch(url, {
      method: method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    return res.json();
  }

  function toast(message, ok) {
    let el = document.getElementById(TOAST_ID);
    if (!el) {
      el = document.createElement("div");
      el.id = TOAST_ID;
      document.body.appendChild(el);
    }
    el.textContent = message;
    el.className = ok === false ? "sd-queue-toast-error" : "sd-queue-toast-ok";
    el.style.opacity = "1";
    clearTimeout(el._hideTimer);
    el._hideTimer = setTimeout(() => { el.style.opacity = "0"; }, 3200);
  }

  async function action(method, url, body, silent) {
    try {
      const json = await api(method, url, body);
      if (json && json.data && json.data.entries) state = json.data;
      if (!silent && json && json.message) toast(json.message, json.status === "success");
      renderAll();
      if (!json || json.status !== "success") return null;
      return json;
    } catch (err) {
      if (!silent) toast("队列请求失败：" + err, false);
      return null;
    }
  }

  // ------------------------------------------------------------------ styles

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${NAV_ID} .sd-queue-badge {
  display: inline-block; min-width: 18px; padding: 0 5px; margin-left: 6px;
  border-radius: 999px; font-size: 11px; line-height: 18px; text-align: center;
  background: #f1e5cd; color: #574d38; vertical-align: 1px;
}
#${NAV_ID} .sd-queue-badge:empty { display: none; }
#${NAV_ID} .sd-queue-badge.live { background: #574d38; color: #fff7df; }
html.dark #${NAV_ID} .sd-queue-badge { background: #574d38; color: #f1e5cd; }
html.dark #${NAV_ID} .sd-queue-badge.live { background: #dfd4bc; color: #2d2411; }
/* Rendered as the last l2 entry inside the boxed 训练 group; give it a subtle
   top separator so it reads as its own 「队列」 zone at the card's bottom edge. */
#${NAV_ID} {
  margin-top: 0.45rem; padding-top: 0.45rem;
  border-top: 1px dashed var(--c-border, #dcdfe6);
}

#${OVERLAY_ID} {
  /* embedded page feel: covers the content area only; JS keeps the left
     edge aligned to the live sidebar so it stays visible and clickable */
  position: fixed; top: 0; right: 0; bottom: 0; left: 0;
  z-index: 1800; display: flex; flex-direction: column;
  background: #ffffff; color: #2d2411;
  font-family: inherit;
}
#${OVERLAY_ID} .sdq-toolbar, #${OVERLAY_ID} .sdq-speed,
#${OVERLAY_ID} .sdq-list, #${OVERLAY_ID} .sdq-hints { max-width: 1160px; }
html.dark #${OVERLAY_ID} { background: #1a130a; color: #f1e5cd; }
#${OVERLAY_ID} .sdq-head {
  display: flex; align-items: center; gap: 14px; padding: 16px 26px;
  border-bottom: 1px solid #f1e5cd; flex: none;
}
html.dark #${OVERLAY_ID} .sdq-head { border-color: #574d38; }
#${OVERLAY_ID} .sdq-title { font-size: 19px; font-weight: 700; }
#${OVERLAY_ID} .sdq-state { display: flex; align-items: center; gap: 7px; font-size: 13px; color: #847964; }
#${OVERLAY_ID} .sdq-state .dot { width: 8px; height: 8px; border-radius: 999px; background: #b4a992; }
#${OVERLAY_ID} .sdq-state.on .dot { background: #22c55e; box-shadow: 0 0 0 4px rgba(34,197,94,.15); }
#${OVERLAY_ID} .sdq-close {
  margin-left: auto; border: 1px solid #dfd4bc; background: transparent; color: inherit;
  border-radius: 8px; padding: 5px 14px; font-size: 14px; cursor: pointer;
}
#${OVERLAY_ID} .sdq-close:hover { background: #fff7df; }
html.dark #${OVERLAY_ID} .sdq-close:hover { background: #2d2411; }
#${OVERLAY_ID} .sdq-body { flex: 1; overflow-y: auto; padding: 18px 26px 30px; }
#${OVERLAY_ID} .sdq-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 14px; }
#${OVERLAY_ID} .sdq-btn {
  border: 1px solid #dfd4bc; border-radius: 8px; padding: 7px 16px; cursor: pointer;
  background: #ffffff; color: #574d38; font-size: 13px;
}
#${OVERLAY_ID} .sdq-btn:hover { background: #fff7df; }
#${OVERLAY_ID} .sdq-btn.primary { background: #574d38; border-color: #574d38; color: #fff7df; }
#${OVERLAY_ID} .sdq-btn.primary:hover { background: #2d2411; }
html.dark #${OVERLAY_ID} .sdq-btn { background: transparent; color: #dfd4bc; border-color: #574d38; }
html.dark #${OVERLAY_ID} .sdq-btn:hover { background: #2d2411; }
html.dark #${OVERLAY_ID} .sdq-btn.primary { background: #dfd4bc; border-color: #dfd4bc; color: #2d2411; }
#${OVERLAY_ID} .sdq-history-head {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  margin: 26px 0 10px; padding-top: 14px; font-size: 13px; font-weight: 600; color: #847964;
  border-top: 1px dashed #dfd4bc;
}
html.dark #${OVERLAY_ID} .sdq-history-head { border-color: #574d38; color: #b4a992; }
#${OVERLAY_ID} .sdq-speed {
  border: 1px solid #f1e5cd; background: #fff7df; color: #574d38;
  border-radius: 10px; padding: 10px 14px; font-size: 13px; line-height: 1.7; margin-bottom: 16px;
}
html.dark #${OVERLAY_ID} .sdq-speed { background: #241b0f; border-color: #574d38; color: #dfd4bc; }
#${OVERLAY_ID} .sdq-list { list-style: none; margin: 0; padding: 0; }
#${OVERLAY_ID} .sdq-item {
  display: flex; gap: 12px; align-items: flex-start;
  border: 1px solid #f1e5cd; border-radius: 12px; background: #ffffff;
  padding: 12px 14px; margin-bottom: 10px;
}
html.dark #${OVERLAY_ID} .sdq-item { background: #211809; border-color: #574d38; }
#${OVERLAY_ID} .sdq-item.drag-over { border-color: #574d38; box-shadow: 0 0 0 2px rgba(87,77,56,.25); }
#${OVERLAY_ID} .sdq-item.is-terminal { opacity: .72; }
#${OVERLAY_ID} .sdq-handle {
  cursor: grab; color: #b4a992; font-size: 17px; line-height: 1;
  padding: 4px 2px; user-select: none; flex: none; margin-top: 2px;
}
#${OVERLAY_ID} .sdq-item[draggable="false"] .sdq-handle { cursor: not-allowed; opacity: .35; }
#${OVERLAY_ID} .sdq-idx { flex: none; color: #b4a992; font-size: 13px; margin-top: 5px; min-width: 20px; }
#${OVERLAY_ID} .sdq-main { flex: 1; min-width: 0; }
#${OVERLAY_ID} .sdq-name { font-size: 15px; font-weight: 600; word-break: break-all; }
#${OVERLAY_ID} .sdq-chips { display: inline-flex; gap: 6px; margin-left: 8px; vertical-align: 2px; flex-wrap: wrap; }
#${OVERLAY_ID} .sdq-chip {
  font-size: 11px; border-radius: 999px; padding: 1px 8px;
  background: #f1e5cd; color: #574d38;
}
html.dark #${OVERLAY_ID} .sdq-chip { background: #574d38; color: #f1e5cd; }
#${OVERLAY_ID} .sdq-chip.st-queued { background: #dbeafe; color: #1d4ed8; }
#${OVERLAY_ID} .sdq-chip.st-paused { background: #e2e8f0; color: #475569; }
#${OVERLAY_ID} .sdq-chip.st-editing { background: #ede9fe; color: #6d28d9; }
#${OVERLAY_ID} .sdq-chip.st-running { background: #fef3c7; color: #b45309; }
#${OVERLAY_ID} .sdq-chip.st-done { background: #dcfce7; color: #15803d; }
#${OVERLAY_ID} .sdq-chip.st-failed { background: #fee2e2; color: #b91c1c; }
#${OVERLAY_ID} .sdq-meta { font-size: 12px; color: #847964; margin-top: 5px; line-height: 1.7; }
html.dark #${OVERLAY_ID} .sdq-meta { color: #b4a992; }
#${OVERLAY_ID} .sdq-error {
  margin-top: 6px; font-size: 12px; color: #b91c1c; background: #fee2e2;
  border-radius: 8px; padding: 6px 10px; word-break: break-all;
}
html.dark #${OVERLAY_ID} .sdq-error { background: rgba(185,28,28,.18); color: #fca5a5; }
#${OVERLAY_ID} .sdq-ops { display: flex; flex-wrap: wrap; gap: 6px; flex: none; align-items: center; margin-top: 2px; }
#${OVERLAY_ID} .sdq-op {
  border: 1px solid #dfd4bc; background: transparent; color: #574d38;
  border-radius: 7px; padding: 4px 10px; font-size: 12px; cursor: pointer; white-space: nowrap;
}
#${OVERLAY_ID} .sdq-op:hover { background: #fff7df; }
#${OVERLAY_ID} .sdq-op.danger { color: #b91c1c; border-color: #fecaca; }
#${OVERLAY_ID} .sdq-op.danger:hover { background: #fee2e2; }
html.dark #${OVERLAY_ID} .sdq-op { color: #dfd4bc; border-color: #574d38; }
html.dark #${OVERLAY_ID} .sdq-op:hover { background: #2d2411; }
#${OVERLAY_ID} .sdq-op a { color: inherit; text-decoration: none; }
#${OVERLAY_ID} .sdq-empty {
  border: 1px dashed #dfd4bc; border-radius: 12px; padding: 34px 20px;
  text-align: center; color: #847964; font-size: 13px; line-height: 2;
}
#${OVERLAY_ID} .sdq-hints { margin-top: 20px; font-size: 12px; color: #b4a992; line-height: 2; }
html.dark #${OVERLAY_ID} .sdq-hints { color: #847964; }
#${TOAST_ID} {
  position: fixed; left: 50%; bottom: 34px; transform: translateX(-50%);
  z-index: 4000; max-width: 80vw; padding: 9px 18px; border-radius: 10px;
  font-size: 13px; opacity: 0; transition: opacity .25s; pointer-events: none;
  box-shadow: 0 8px 30px rgba(17,9,0,.18);
}
#${TOAST_ID}.sd-queue-toast-ok { background: #2d2411; color: #fff7df; }
#${TOAST_ID}.sd-queue-toast-error { background: #b91c1c; color: #fff; }
@media (max-width: 760px) {
  #${OVERLAY_ID} .sdq-item { flex-wrap: wrap; }
  #${OVERLAY_ID} .sdq-ops { width: 100%; }
}
`;
    document.head.appendChild(style);
  }

  // ----------------------------------------------------------------- sidebar

  function ensureNav() {
    // Vue re-renders can absorb foreign nodes (the id survives on a recycled
    // li while our content is replaced) — treat such husks as gone and re-add.
    const existing = document.getElementById(NAV_ID);
    if (existing) {
      if (existing.querySelector('a[href="#sd-queue"]')) return;
      existing.removeAttribute("id");
    }
    const sidebar = document.querySelector(".sidebar .sidebar-items");
    if (!sidebar) return;
    let trainChildren = null;
    sidebar.querySelectorAll(":scope > li").forEach((li) => {
      if (trainChildren) return;
      const heading = li.querySelector(":scope > p.sidebar-item.sidebar-heading");
      const text = normalize(heading && heading.textContent);
      if (text === "训练" || text === "Training") {
        trainChildren = li.querySelector(":scope > ul.sidebar-item-children");
      }
    });
    if (!trainChildren) return;

    // appended at the END of the v-for list — the position sd-nav-i18n's
    // terminal link proved survives hydration re-renders
    const en = isEnglish();
    const li = document.createElement("li");
    li.id = NAV_ID;
    const a = document.createElement("a");
    a.className = "sidebar-item sidebar-heading";
    a.href = "#sd-queue";
    a.setAttribute("aria-label", en ? "Training Queue" : "训练队列");
    a.appendChild(document.createTextNode(en ? " Training Queue " : " 训练队列 "));
    const badge = document.createElement("span");
    badge.className = "sd-queue-badge";
    a.appendChild(badge);
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      openOverlay();
    });
    li.appendChild(a);
    trainChildren.appendChild(li);
    console.debug("[sd-queue] sidebar entry injected");
    renderNavBadge();
  }

  function renderNavBadge() {
    const badge = document.querySelector(`#${NAV_ID} .sd-queue-badge`);
    if (!badge || !state) return;
    const pending = (state.entries || []).filter((e) =>
      ["queued", "paused", "editing", "running"].includes(e.status)
    ).length;
    badge.textContent = pending > 0 ? String(pending) : "";
    badge.classList.toggle("live", !!state.active);
  }

  // ------------------------------------------------------------ start button

  function relabelStartButtons() {
    if (!state) return;
    const en = isEnglish();
    let wanted;
    if (state.editing_entry_id) {
      wanted = en ? "Save to queue" : "保存修改到队列";
    } else if (state.busy ||
      (state.entries || []).some((e) => e.status === "queued")) {
      wanted = en ? "Add to training queue" : "加入训练队列";
    } else {
      wanted = en ? "Start training" : "开始训练";
    }
    document.querySelectorAll(".right-container .el-button--primary span").forEach((span) => {
      const cur = normalize(span.textContent);
      if (!START_LABELS.includes(cur)) return;
      if (cur !== wanted) span.textContent = wanted;
    });
  }

  // ----------------------------------------------------------------- overlay

  function humanEta(seconds) {
    if (!seconds || seconds <= 0) return null;
    const minutes = Math.round(seconds / 60);
    if (minutes < 1) return "不到 1 分钟";
    if (minutes < 60) return `约 ${minutes} 分钟`;
    return `约 ${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分`;
  }

  function humanDuration(seconds) {
    if (seconds == null || seconds < 0) return null;
    if (seconds < 60) return `${Math.max(1, Math.round(seconds))} 秒`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes} 分 ${Math.round(seconds % 60)} 秒`;
    return `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分`;
  }

  function elapsedSince(iso) {
    if (!iso) return null;
    const start = new Date(iso).getTime();
    if (!isFinite(start)) return null;
    return humanDuration((Date.now() - start) / 1000);
  }

  function entryOps(entry) {
    const ops = [];
    const op = (act, label, extra) => `<button class="sdq-op ${extra || ""}" data-act="${act}" data-id="${entry.id}">${label}</button>`;
    switch (entry.status) {
      case "queued":
        ops.push(op("start", "▶ 立即开始"), op("pause", "⏸ 暂停"), op("edit", "✎ 编辑"), op("delete", "删除", "danger"));
        break;
      case "paused":
        ops.push(op("resume", "▶ 恢复"), op("edit", "✎ 编辑"), op("delete", "删除", "danger"));
        break;
      case "editing":
        ops.push(op("cancel-edit", "取消编辑"), op("delete", "删除", "danger"));
        break;
      case "running":
        ops.push(`<a class="sdq-op" href="/train-log?task_id=${encodeURIComponent(entry.task_id || "")}" target="_blank" rel="noopener">查看日志</a>`);
        break;
      case "failed":
        ops.push(op("requeue", "↻ 重新排队"), op("edit", "✎ 编辑"), op("delete", "删除", "danger"));
        break;
      case "done":
        ops.push(op("requeue", "↻ 再训一次"), op("delete", "删除", "danger"));
        break;
    }
    return ops.join("");
  }

  function entryMeta(entry) {
    const bits = [];
    if (entry.train_type) bits.push(escapeHtml(entry.train_type));
    if (entry.lora_type) bits.push(escapeHtml(entry.lora_type));
    const terminal = entry.status === "done" || entry.status === "failed";
    if (!terminal) {
      if (entry.images) bits.push(`图片×重复 ${entry.images}`);
      if (entry.steps) bits.push(`预计 ${entry.steps} 步`);
      const eta = humanEta(entry.eta_seconds);
      if (eta) bits.push(`预计时长 ${eta}（仅供参考）`);
      else if (entry.steps) bits.push("预计时长：暂无速度参考");
    }
    if (entry.status === "running" && entry.started_at) {
      const elapsed = elapsedSince(entry.started_at);
      bits.push(`开始于 ${escapeHtml(String(entry.started_at).replace("T", " "))}`);
      if (elapsed) bits.push(`已训练 ${elapsed}`);
    }
    if (terminal && entry.finished_at) {
      const label = entry.status === "done" ? "完成于" : "结束于";
      bits.push(`${label} ${escapeHtml(String(entry.finished_at).replace("T", " "))}`);
      const duration = humanDuration(entry.duration_seconds);
      if (duration) bits.push(`耗时 ${duration}`);
    }
    return bits.join(" · ");
  }

  function entryHtml(entry, idx) {
    const st = STATUS_META[entry.status] || { label: entry.status, cls: "queued" };
    const terminal = entry.status === "done" || entry.status === "failed";
    const draggable = !terminal && entry.status !== "running";
    return `
<li class="sdq-item${terminal ? " is-terminal" : ""}" data-id="${entry.id}" draggable="${draggable}">
  ${draggable ? `<span class="sdq-handle" title="拖动调整顺序">⠿</span>` : ""}
  ${idx != null ? `<span class="sdq-idx">${idx + 1}</span>` : ""}
  <div class="sdq-main">
    <span class="sdq-name">${escapeHtml(entry.name)}</span>
    <span class="sdq-chips"><span class="sdq-chip st-${st.cls}">${st.label}</span></span>
    <div class="sdq-meta">${entryMeta(entry)}</div>
    ${entry.error ? `<div class="sdq-error">${escapeHtml(entry.error)}</div>` : ""}
  </div>
  <div class="sdq-ops">${entryOps(entry)}</div>
</li>`;
  }

  function renderOverlay() {
    const overlay = document.getElementById(OVERLAY_ID);
    if (!overlay || !state || dragging) return;

    const s = state;
    let stateText;
    if (s.active) {
      stateText = "队列进行中：任务完成后自动开始下一个";
    } else if (s.user_paused) {
      stateText = s.halt_reason || "队列已暂停：新任务只入队不开始";
    } else {
      stateText = "队列空闲：提交训练任务后自动按序开始";
    }
    overlay.querySelector(".sdq-state").className = "sdq-state" + (s.active ? " on" : "");
    overlay.querySelector(".sdq-state-text").textContent = stateText;

    const entries = s.entries || [];
    const pending = entries.filter((e) => e.status !== "done" && e.status !== "failed");
    const history = entries.filter((e) => e.status === "done" || e.status === "failed").reverse();

    const toolbar = overlay.querySelector(".sdq-toolbar");
    const anyPaused = pending.some((e) => e.status === "paused");
    toolbar.innerHTML = `
      ${s.active
        ? `<button class="sdq-btn" data-act="queue-stop">⏸ 暂停队列</button>`
        : `<button class="sdq-btn primary" data-act="queue-start">▶ 开始队列</button>`}
      ${anyPaused ? `<button class="sdq-btn" data-act="queue-start-all">▶ 恢复全部并开始</button>` : ""}
    `;

    const speed = s.last_speed;
    overlay.querySelector(".sdq-speed").innerHTML = speed
      ? `速度参考：<b>${speed.it_s} it/s</b>（来自上一任务「${escapeHtml(speed.name)}」${speed.lora_type ? "，" + escapeHtml(speed.lora_type) : ""}）。` +
        `<br>注意：LoRA / LoKr 等不同算法、不同分辨率与参数下速度差异明显，预计时长<b>仅供参考</b>。`
      : "暂无速度参考：完成一次训练后会自动记录上一任务的 it/s，用于折算预计时长（LoRA / LoKr 等算法速度不同，仅供参考）。";

    const list = overlay.querySelector(".sdq-list");
    if (!pending.length) {
      list.innerHTML = `<li class="sdq-empty">队列是空的喵。<br>在训练页配置好参数后点「开始训练」即可入队并自动开始；<br>连续多次提交就会自动排成队，按顺序训练。</li>`;
    } else {
      list.innerHTML = pending.map((entry, idx) => entryHtml(entry, idx)).join("");
    }

    const historyBox = overlay.querySelector(".sdq-history");
    if (!history.length) {
      historyBox.innerHTML = "";
    } else {
      historyBox.innerHTML = `
<div class="sdq-history-head">
  <span>历史记录（${history.length}）</span>
  <button class="sdq-op" data-act="clear-finished">归档清除全部历史</button>
</div>
<ul class="sdq-list">${history.map((entry) => entryHtml(entry, null)).join("")}</ul>`;
    }
  }

  function buildOverlay() {
    if (document.getElementById(OVERLAY_ID)) return;
    injectStyle();
    const overlay = document.createElement("div");
    overlay.id = OVERLAY_ID;
    overlay.style.display = "none";
    overlay.innerHTML = `
<div class="sdq-head">
  <span class="sdq-title">训练队列</span>
  <span class="sdq-state"><span class="dot"></span><span class="sdq-state-text">载入中…</span></span>
  <button class="sdq-close" data-act="close">✕ 关闭</button>
</div>
<div class="sdq-body">
  <div class="sdq-toolbar"></div>
  <div class="sdq-speed">载入中…</div>
  <ul class="sdq-list"></ul>
  <div class="sdq-history"></div>
  <div class="sdq-hints">
    · 所有训练任务都经由队列：空闲时提交会立即自动开始，忙碌时自动排队，完成后自动接跑下一个。<br>
    · 预计时长 = 预计步数 ÷ 上一任务实测 it/s。LoRA 与 LoKr 等算法速度差异明显，仅供参考。<br>
    · 「编辑」会把该任务参数载入对应训练页表单（当前表单未保存内容会被覆盖）；编辑期间队列暂停，点「保存修改到队列」或取消编辑后自动继续（若编辑前手动暂停过队列，则保持暂停）。<br>
    · 完成/失败的任务会留在历史记录里（含完成时间与耗时），可单个删除、整体归档清除，或放着不管。<br>
    · 想停下正在训练的任务：先「暂停队列」再到日志页终止，避免队列自动接跑下一个。正在训练的任务无法在此暂停或删除。
  </div>
</div>`;
    overlay.addEventListener("click", onOverlayClick);
    bindDragEvents(overlay);
    document.body.appendChild(overlay);
  }

  function positionOverlay() {
    const overlay = document.getElementById(OVERLAY_ID);
    if (!overlay || !overlayOpen) return;
    const sidebar = document.querySelector("aside.sidebar");
    const left = sidebar ? Math.max(0, Math.round(sidebar.getBoundingClientRect().right)) : 0;
    overlay.style.left = left + "px";
  }

  function setNavActive(on) {
    const a = document.querySelector(`#${NAV_ID} a[href="#sd-queue"]`);
    if (a) a.classList.toggle("active", on);
  }

  function openOverlay() {
    buildOverlay();
    const overlay = document.getElementById(OVERLAY_ID);
    overlay.style.display = "flex";
    overlayOpen = true;
    positionOverlay();
    setNavActive(true);
    document.body.style.overflow = "hidden";
    refresh();
  }

  function closeOverlay() {
    const overlay = document.getElementById(OVERLAY_ID);
    if (overlay) overlay.style.display = "none";
    overlayOpen = false;
    setNavActive(false);
    document.body.style.overflow = "";
  }

  // ------------------------------------------------------------------ events

  function onOverlayClick(ev) {
    const target = ev.target.closest("[data-act]");
    if (!target) return;
    const act = target.getAttribute("data-act");
    const id = target.getAttribute("data-id");
    switch (act) {
      case "close": closeOverlay(); break;
      case "queue-start": action("POST", "/api/queue/start", {}); break;
      case "queue-start-all": action("POST", "/api/queue/start", { include_paused: true }); break;
      case "queue-stop": action("POST", "/api/queue/stop"); break;
      case "clear-finished":
        if (confirm("归档清除全部历史记录？（不影响排队和正在训练的任务）")) {
          action("POST", "/api/queue/clear-finished");
        }
        break;
      case "pause": action("POST", `/api/queue/entries/${id}/pause`); break;
      case "resume": action("POST", `/api/queue/entries/${id}/resume`); break;
      case "requeue": action("POST", `/api/queue/entries/${id}/requeue`); break;
      case "start": action("POST", `/api/queue/entries/${id}/start`); break;
      case "cancel-edit": action("POST", `/api/queue/entries/${id}/editing`, { editing: false }); break;
      case "delete":
        if (confirm("确定删除这个队列任务吗？")) action("DELETE", `/api/queue/entries/${id}`);
        break;
      case "edit": startEdit(id); break;
    }
  }

  async function startEdit(id) {
    const entry = (state && state.entries || []).find((e) => e.id === id);
    if (!entry) return;
    const page = PAGE_MAP[entry.train_type];
    if (!page) {
      toast(`该任务类型（${entry.train_type || "未知"}）暂不支持在队列中编辑`, false);
      return;
    }
    const ok = confirm(
      `编辑「${entry.name}」？\n\n将把该任务参数载入「${page.path}」训练页表单，` +
      `当前该页表单里未保存的内容会被覆盖。\n改完后点页面上的「保存修改到队列」按钮保存，队列会自动继续。`
    );
    if (!ok) return;
    const cfgResp = await api("GET", `/api/queue/entries/${id}/config`);
    if (!cfgResp || cfgResp.status !== "success" || !cfgResp.data || !cfgResp.data.config) {
      toast("读取任务参数失败", false);
      return;
    }
    const marked = await action("POST", `/api/queue/entries/${id}/editing`, { editing: true }, true);
    if (!marked) {
      toast("进入编辑状态失败（可能有其他任务正在编辑）", false);
      return;
    }
    try {
      // NEVER write the entry config into configs-*-autosave: it is the flat
      // POSTed config (string LRs parsed to numbers, branch fields folded into
      // network_args) and restoring it verbatim blanks the form. The pending
      // import channel re-hydrates it server-side and merges schema defaults.
      sessionStorage.setItem("mikazuki-pending-import", JSON.stringify(cfgResp.data.config));
    } catch (err) {
      toast("写入本地表单缓存失败：" + err, false);
      return;
    }
    // hard navigation on purpose: the page applies the pending import on mount
    if (location.pathname === page.path) {
      location.reload();
    } else {
      location.href = page.path;
    }
  }

  // ------------------------------------------------------------- drag & drop

  let draggedId = null;

  function bindDragEvents(overlay) {
    const list = () => overlay.querySelector(".sdq-list");
    overlay.addEventListener("dragstart", (ev) => {
      const item = ev.target.closest(".sdq-item");
      if (!item || item.getAttribute("draggable") !== "true") { ev.preventDefault(); return; }
      draggedId = item.getAttribute("data-id");
      dragging = true;
      ev.dataTransfer.effectAllowed = "move";
      try { ev.dataTransfer.setData("text/plain", draggedId); } catch (e) { /* ignore */ }
    });
    overlay.addEventListener("dragover", (ev) => {
      const item = ev.target.closest(".sdq-item");
      if (!item || !draggedId) return;
      if (item.parentNode !== list()) return; // pending list only, not history
      ev.preventDefault();
      ev.dataTransfer.dropEffect = "move";
      const dragged = list().querySelector(`.sdq-item[data-id="${draggedId}"]`);
      if (!dragged || dragged === item) return;
      const rect = item.getBoundingClientRect();
      const before = ev.clientY < rect.top + rect.height / 2;
      item.parentNode.insertBefore(dragged, before ? item : item.nextSibling);
    });
    overlay.addEventListener("drop", (ev) => { if (draggedId) ev.preventDefault(); });
    overlay.addEventListener("dragend", () => {
      if (!draggedId) return;
      draggedId = null;
      dragging = false;
      const ids = Array.from(list().querySelectorAll(".sdq-item")).map((el) => el.getAttribute("data-id"));
      action("POST", "/api/queue/reorder", { ids: ids }, true);
    });
  }

  // -------------------------------------------------------------------- poll

  async function refresh() {
    if (document.hidden) return;
    try {
      const json = await api("GET", "/api/queue");
      if (json && json.status === "success" && json.data) {
        state = json.data;
        renderAll();
      }
    } catch (err) { /* offline / restarting server — keep quiet */ }
  }

  function renderAll() {
    ensureNav();
    renderNavBadge();
    relabelStartButtons();
    if (overlayOpen) {
      setNavActive(true);
      positionOverlay();
      renderOverlay();
    }
  }

  function boot() {
    console.debug("[sd-queue] boot");
    injectStyle();
    ensureNav();
    refresh();
    window.addEventListener("resize", positionOverlay);
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && overlayOpen) closeOverlay();
    });
    // clicking any other sidebar link while the panel is open: fold the panel
    // so the SPA navigation underneath stays visible (embedded-page behavior)
    document.addEventListener("click", (ev) => {
      if (!overlayOpen || !ev.target.closest) return;
      const link = ev.target.closest(".sidebar a");
      if (link && link.getAttribute("href") !== "#sd-queue") closeOverlay();
    }, true);
    if (location.hash === "#sd-queue") openOverlay();
    if (!pollTimer) pollTimer = setInterval(refresh, POLL_MS);
    const root = document.querySelector("#app");
    if (root) {
      let scheduled = null;
      new MutationObserver(() => {
        if (scheduled) return;
        scheduled = setTimeout(() => {
          scheduled = null;
          ensureNav();
          renderNavBadge();
          relabelStartButtons();
        }, 120);
      }).observe(root, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
