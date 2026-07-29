/**
 * Version chip next to the "Next Trainer" sidebar title (reads /api/version).
 */
(function () {
  const VERSION_URL = "/api/version";
  const CHIP_ID = "sd-brand-version-chip";
  const BRAND_TITLE = "Next Trainer";
  const GAP_PX = 6;
  const OFFSET_Y_PX = 3;

  function versionFromScriptTag() {
    const el = document.querySelector('script[src*="sd-trainer-brand.js"]');
    if (!el) return null;
    try {
      const v = new URL(el.src, window.location.origin).searchParams.get("v");
      return v ? String(v).trim() : null;
    } catch (e) {
      return null;
    }
  }

  async function fetchVersion() {
    try {
      const res = await fetch(VERSION_URL);
      const json = await res.json();
      if (json && json.status === "success" && json.data && json.data.version) {
        return String(json.data.version).trim();
      }
    } catch (e) {
      /* backend offline */
    }
    return versionFromScriptTag();
  }

  function findBrandLink() {
    const sidebar = document.querySelector(".sidebar .sidebar-items");
    if (!sidebar) return null;
    return (
      sidebar.querySelector("li:first-child > a.sidebar-item.sidebar-heading[href='/']") ||
      sidebar.querySelector('a.sidebar-item.sidebar-heading[aria-label="Next Trainer"]')
    );
  }

  function measureBrandTitleRect(link) {
    const walker = document.createTreeWalker(link, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const raw = node.textContent || "";
      const idx = raw.indexOf(BRAND_TITLE);
      if (idx !== -1) {
        const range = document.createRange();
        range.setStart(node, idx);
        range.setEnd(node, idx + BRAND_TITLE.length);
        const r = range.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) return r;
      }
    }
    return link.getBoundingClientRect();
  }

  function positionChip() {
    const chip = document.getElementById(CHIP_ID);
    const link = findBrandLink();
    if (!chip || !link) {
      if (chip) chip.style.visibility = "hidden";
      return false;
    }

    const linkRect = link.getBoundingClientRect();
    const titleRect = measureBrandTitleRect(link);
    if (linkRect.width <= 0 || linkRect.height <= 0) {
      chip.style.visibility = "hidden";
      return false;
    }

    chip.style.visibility = "visible";
    const anchor = titleRect.height > 0 ? titleRect : linkRect;
    chip.style.top =
      Math.round(anchor.top + (anchor.height - chip.offsetHeight) / 2 + OFFSET_Y_PX) + "px";
    chip.style.left = Math.round(titleRect.right + GAP_PX) + "px";
    chip.style.right = "auto";
    return true;
  }

  function ensureChip(version) {
    if (!version) return;
    document.documentElement.dataset.sdTrainerVersion = version;

    let chip = document.getElementById(CHIP_ID);
    if (!chip) {
      chip = document.createElement("div");
      chip.id = CHIP_ID;
      chip.className = "sd-brand-version-chip";
      chip.setAttribute("title", "Next Trainer / SD-Trainer 版本号");
      document.body.appendChild(chip);
    }
    chip.textContent = "v" + version;
    positionChip();
  }

  let resizeTimer = null;
  function scheduleReposition() {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(positionChip, 80);
  }

  function loadAnimaFastInstall() {
    if (window.__ANIMA_FAST_INSTALL_GUARD__) return;
    const existing = document.querySelector('script[src*="anima-fast-install.js"]');
    if (existing) return;
    const version = versionFromScriptTag() || "2.7.0";
    const script = document.createElement("script");
    script.src = "/assets/anima-fast-install.js?v=" + encodeURIComponent(version);
    script.defer = true;
    document.head.appendChild(script);
  }

  const GUIDE_PAGE_HASHES = ["#新手上路", "#从秋叶版迁移", "#anima-fast-lora"];

  function hashToGuideIndex() {
    const hash = location.hash || "#新手上路";
    const i = GUIDE_PAGE_HASHES.indexOf(hash);
    return i >= 0 ? i : 0;
  }

  function setupGuidePagerRoot(root) {
    if (!root || root.dataset.guidePagerReady === "1") return;
    root.dataset.guidePagerReady = "1";

    const pages = Array.from(root.querySelectorAll("[data-guide-page]"));
    if (!pages.length) return;

    const prevBtn = root.querySelector("[data-guide-prev]");
    const nextBtn = root.querySelector("[data-guide-next]");
    const countEl = root.querySelector("[data-guide-count]");
    let index = hashToGuideIndex();

    function setPage(i, opts) {
      index = Math.max(0, Math.min(pages.length - 1, i));
      pages.forEach(function (p, j) {
        p.classList.toggle("is-active", j === index);
        p.hidden = j !== index;
      });
      if (prevBtn) prevBtn.disabled = index === 0;
      if (nextBtn) nextBtn.disabled = index === pages.length - 1;
      if (countEl) countEl.textContent = index + 1 + " / " + pages.length;
      root.dataset.guideCurrentPage = String(index);
      const hash = GUIDE_PAGE_HASHES[index];
      if (!opts || !opts.skipHash) {
        const url = location.pathname + location.search + hash;
        if (location.pathname + location.search + location.hash !== url) {
          history.replaceState(null, "", url);
        }
      }
      const viewport = root.querySelector(".sd-guide-pager__viewport");
      if (viewport) viewport.scrollTop = 0;
      const main = document.querySelector("main.page");
      if (main) main.scrollTop = 0;
    }

    root._guideSetPage = setPage;
    setPage(index, { skipHash: true });
  }

  function scanGuidePagers() {
    if (!/^\/help\/guide(\.html|\.md)?$/i.test(location.pathname)) return;
    document.querySelectorAll("[data-guide-pager]").forEach(setupGuidePagerRoot);
  }

  function syncGuidePagerFromHash() {
    if (!/^\/help\/guide(\.html|\.md)?$/i.test(location.pathname)) return;
    const idx = hashToGuideIndex();
    document.querySelectorAll("[data-guide-pager]").forEach(function (root) {
      if (root.dataset.guidePagerReady === "1" && typeof root._guideSetPage === "function") {
        root._guideSetPage(idx, { skipHash: true });
      }
    });
  }

  function onGuidePagerClick(ev) {
    const prev = ev.target && ev.target.closest && ev.target.closest("[data-guide-prev]");
    const next = ev.target && ev.target.closest && ev.target.closest("[data-guide-next]");
    if (!prev && !next) return;
    const root = (prev || next).closest("[data-guide-pager]");
    if (!root) return;
    ev.preventDefault();
    if (root.dataset.guidePagerReady !== "1") setupGuidePagerRoot(root);
    const idx = parseInt(root.dataset.guideCurrentPage || "0", 10);
    const total = root.querySelectorAll("[data-guide-page]").length;
    if (prev && idx > 0 && typeof root._guideSetPage === "function") root._guideSetPage(idx - 1);
    if (next && idx < total - 1 && typeof root._guideSetPage === "function") root._guideSetPage(idx + 1);
  }

  function watchGuidePagerMount() {
    if (window.__sdGuidePagerWatcher__) return;
    window.__sdGuidePagerWatcher__ = true;
    const obs = new MutationObserver(function () {
      scanGuidePagers();
    });
    if (document.body) {
      obs.observe(document.body, { childList: true, subtree: true });
    }
  }

  function scheduleGuidePager() {
    scanGuidePagers();
    let left = 60;
    const timer = setInterval(function () {
      scanGuidePagers();
      if (--left <= 0) clearInterval(timer);
    }, 150);
  }

  document.addEventListener("click", onGuidePagerClick);

  document.addEventListener("click", function (ev) {
    const link = ev.target && ev.target.closest && ev.target.closest("[data-guide-fast-link]");
    if (!link) return;
    ev.preventDefault();
    window.location.assign("/help/guide.html#anima-fast-lora");
  });

  async function boot() {
    const version = (await fetchVersion()) || versionFromScriptTag();
    if (version) ensureChip(version);
    setupMobileNav();
    loadAnimaFastInstall();
    watchGuidePagerMount();
    scheduleGuidePager();
    window.addEventListener("hashchange", function () {
      scanGuidePagers();
      syncGuidePagerFromHash();
    });

    let tries = 0;
    const retry = setInterval(function () {
      positionChip();
      if (++tries >= 30) clearInterval(retry);
    }, 200);

    window.addEventListener("resize", scheduleReposition);
    window.addEventListener("scroll", scheduleReposition, true);
  }

  function setupMobileNav() {
    const root = document.querySelector(".theme-container.no-navbar");
    if (!root || document.querySelector(".sd-mobile-nav-toggle")) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sd-mobile-nav-toggle";
    btn.setAttribute("aria-label", "打开导航菜单");
    btn.setAttribute("aria-expanded", "false");
    btn.textContent = "\u2630";
    document.body.appendChild(btn);

    const mask = root.querySelector(".sidebar-mask");

    function closeNav() {
      root.classList.remove("sidebar-open");
      btn.setAttribute("aria-expanded", "false");
      btn.setAttribute("aria-label", "打开导航菜单");
    }

    btn.addEventListener("click", function () {
      if (root.classList.contains("sidebar-open")) {
        closeNav();
        return;
      }
      root.classList.add("sidebar-open");
      btn.setAttribute("aria-expanded", "true");
      btn.setAttribute("aria-label", "关闭导航菜单");
    });

    if (mask) {
      mask.addEventListener("click", closeNav);
    }

    window.addEventListener("resize", function () {
      if (window.innerWidth > 959) {
        closeNav();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();

/**
 * Param effectiveness highlighter: some lora_type algorithms ignore or
 * reinterpret shared fields (network_dim / network_alpha). Instead of a long
 * static description, highlight the affected field for the CURRENT algo.
 * Facts audited 2026-07-29 against vendor/lycoris + vendor/sd-scripts.
 */
(function () {
  if (window.__sdParamAdvisor) return;
  window.__sdParamAdvisor = true;

  const KRON_DIM = ["warn", "当前算法：dim 仅是\u201c是否继续分解\u201d的阈值，超过阈值转满矩阵后再加大无变化（大值是常态）"];
  const KRON_ALPHA = ["warn", "当前算法：满矩阵模式下被静默丢弃（强制 scale=1）"];
  const RULES = {
    network_dim: {
      lokr: KRON_DIM, glokr: KRON_DIM, bokr: KRON_DIM, gsokr: KRON_DIM,
      glora_boft: ["warn", "当前算法：兼作 BOFT 分块因子，参数量随 dim 非单调跳变"],
      waveft: ["warn", "当前算法：被 waveft_n_frequency 覆盖，此值不生效"],
      deft: ["warn", "当前算法：会被静默钳制到层宽"],
      cdka: ["dead", "对当前算法无效：容量由 cdka_r1 / r2 / r 决定"],
    },
    network_alpha: {
      lokr: KRON_ALPHA, glokr: KRON_ALPHA, bokr: KRON_ALPHA, gsokr: KRON_ALPHA,
      glora_boft: ["warn", "当前算法：仅作用于 GLoRA 路径，BOFT 旋转由 boft_constraint 控制"],
      delora: ["dead", "对当前算法无效：真正的缩放是 delora_lambda"],
      waveft: ["dead", "对当前算法无效：真正的缩放是 waveft_scaling"],
      deft: ["dead", "对当前算法无效：真正的缩放是 deft_alpha"],
      cdka: ["dead", "对当前算法无效：缩放 = cdka_alpha/√(r·r₂)"],
    },
  };
  const BADGE_CLASS = "sd-param-advice";

  // 左侧亮条：默认「已修改」蓝色 → 粉色；下拉框按选中项序号取色；开关仅开启时亮
  const BAR_PINK = "#EEB2B3";
  const BAR_PALETTE = [
    "#EEB2B3", "#A9C8E8", "#A8D8B9", "#E8D3A2", "#C3B1E1",
    "#F4B183", "#8FD3C7", "#E79FC4", "#B5C99A", "#9FB8D8",
  ];

  function injectStyle() {
    if (document.getElementById("sd-param-advice-style")) return;
    const css = [
      "." + BADGE_CLASS + "{display:block;margin-top:4px;padding:2px 10px;border-radius:6px;font-size:12px;line-height:1.6;font-weight:600;width:fit-content}",
      "." + BADGE_CLASS + ".warn{background:#fdf3c9;color:#8a6100;border:1px solid #f2d67c}",
      "." + BADGE_CLASS + ".dead{background:#fde8e8;color:#9b1c1c;border:1px solid #f5b6b6}",
      "html.dark ." + BADGE_CLASS + ".warn{background:#4a3d10;color:#f2d67c;border-color:#6b591c}",
      "html.dark ." + BADGE_CLASS + ".dead{background:#4a1717;color:#f5a3a3;border-color:#6b2424}",
      ".k-schema-item.changed .actions{border-left-color:" + BAR_PINK + "}",
      // 主视觉配色：奶油咖啡单色渐变 50#fff7df…950#020000（用户指定调色板）
      // 浅色主题：大面积纯白留白，咖啡色只用于文字与重点（品牌色 700#574d38）
      ":root{" +
        "--white:#ffffff;--white-soft:#fdfcf8;--white-mute:#f7f4ea;" +
        "--c-bg:#ffffff;--c-bg-light:#f7f4ea;--c-bg-lighter:#efe9d9;--c-bg-arrow:#b4a992;" +
        "--c-text:#2d2411;--c-text-light:#574d38;--c-text-lighter:#6e6350;--c-text-lightest:#847964;" +
        "--c-border:#e9e2d0;--c-border-dark:#dfd4bc;--c-divider:#e9e2d0;--c-divider-light:#f1ecdd;" +
        "--brand:#574d38;--brand-light:#6e6350;--brand-lighter:#847964;--brand-lightest:#b4a992;" +
        "--brand-dark:#463d2c;--brand-darker:#2d2411;--brand-dimm:rgba(87,77,56,.08);" +
        "--c-brand:#574d38;--c-brand-light:#6e6350;--c-brand-dark:#2d2411;" +
        "--el-color-primary:#574d38;--el-color-primary-rgb:87,77,56;--el-color-primary-dark-2:#2d2411;" +
        "--el-color-primary-light-3:#847964;--el-color-primary-light-5:#b4a992;" +
        "--el-color-primary-light-7:#cec2ab;--el-color-primary-light-8:#dfd4bc;" +
        "--el-color-primary-light-9:#f1e5cd" +
      "}",
      // 深色主题：深咖底 + 奶油字，品牌色取 200#dfd4bc
      "html.dark{" +
        "--black:#110900;--black-soft:#1a1206;--black-mute:#241a0a;" +
        "--c-bg:#110900;--c-bg-light:#1a1206;--c-bg-lighter:#241a0a;--c-bg-arrow:#574d38;" +
        "--c-text:#f1e5cd;--c-text-light:#dfd4bc;--c-text-lighter:#cec2ab;--c-text-lightest:#b4a992;" +
        "--c-border:#2d2411;--c-border-dark:#3a3020;--c-divider:#2d2411;--c-divider-light:#3a3020;" +
        "--brand:#dfd4bc;--brand-light:#f1e5cd;--brand-lighter:#fff7df;--brand-lightest:#fff7df;" +
        "--brand-dark:#cec2ab;--brand-darker:#b4a992;--brand-dimm:rgba(223,212,188,.08);" +
        "--c-brand:#dfd4bc;--c-brand-light:#f1e5cd;--c-brand-dark:#b4a992;" +
        "--el-color-primary:#dfd4bc;--el-color-primary-rgb:223,212,188;--el-color-primary-dark-2:#cec2ab;" +
        "--el-color-primary-light-3:#847964;--el-color-primary-light-5:#574d38;" +
        "--el-color-primary-light-7:#3a3020;--el-color-primary-light-8:#2d2411;" +
        "--el-color-primary-light-9:#1e1508" +
      "}",
    ].join("\n");
    const style = document.createElement("style");
    style.id = "sd-param-advice-style";
    style.textContent = css;
    document.head.appendChild(style);
  }

  function schemaItems() {
    const map = {};
    document.querySelectorAll(".k-schema-item").forEach(function (item) {
      const left = item.querySelector(".k-schema-left");
      if (!left) return;
      const header = left.querySelector("h3") || left.firstElementChild;
      if (!header) return;
      // header may carry anchors/required markers — the raw field name is the first token
      const name = ((header.textContent || "").trim().split(/\s+/)[0] || "");
      if (name && !map[name]) map[name] = { item: item, left: left, header: header };
    });
    return map;
  }

  function currentLoraType(items) {
    const entry = items["lora_type"];
    if (!entry) return null;
    const input = entry.item.querySelector(".k-schema-right input");
    return input ? String(input.value || "").trim() : null;
  }

  function setBadge(entry, rule) {
    if (!entry) return;
    let badge = entry.left.querySelector("." + BADGE_CLASS);
    if (!rule) {
      if (badge) badge.remove();
      return;
    }
    if (!badge) {
      badge = document.createElement("div");
      entry.left.insertBefore(badge, entry.header.nextSibling);
    }
    // Only write on change \u2014 the MutationObserver watches our own badges, and
    // an unconditional textContent write would re-trigger apply() every frame.
    const cls = BADGE_CLASS + " " + rule[0];
    const text = (rule[0] === "dead" ? "\u26d4 " : "\u26a0\ufe0f ") + rule[1];
    if (badge.className !== cls) badge.className = cls;
    if (badge.textContent !== text) badge.textContent = text;
  }

  function unionOptionIndex(selectEl, value) {
    // schemastery-vue keeps the union schema on the Vue component chain; walk up
    // from the el-select until a component carries props.schema.list
    try {
      let comp = selectEl.__vueParentComponent;
      for (let hops = 0; comp && hops < 12; hops += 1, comp = comp.parent) {
        const schema = comp.props && comp.props.schema;
        const list = schema && schema.list;
        if (Array.isArray(list) && list.length) {
          return list.findIndex(function (branch) {
            const v = branch && typeof branch === "object" && "value" in branch ? branch.value : branch;
            return String(v) === String(value);
          });
        }
      }
    } catch (e) { /* Vue internals unavailable — fall through */ }
    return -1;
  }

  function hashIndex(text) {
    let h = 0;
    for (let i = 0; i < text.length; i += 1) h = (h * 31 + text.charCodeAt(i)) >>> 0;
    return h % BAR_PALETTE.length;
  }

  function recolorBars() {
    document.querySelectorAll(".k-schema-item").forEach(function (item) {
      const actions = item.querySelector(":scope > .actions") || item.querySelector(".actions");
      if (!actions || actions.closest(".k-schema-item") !== item) return;
      let color = "";
      // nested groups: only honor controls belonging to THIS item, not a child item
      let sw = item.querySelector(".el-switch");
      if (sw && sw.closest(".k-schema-item") !== item) sw = null;
      let select = sw ? null : item.querySelector(".el-select");
      if (select && select.closest(".k-schema-item") !== item) select = null;
      if (sw) {
        // 开关：启用才亮起
        color = sw.classList.contains("is-checked") ? BAR_PINK : "transparent";
      } else if (select) {
        const input = select.querySelector("input");
        const value = input ? String(input.value || "").trim() : "";
        if (value) {
          const idx = unionOptionIndex(select, value);
          color = BAR_PALETTE[(idx >= 0 ? idx : hashIndex(value)) % BAR_PALETTE.length];
        }
      }
      if (actions.style.borderLeftColor !== color) {
        actions.style.borderLeftColor = color;
      }
    });
  }

  let scheduled = false;
  function apply() {
    scheduled = false;
    injectStyle();
    recolorBars();
    const items = schemaItems();
    const loraType = currentLoraType(items);
    if (loraType === null) return; // page without a lora_type form
    Object.keys(RULES).forEach(function (field) {
      setBadge(items[field], RULES[field][loraType] || null);
    });
  }

  function schedule() {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(apply);
  }

  function boot() {
    const app = document.getElementById("app") || document.body;
    new MutationObserver(schedule).observe(app, { childList: true, subtree: true });
    // switch toggles only mutate class attributes, and el-select dropdowns are
    // teleported outside #app — cover both via capture-phase events
    document.addEventListener("click", schedule, true);
    document.addEventListener("change", schedule, true);
    schedule();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
