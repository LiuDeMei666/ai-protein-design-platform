/* ==========================================================================
   应用外壳：顶部导航、底部状态栏、Toast、项目上下文
   --------------------------------------------------------------------------
   五个页面的导航与状态栏由本文件统一注入，避免在 5 份 HTML 里复制粘贴导致
   后续改一处要改五处。页面只需给出 <div id="app-nav"></div> 等占位容器。
   ========================================================================== */

(function (global) {
  "use strict";

  const { get, fmt, CONST } = global.API;

  const PAGES = [
    { href: "index.html", label: "工作台", key: "index" },
    { href: "structure.html", label: "序列与结构", key: "structure" },
    { href: "property.html", label: "性质报告", key: "property" },
    { href: "design.html", label: "突变设计", key: "design" },
    { href: "experiment.html", label: "实验与迭代", key: "experiment" },
  ];

  /* ---------------- 全局状态（跨页面用 sessionStorage 保持选择） ---------------- */
  const store = {
    get(key, fallback = null) {
      try {
        const raw = sessionStorage.getItem("dbz:" + key);
        return raw === null ? fallback : JSON.parse(raw);
      } catch (_) {
        return fallback;
      }
    },
    set(key, value) {
      try {
        sessionStorage.setItem("dbz:" + key, JSON.stringify(value));
      } catch (_) {}
    },
  };

  const state = {
    projects: [],
    currentProjectId: store.get("projectId", null),
    hostSystem: store.get("hostSystem", "ecoli"),
    health: null,
    queue: null,
  };

  /* ---------------- Toast ---------------- */
  let toastHost = null;

  function toast(message, kind = "info", durationMs = 4200) {
    if (!toastHost) {
      toastHost = document.createElement("div");
      toastHost.className = "toast-host";
      document.body.appendChild(toastHost);
    }
    const icons = { ok: "✓", warn: "!", danger: "✕", info: "i" };
    const node = document.createElement("div");
    node.className = "toast " + kind;
    node.innerHTML = `<span class="ico">${icons[kind] || "i"}</span><span class="msg"></span>`;
    node.querySelector(".msg").textContent = message;
    toastHost.appendChild(node);

    setTimeout(() => {
      node.classList.add("out");
      setTimeout(() => node.remove(), 300);
    }, durationMs);
    return node;
  }

  /** 统一错误提示：把 ApiError 的 code 转成对用户有用的说明 */
  function toastError(err) {
    if (!err) return toast("发生未知错误", "danger");
    if (err.name === "JobError" && err.job) {
      const job = err.job;
      toast(`作业失败：${job.error || "未知原因"}`, "danger", 9000);
      return;
    }
    const hints = {
      network_error: "请确认后端服务已启动（bash run_server.sh）",
      timeout: "请求超时，服务可能正在处理大任务，可稍后在工作台查看作业状态",
    };
    const extra = hints[err.code] ? `（${hints[err.code]}）` : "";
    toast(`${err.message || err}${extra}`, "danger", 8000);
  }

  /* ---------------- 导航与状态栏 ---------------- */
  function mountShell(activeKey) {
    const navHost = document.getElementById("app-nav");
    if (navHost) {
      navHost.className = "app-nav";
      navHost.innerHTML = `
        <a class="brand" href="index.html">
          <span class="brand-mark">DBZ</span>
          <span>
            <div>AI 辅助蛋白设计平台</div>
            <div class="brand-sub">Protein Design Console</div>
          </span>
        </a>
        <nav class="nav-links">
          ${PAGES.map(
            (page) =>
              `<a class="nav-link${page.key === activeKey ? " active" : ""}" href="${page.href}">${page.label}</a>`
          ).join("")}
        </nav>
        <div class="nav-tools">
          <select class="select" id="project-select" style="width:auto;min-width:150px"
                  title="当前项目：序列、设计批次与实验数据都归属项目"></select>
        </div>
      `;
      navHost.querySelector("#project-select").addEventListener("change", (event) => {
        state.currentProjectId = Number(event.target.value);
        store.set("projectId", state.currentProjectId);
        document.dispatchEvent(new CustomEvent("project-changed", { detail: state.currentProjectId }));
      });
    }

    const statusHost = document.getElementById("app-status");
    if (statusHost) {
      statusHost.className = "app-status";
      statusHost.innerHTML = `
        <span>状态 <b id="st-health">检测中…</b></span>
        <span class="sep">|</span>
        <span>结构通道 <b id="st-provider">—</b></span>
        <span class="sep">|</span>
        <span>模型 <b id="st-model">—</b></span>
        <span class="sep">|</span>
        <span>作业 <b id="st-jobs">—</b></span>
        <span class="sep">|</span>
        <span>缓存 <b id="st-cache">—</b></span>
        <span class="sep">|</span>
        <span>v<b id="st-version">—</b></span>
      `;
    }
  }

  /* ---------------- 项目 ---------------- */
  async function loadProjects() {
    try {
      const page = await get("/projects", { limit: 200 });
      state.projects = page.items || [];
      if (!state.currentProjectId && state.projects.length) {
        state.currentProjectId = state.projects[0].id;
      }
      const select = document.getElementById("project-select");
      if (select) {
        select.innerHTML = state.projects
          .map(
            (project) =>
              `<option value="${project.id}"${
                project.id === state.currentProjectId ? " selected" : ""
              }>${fmt.escape(project.name)}（${project.sequence_count} 序列）</option>`
          )
          .join("");
      }
      store.set("projectId", state.currentProjectId);
    } catch (err) {
      console.warn("加载项目失败", err);
    }
    return state.projects;
  }

  function currentProjectId() {
    // 页面未选择时回落到第一个项目，保证作业结果能落库（否则导出/历史会空）
    return state.currentProjectId || (state.projects[0] && state.projects[0].id) || null;
  }

  /* ---------------- 状态栏刷新 ---------------- */
  async function refreshStatus() {
    const setText = (id, text, color) => {
      const node = document.getElementById(id);
      if (!node) return;
      node.textContent = text;
      if (color) node.style.color = color;
    };

    try {
      const health = await get("/health");
      state.health = health;
      const providers = (health.structure && health.structure.providers) || {};
      const online = providers.esmatlas && providers.esmatlas.available;
      const localOk = providers.local_esmfold && providers.local_esmfold.available;
      const stubOnly = !online && !localOk;

      setText(
        "st-health",
        health.healthy ? "正常" : "降级",
        health.healthy ? "var(--ok)" : "var(--warn)"
      );
      setText(
        "st-provider",
        online ? "ESM Atlas 在线" : localOk ? "本地 ESMFold" : "离线占位",
        online ? "var(--primary)" : stubOnly ? "var(--warn)" : "var(--cyan)"
      );
      const embedding = health.embedding || {};
      setText(
        "st-model",
        embedding.loaded
          ? embedding.loaded_model.replace("facebook/", "")
          : embedding.main_model_cached
          ? "待加载"
          : "未下载",
        embedding.loaded || embedding.main_model_cached ? "var(--primary)" : "var(--warn)"
      );
      setText("st-cache", `${fmt.num(health.storage.cache_size_mb, 0)} MB`);
      setText("st-version", health.version);
    } catch (err) {
      setText("st-health", "不可达", "var(--danger)");
      setText("st-provider", "—");
      setText("st-model", "—");
    }

    try {
      const queue = await get("/jobs/queue");
      state.queue = queue;
      setText(
        "st-jobs",
        `${queue.active_jobs} 运行中 / ${queue.workers} 并发`,
        queue.active_jobs > 0 ? "var(--primary)" : undefined
      );
    } catch (_) {
      setText("st-jobs", "—");
    }
  }

  /* ---------------- 通用渲染片段 ---------------- */
  const render = {
    alert(message, kind = "info", icon) {
      const icons = { info: "i", ok: "✓", warn: "!", danger: "✕" };
      return `<div class="alert ${kind}"><span class="ico">${icon || icons[kind] || "i"}</span><span>${message}</span></div>`;
    },
    empty(text, sub = "") {
      return `<div class="empty"><div class="big">◇</div><div>${fmt.escape(text)}</div>${
        sub ? `<div class="text-xs mt-1">${fmt.escape(sub)}</div>` : ""
      }</div>`;
    },
    skeleton(lines = 4) {
      return Array.from({ length: lines })
        .map((_, i) => `<div class="skeleton skeleton-line" style="width:${88 - i * 9}%"></div>`)
        .join("");
    },
    stat(label, value, { unit = "", foot = "", plain = false, small = false } = {}) {
      return `<div class="stat">
        <div class="stat-label">${label}</div>
        <div class="stat-value${plain ? " plain" : ""}${small ? " sm" : ""}">${value}${
        unit ? `<span class="stat-unit">${unit}</span>` : ""
      }</div>
        ${foot ? `<div class="stat-foot">${foot}</div>` : ""}
      </div>`;
    },
    miniBar(score, { max = 100, label = null } = {}) {
      const pct = Math.max(0, Math.min(100, (score / max) * 100));
      return `<span class="mini-bar"><span class="bar"><i style="width:${pct}%;background:${fmt.scoreColor(
        (score / max) * 100
      )}"></i></span><span class="val">${label === null ? Math.round(score) : label}</span></span>`;
    },
    progress(value) {
      return `<div class="progress"><i style="width:${Math.round(value * 100)}%"></i></div>`;
    },
    jobRow(job) {
      const kindLabel = CONST.JOB_KIND_LABEL[job.kind] || job.kind;
      const short = CONST.JOB_KIND_SHORT[job.kind] || "??";
      const failed = job.status === "failed";
      return `<div class="job-row">
        <div class="job-kind ${failed ? "failed" : job.kind}">${short}</div>
        <div class="job-meta">
          <div class="job-title">${fmt.escape(job.title || kindLabel)}</div>
          <div class="job-stage">
            <span>${fmt.escape(job.stage || job.status)}</span>
            ${
              job.status === "running"
                ? `<span class="progress" style="flex:1;max-width:150px"><i style="width:${Math.round(
                    (job.progress || 0) * 100
                  )}%"></i></span>`
                : ""
            }
            ${
              failed
                ? `<span class="text-danger text-xs">${fmt.escape(
                    (job.error || "").slice(0, 90)
                  )}</span>`
                : ""
            }
          </div>
        </div>
        <div class="row" style="gap:8px">
          <span class="job-pct">${
            job.status === "success"
              ? "完成"
              : job.status === "failed"
              ? "失败"
              : job.status === "cancelled"
              ? "已取消"
              : Math.round((job.progress || 0) * 100) + "%"
          }</span>
          <span class="text-xs text-3 nowrap">${fmt.relativeTime(job.created_at)}</span>
        </div>
      </div>`;
    },
  };

  /* ---------------- 初始化 ---------------- */
  async function init(activeKey) {
    mountShell(activeKey);
    await loadProjects();
    await refreshStatus();
    // 状态栏每 12 秒刷新一次；作业在跑时缩短为 4 秒
    setInterval(() => {
      const active = state.queue && state.queue.active_jobs > 0;
      if (active) refreshStatus();
    }, 4000);
    setInterval(refreshStatus, 12000);
  }

  global.App = {
    init,
    toast,
    toastError,
    render,
    state,
    store,
    get,
    currentProjectId,
    refreshStatus,
    loadProjects,
    PAGES,
  };
})(window);
