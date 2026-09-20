/* ==========================================================================
   突变设计页逻辑
   --------------------------------------------------------------------------
   页面职责：
   * 参数配置与作业提交（含分阶段进度）
   * 候选排序表（可排序 / 可过滤 / 可勾选对比）
   * 可解释评分卡（瀑布式分解 + 逐维度中文依据）
   * 3D 结构热点联动
   * 扫描透明度（被排除位点与保护位点，让用户知道"为什么没推荐某位点"）
   ========================================================================== */

(function (global) {
  "use strict";

  const state = {
    deps: null,
    result: null,
    candidates: [],
    filtered: [],
    selectedIndex: -1,
    sortKey: "total_score",
    sortDir: -1,
    minScore: null,
    viewer: null,
    catalog: null,
    proteinType: "generic",
    mode: "single",
    structureId: null,
    pdbLoaded: false,
    manualPreview: null,
  };

  const MODE_HINT = {
    single: "单点突变：对每个可突变位点的 19 种替换全部打分排序，适合逐步验证。",
    combination: "多点组合：在单点基础上做束搜索，并对头部组合做上下文依赖的精确重评（考虑上位效应）。耗时明显更长。",
    local: "局部序列优化：仅在指定区段内做保守替换（体积等级与极性不变），适合「不想大改」的场景。",
    manual:
      "人工指定突变：位点、替换氨基酸与锁定位点全部由人填写，平台只按同一套评分链路逐条评估。对应需求「验证 MVP 阶段开发」的人工工作流。",
  };

  const { fmt } = global.API;

  /* ------------------------------------------------------------------ */
  function el(id) {
    return state.deps.el(id);
  }

  async function bindCatalog() {
    try {
      state.catalog = await global.API.get("/design/catalog");
    } catch (err) {
      global.App.toastError(err);
      return;
    }

    const types = state.catalog.protein_types || [];
    el("protein-types").innerHTML = types
      .map(
        (type) =>
          `<span class="chip${type.key === state.proteinType ? " active" : ""}" data-type="${type.key}" title="${fmt.escape(
            type.description || ""
          )}">${fmt.escape(type.label)}</span>`
      )
      .join("");
    el("protein-types").querySelectorAll(".chip").forEach((chip) =>
      chip.addEventListener("click", () => {
        state.proteinType = chip.dataset.type;
        el("protein-types").querySelectorAll(".chip").forEach((item) => item.classList.remove("active"));
        chip.classList.add("active");
      })
    );

    const modes = state.catalog.modes || [];
    el("modes").innerHTML = modes
      .map(
        (mode) => `<span class="chip${mode.key === state.mode ? " active" : ""}" data-mode="${mode.key}">${mode.label}</span>`
      )
      .join("");
    el("modes").querySelectorAll(".chip").forEach((chip) =>
      chip.addEventListener("click", () => {
        state.mode = chip.dataset.mode;
        el("modes").querySelectorAll(".chip").forEach((item) => item.classList.remove("active"));
        chip.classList.add("active");
        applyModeVisibility();
      })
    );
    applyModeVisibility();

    el("host").innerHTML = global.API.CONST.HOST_SYSTEMS.map(
      (host) => `<option value="${host.key}">${host.label}</option>`
    ).join("");
  }

  /* ------------------------------------------------------------------ */
  /**
   * 按当前模式决定各参数区的显隐。
   *
   * 三个参数区互斥关系：
   * * 组合参数（最多位点 / 束宽）—— 仅 combination 与 local 有意义；
   * * 区段参数 —— 定义"扫描范围"，人工模式下位点由人直接指定，范围无意义；
   * * 人工参数区 —— 仅 manual 模式。
   * 让不适用的控件留在界面上是最容易误导实验人员的地方（填了却不生效），
   * 因此这里直接隐藏，而不是留在那里当装饰。
   */
  function applyModeVisibility() {
    const isManual = state.mode === "manual";
    const isSingle = state.mode === "single";
    el("mode-hint").textContent = MODE_HINT[state.mode] || "";
    el("combo-params").style.display = isSingle || isManual ? "none" : "";
    const region = el("region-params");
    if (region) region.style.display = isManual ? "none" : "";
    el("manual-params").classList.toggle("hidden", !isManual);
    if (isManual) renderAaEditor();
  }

  /* ------------------------------------------------------------------ */
  async function run() {
    const sequence = el("sequence").value.trim();
    if (!sequence) {
      global.App.toast("请先输入序列", "warn");
      return;
    }

    const regionStart = Number(el("region-start").value) || null;
    const regionEnd = Number(el("region-end").value) || null;
    let region = null;
    if (regionStart && regionEnd) {
      if (regionStart >= regionEnd) {
        global.App.toast("区段起点必须小于终点", "warn");
        return;
      }
      region = [regionStart, regionEnd];
    }

    // 人工模式先把人工参数收齐并做前端校验：不通过就**根本不提交**。
    // 否则要等结构预测 + 打分跑完（数十秒到数分钟）才在服务端报"位点冲突"，
    // 白等一轮。服务端仍会做一次独立校验（前端校验只是提前拦截）。
    let manualFields = null;
    if (state.mode === "manual") {
      const manual = manualPayload();
      if (!manual) return;
      manualFields = {
        name: manual.name,
        target_positions: manual.target_positions,
        locked_positions: manual.locked_positions,
        substitutions: manual.substitutions,
        mutation_notes: manual.mutation_notes,
      };
    }

    const button = el("btn-run");
    button.disabled = true;
    button.textContent = "设计中…";
    el("alerts").innerHTML = "";

    try {
      const accepted = await global.API.post("/design/run", {
        sequence,
        protein_type: state.proteinType,
        mode: state.mode,
        host_system: el("host").value,
        region,
        top_n: Number(el("top-n").value) || 50,
        max_sites: Number(el("max-sites").value) || 3,
        beam_width: Number(el("beam-width").value) || 8,
        use_structure: el("use-structure").checked,
        project_id: global.App.currentProjectId(),
        ...(manualFields || {}),
      });

      const result = await global.API.pollJob(accepted.job_id, {
        onProgress: (job) => {
          el("job-status").innerHTML = `<span class="badge info">${fmt.escape(job.stage || "处理中")} ${Math.round(
            (job.progress || 0) * 100
          )}%</span>`;
        },
      });

      render(result);
      el("job-status").innerHTML = `<span class="badge ok">完成 · ${fmt.duration(
        (result.summary && result.summary.elapsed_seconds) || 0
      )}</span>`;
      global.App.toast(`设计完成：${(result.candidates || []).length} 条候选`, "ok");
    } catch (err) {
      el("job-status").innerHTML = '<span class="badge danger">失败</span>';
      global.App.toastError(err);
    } finally {
      button.disabled = false;
      button.textContent = "开始设计";
    }
  }

  /* ------------------------------------------------------------------ */
  function render(result) {
    state.result = result;
    state.candidates = result.candidates || [];
    state.structureId = result.structure && result.structure.structure_id;
    state.pdbLoaded = false;
    // 结构在作业结果里就已经拿到了，下载入口应当**立即**可用。
    // 只靠 highlightStructure 里的调用不够——那个函数仅在用户点开某个候选时触发，
    // 结果是设计跑完后按钮一直灰着，必须点一次候选才亮，容易被当成功能没做好。
    setViewerPdbLink(state.structureId);

    el("result").classList.remove("hidden");
    el("btn-export-csv").disabled = !result.design_run_id;
    el("btn-export-md").disabled = !result.design_run_id;

    const warnings = result.warnings || [];
    const structureWarnings =
      result.structure && result.structure.degradation_reason ? [result.structure.degradation_reason] : [];
    el("alerts").innerHTML = [...warnings, ...structureWarnings]
      .map((warning) => global.App.render.alert(fmt.escape(warning), "warn"))
      .join("");

    renderSummary(result);
    renderPlan(result.plan || {});
    renderHints(result.hints || []);
    applyFilter();
    renderCombos(result.combinations || []);
  }

  function renderSummary(result) {
    const summary = result.summary || {};
    el("summary-cards").innerHTML = [
      global.App.render.stat("候选总数", fmt.int(summary.candidate_count), { foot: "按总分降序" }),
      global.App.render.stat("最高分", fmt.num(summary.score_max, 2), { foot: "最佳单点方案" }),
      global.App.render.stat("平均分", fmt.num(summary.score_mean, 2), { foot: "全体候选均值" }),
      global.App.render.stat("组合候选", fmt.int(summary.combination_count), {
        foot: summary.combination_count ? "含上位效应精评" : "单点模式",
        plain: !summary.combination_count,
      }),
      global.App.render.stat("带告警候选", fmt.int(summary.flagged_count), {
        foot: "引入了新的风险位点", plain: true,
      }),
      global.App.render.stat("扫描位点", fmt.int((result.plan || {}).position_count), {
        foot: `共评估 ${fmt.int(summary.evaluated_candidates)} 条`,
        plain: true,
      }),
    ].join("");
  }

  function renderPlan(plan) {
    const protectedMap = plan.protected || {};
    const protectedKeys = Object.keys(protectedMap);
    const exclusions = plan.exclusions || [];

    // 排除原因归类统计
    const reasons = {};
    exclusions.forEach((item) => {
      const key = item.reason.split("（")[0].split("：")[0].slice(0, 28);
      reasons[key] = (reasons[key] || 0) + 1;
    });

    el("plan-sub").textContent = `${plan.position_count} 个位点参与扫描 · ${plan.candidate_count} 条候选`;
    el("plan-body").innerHTML = `
      ${
        plan.downsampled
          ? global.App.render.alert(fmt.escape(plan.downsample_note || "已降采样"), "warn")
          : ""
      }
      <div class="row between" style="padding:6px 0;border-bottom:1px solid var(--hairline)">
        <span class="text-xs text-1">参与扫描位点</span>
        <span class="mono text-xs">${fmt.int(plan.position_count)}</span></div>
      <div class="row between" style="padding:6px 0;border-bottom:1px solid var(--hairline)">
        <span class="text-xs text-1">受保护位点（规则包）</span>
        <span class="mono text-xs text-warn">${fmt.int(protectedKeys.length)}</span></div>
      <div class="row between" style="padding:6px 0;border-bottom:1px solid var(--hairline)">
        <span class="text-xs text-1">被排除位点</span>
        <span class="mono text-xs">${fmt.int(exclusions.length)}</span></div>
      <div class="mt-2">
        <div class="text-xs text-2 mb-1">排除原因分布</div>
        ${Object.entries(reasons)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 6)
          .map(
            ([reason, count]) => `
          <div class="row between" style="padding:3px 0">
            <span class="text-xs text-1">${fmt.escape(reason)}</span>
            <span class="mono text-xs">${count}</span>
          </div>`
          )
          .join("") || '<div class="text-xs text-2">无</div>'}
      </div>
      ${
        protectedKeys.length
          ? `<details class="acc mt-2"><summary>受保护位点明细（前 60 个）</summary>
               <div class="acc-body">${protectedKeys
                 .slice(0, 60)
                 .map(
                   (key) =>
                     `<div style="padding:3px 0"><span class="mono text-primary">${key}</span> · ${fmt.escape(
                       protectedMap[key]
                     )}</div>`
                 )
                 .join("")}</div></details>`
          : ""
      }
      <div class="text-xs text-2 mt-2">
        未参与扫描的位点不会被推荐，但也不代表它们"不可改造"——通常是末端、受保护位点或超出区段设置。
      </div>
    `;
  }

  function renderHints(hints) {
    el("hints-body").innerHTML =
      hints.map((hint) => `<div class="alert info" style="align-items:flex-start">
        <span class="ico">›</span><span>${fmt.escape(hint)}</span></div>`).join("") ||
      '<div class="text-2 text-sm">当前蛋白类型暂无特定策略建议。</div>';
  }

  /* ------------------------------------------------------------------ */
  function applyFilter() {
    const raw = el("min-score").value;
    state.minScore = raw === "" ? null : Number(raw);
    state.filtered = state.candidates.filter(
      (candidate) => state.minScore === null || candidate.total_score >= state.minScore
    );
    sortCandidates();
    renderTable();
  }

  function sortCandidates() {
    const key = state.sortKey;
    const dir = state.sortDir;
    state.filtered.sort((a, b) => {
      let left;
      let right;
      if (key === "total_score" || key === "site_count") {
        left = a[key];
        right = b[key];
      } else if (key === "mutations") {
        left = a.mutations.join(",");
        right = b.mutations.join(",");
      } else if (key === "delta_logp") {
        // 原始读数嵌在 detail 里，不走上面对 dimensions 的通用取值
        left = (a.detail && a.detail.delta_logp) || 0;
        right = (b.detail && b.detail.delta_logp) || 0;
      } else {
        left = dimensionRaw(a, key);
        right = dimensionRaw(b, key);
      }
      if (typeof left === "string") return dir * left.localeCompare(right);
      const diff = (left || 0) - (right || 0);
      if (diff !== 0) return dir * diff;
      // ---------- 并列兜底 ----------
      // 稳定性维度把 ΔlogP >= 0 全部映射为满分 100，因此**并列极其常见**
      // （实测 W23A/W23S/W23L/W23T 四项总分都是 71.00、五维分数完全相同）。
      // 若不做兜底，同一批替换的先后完全取决于后端排序的偶然结果，
      // 用户看到的第一条并不是"最优"，而是"碰巧排在前面"。
      // 用原始 ΔlogP 兜底后，并列内部的次序有明确含义且可复现。
      const leftDelta = (a.detail && a.detail.delta_logp) || 0;
      const rightDelta = (b.detail && b.detail.delta_logp) || 0;
      return dir * (leftDelta - rightDelta);
    });
  }

  function dimensionRaw(candidate, key) {
    const dimension = (candidate.dimensions || []).find((item) => item.key === key);
    return dimension ? dimension.raw : 0;
  }

  const COLUMNS = [
    { key: "rank", label: "#", sortable: false },
    { key: "mutations", label: "突变", sortable: true },
    { key: "total_score", label: "总分", sortable: true },
    { key: "delta_logp", label: "ΔlogP", sortable: true },
    { key: "stability", label: "稳定性", sortable: true },
    { key: "activity", label: "活性", sortable: true },
    { key: "foldability", label: "折叠", sortable: true },
    { key: "expression", label: "表达", sortable: true },
    { key: "risk", label: "安全性", sortable: true },
    { key: "flags", label: "告警", sortable: false },
  ];

  /**
   * 当前应显示的列。
   *
   * ΔlogP 由后端的 ``evaluate_single`` 统一写入 ``detail``，与设计模式无关；
   * 但**从数据库载入的历史批次**没有持久化 ``detail``（``MutationCandidate``
   * 只存总分与各维度），此时整列都是"—"。这种情况直接隐藏该列，
   * 而不是让用户对着一列横杠猜。
   */
  function visibleColumns() {
    const hasDelta = state.candidates.some(
      (candidate) =>
        candidate.detail &&
        candidate.detail.delta_logp !== undefined &&
        candidate.detail.delta_logp !== null
    );
    return hasDelta ? COLUMNS : COLUMNS.filter((column) => column.key !== "delta_logp");
  }

  /**
   * ΔlogP 单元格（ESM-2 掩码边缘似然比）。
   *
   * 为什么必须把它单独列出来：稳定性维度已经把 ΔlogP 线性映射到 0-100，
   * 而映射上界取 0（配置 ``zero_shot.delta_best``）——也就是 **ΔlogP ≥ 0
   * 一律记满分 100**。于是同一位置的若干替换会全部顶格、总分完全并列：
   * 实测 W23A/W23S/W23L/W23T 四项总分都是 71.00、稳定性都是 100，
   * 光看分数无法判断该挑哪个。原始 ΔlogP 保留了差异
   * （0.53 / 0.99 / 1.06 / 0.82），这才是排序与取舍的依据。
   */
  function deltaCell(candidate) {
    const value = candidate.detail && candidate.detail.delta_logp;
    if (value === undefined || value === null) return '<span class="text-3">—</span>';
    const tone = value > 0 ? "ok" : value > -1 ? "info" : value > -3 ? "warn" : "danger";
    const tip = "ESM-2 掩码边缘似然比 ΔlogP：>0 表示模型偏好该替换。稳定性分满分只代表 ΔlogP ≥ 0，看不出具体优多少，需要看这里的原始值。";
    return `<span class="badge ${tone}" title="${fmt.escape(tip)}">${
      value > 0 ? "+" : ""
    }${value.toFixed(2)}</span>`;
  }

  function renderTable() {
    const host = el("candidate-table");
    el("candidate-sub").textContent = `${state.filtered.length} / ${state.candidates.length} 条`;

    if (!state.filtered.length) {
      host.innerHTML = global.App.render.empty(
        state.candidates.length ? "当前过滤条件下没有候选" : "没有候选方案",
        state.candidates.length ? "请放宽最低分限制" : "请检查序列与区段设置"
      );
      return;
    }

    host.innerHTML = `
      <table class="data">
        <thead><tr>
          ${visibleColumns().map(
            (column) =>
              `<th class="${column.sortable ? "sortable" : ""} ${
                state.sortKey === column.key ? "sorted" : ""
              }" data-key="${column.key}">${column.label}${
                column.sortable ? ' <span class="arrow">▼</span>' : ""
              }</th>`
          ).join("")}
        </tr></thead>
        <tbody>
          ${state.filtered
            .slice(0, 400)
            .map((candidate, index) => {
              const originalIndex = state.candidates.indexOf(candidate);
              const selected = originalIndex === state.selectedIndex;
              return `<tr class="${selected ? "selected" : ""} ${
                candidate.flags && candidate.flags.length ? "row-flagged" : ""
              }" data-index="${originalIndex}">
                <td class="mono text-2">${index + 1}</td>
                <td class="mutation">${candidate.mutations.join(", ")}</td>
                <td class="num"><span class="score-pill ${fmt.scoreClass(candidate.total_score)}">${fmt.num(
                candidate.total_score,
                2
              )}</span></td>
                <td class="num">${deltaCell(candidate)}</td>
                <td>${mini(candidate, "stability")}</td>
                <td>${mini(candidate, "activity")}</td>
                <td>${mini(candidate, "foldability")}</td>
                <td>${mini(candidate, "expression")}</td>
                <td>${mini(candidate, "risk")}</td>
                <td>${
                  candidate.flags && candidate.flags.length
                    ? `<span class="badge warn" title="${fmt.escape(candidate.flags.join("；"))}">${
                        candidate.flags.length
                      } 项</span>`
                    : '<span class="text-3 text-xs">—</span>'
                }</td>
              </tr>`;
            })
            .join("")}
        </tbody>
      </table>
      ${
        state.filtered.length > 400
          ? `<div class="text-xs text-2" style="padding:10px 12px">仅显示前 400 条，请使用最低分或导出获取完整列表。</div>`
          : ""
      }
    `;

    host.querySelectorAll("thead th.sortable").forEach((th) =>
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        if (state.sortKey === key) state.sortDir *= -1;
        else {
          state.sortKey = key;
          state.sortDir = -1;
        }
        sortCandidates();
        renderTable();
      })
    );

    host.querySelectorAll("tbody tr").forEach((row) =>
      row.addEventListener("click", () => selectCandidate(Number(row.dataset.index)))
    );
  }

  function mini(candidate, key) {
    const dimension = (candidate.dimensions || []).find((item) => item.key === key);
    if (!dimension) return '<span class="text-3">—</span>';
    return `<span class="mini-bar"><span class="bar"><i style="width:${Math.min(
      100,
      dimension.raw
    )}%;background:${fmt.scoreColor(dimension.raw)}"></i></span><span class="val">${Math.round(
      dimension.raw
    )}</span></span>`;
  }

  /* ------------------------------------------------------------------ */
  function selectCandidate(index) {
    if (index < 0 || index >= state.candidates.length) return;
    state.selectedIndex = index;
    renderTable();

    const candidate = state.candidates[index];
    renderScoreCard(candidate);
    highlightStructure(candidate);
  }

  function renderScoreCard(candidate) {
    const dimensions = candidate.dimensions || [];
    el("card-sub").textContent = `${candidate.mutations.join(", ")} · ${fmt.num(candidate.total_score, 2)} 分`;

    const totalContribution = dimensions.reduce((acc, item) => acc + item.contribution, 0);

    el("card-body").innerHTML = `
      ${
        candidate.flags && candidate.flags.length
          ? global.App.render.alert(candidate.flags.map(fmt.escape).join("；"), "warn")
          : global.App.render.alert("该候选未触发风险告警", "ok")
      }
      <div class="mb-1" style="font-size:12.5px;color:var(--text-1);line-height:1.7">
        ${fmt.escape(candidate.rationale || "")}
      </div>
      <div class="chart" id="card-chart" style="height:190px"></div>
      <div class="text-xs text-2 mt-1">
        各维度贡献之和 = ${fmt.num(totalContribution, 2)}（与总分一致，无未归因残差）
      </div>
      <div class="divider"></div>
      ${dimensions
        .map(
          (item) => `
        <div class="dim-card">
          <div class="dim-head">
            <span class="label">${fmt.escape(item.label)}</span>
            <span class="score" style="color:${fmt.scoreColor(item.raw)}">${Math.round(item.raw)}</span>
          </div>
          <div class="contrib-bar mb-1">
            <span class="bar">
              <i class="${item.contribution >= 0 ? "pos" : "neg"}"
                 style="width:${Math.min(50, Math.abs(item.contribution) * 2.2)}%"></i>
            </span>
            <span class="val ${item.contribution >= 0 ? "pos" : "neg"}">
              ${item.contribution >= 0 ? "+" : ""}${fmt.num(item.contribution, 2)}
              <span class="text-3">(权重 ${(item.weight * 100).toFixed(1)}%)</span>
            </span>
          </div>
          <div class="dim-rationale">${fmt.escape(item.rationale || "")}</div>
        </div>`
        )
        .join("")}
      ${
        candidate.detail && candidate.detail.delta_logp !== null && candidate.detail.delta_logp !== undefined
          ? `<details class="acc mt-2"><summary>模型原始信号与结构上下文</summary><div class="acc-body">
              <div>ΔlogP（掩码边缘似然比）：<b class="mono">${fmt.num(candidate.detail.delta_logp, 3)}</b></div>
              <div>野生型掩码对数概率：<b class="mono">${fmt.num(candidate.detail.wt_logprob, 3)}</b></div>
              <div>该位点结构上下文：${fmt.escape(candidate.detail.context || "无")}</div>
              <div>体积变化：<b class="mono">${fmt.num(candidate.detail.deltas && candidate.detail.deltas.volume_change, 1)}</b> Å³ ·
                   电荷变化：<b class="mono">${fmt.num(candidate.detail.deltas && candidate.detail.deltas.charge_change, 2)}</b></div>
              <div>保守替换：${candidate.detail.is_conservative ? "是" : "否"}</div>
              <div class="text-2 mt-1">ΔlogP 是零样本代理指标，与实验 ΔΔG 单调相关但未做 kcal/mol 校准。</div>
            </div></details>`
          : ""
      }
    `;

    global.Charts.contributions(el("card-chart"), dimensions);
  }

  /**
   * 同步「下载 PDB」按钮的可用状态。
   *
   * `<a>` 不支持 `:disabled`，因此用 aria-disabled 表达禁用态（CSS 已配对应样式），
   * 同时移除 href —— 没有 href 的 `<a>` 本身也不可点击，双保险。
   */
  function setViewerPdbLink(structureId) {
    const link = el("btn-viewer-pdb");
    if (!link) return;
    if (structureId) {
      link.href = `/api/structure/${structureId}/pdb`;
      link.setAttribute("download", `structure_${structureId}.pdb`);
      link.removeAttribute("aria-disabled");
    } else {
      link.removeAttribute("href");
      link.setAttribute("aria-disabled", "true");
    }
  }

  /* ------------------------------------------------------------------ */
  async function highlightStructure(candidate) {
    if (!state.structureId) {
      el("viewer-sub").textContent = "无结构数据";
      setViewerPdbLink(null);
      return;
    }
    const container = el("viewer");
    if (!state.viewer) state.viewer = new global.Viewer3D.Viewer(container);

    if (!state.pdbLoaded) {
      try {
        const response = await fetch(`/api/structure/${state.structureId}/pdb`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const pdbText = await response.text();
        state.pdbLoaded = true;
        container.innerHTML = '<div class="viewer-3d" style="width:100%;height:100%"></div>';
        state.viewer.render(pdbText, {
          plddt: (state.result.structure_plddt || []).length ? state.result.structure_plddt : [],
        });
        el("viewer-sub").innerHTML = `<span class="badge">${fmt.escape(
          (state.result.structure || {}).source || "结构"
        )}</span>`;
        setViewerPdbLink(state.structureId);
      } catch (err) {
        container.innerHTML = `<div class="empty" style="padding:28px">无法加载结构：${fmt.escape(
          err.message
        )}</div>`;
        return;
      }
    }

    // 高亮当前候选的全部突变位点
    const highlights = (candidate.positions || []).map((position) => ({ residue: position }));
    state.viewer.setHighlight(highlights);
  }

  /* ------------------------------------------------------------------ */
  function renderCombos(combinations) {
    const host = el("combo-table");
    if (!combinations.length) {
      host.innerHTML = global.App.render.empty(
        "本次未生成组合候选",
        "组合模式会输出多点叠加方案；单点模式下不生成"
      );
      return;
    }

    host.innerHTML = `
      <table class="data">
        <thead><tr>
          <th>#</th><th>组合突变</th><th class="mono">位点数</th>
          <th class="mono">最终分</th><th class="mono">估计分</th>
          <th class="mono">上位惩罚</th><th>精评</th>
        </tr></thead>
        <tbody>
          ${combinations
            .map(
              (item, index) => `
            <tr>
              <td class="mono text-2">${index + 1}</td>
              <td class="mutation">${item.mutations.join(", ")}</td>
              <td class="num">${item.mutations.length}</td>
              <td class="num"><span class="score-pill ${fmt.scoreClass(item.final_score)}">${fmt.num(
                item.final_score,
                2
              )}</span></td>
              <td class="num text-2">${fmt.num(item.estimated_score, 2)}</td>
              <td class="num ${item.epitasis_penalty > 0 ? "text-warn" : "text-3"}">${fmt.num(
                item.epitasis_penalty,
                2
              )}</td>
              <td>${
                item.refined
                  ? '<span class="badge primary">已精评</span>'
                  : '<span class="badge">估计值</span>'
              }</td>
            </tr>`
            )
            .join("")}
        </tbody>
      </table>
      <div class="text-xs text-2" style="padding:10px 12px">
        「最终分」为上下文依赖的精确重评分（考虑上位效应）；「估计分」为单点分数平均后扣除上位惩罚的快速估计。
        位点间距过近时会施加惩罚——结构上相邻的位点彼此扰动，单点效应不能简单叠加。
      </div>
    `;
  }

  /* ------------------------------------------------------------------ */
  function exportRun(format) {
    if (!state.result || !state.result.design_run_id) {
      global.App.toast("本次结果未落库，无法导出（请确认已选择项目）", "warn");
      return;
    }
    const limit = Math.max(1, Math.min(2000, Number(el("top-n").value) || 20));
    const url = `/api/design/runs/${state.result.design_run_id}/export?format=${format}&top_n=${limit}`;
    window.open(url, "_blank");
    global.App.toast(`正在导出${format === "csv" ? " CSV" : " Markdown 报告"}…`, "info");
  }

  function loadStoredRun(run, candidates) {
    // 把入库的候选转换为与作业结果一致的结构，复用渲染逻辑
    const normalized = (candidates.items || []).map((item) => ({
      mutations: item.mutations,
      positions: item.positions,
      total_score: item.total_score,
      dimensions: item.dimensions,
      flags: item.flags,
      rationale: item.rationale,
      category: run.category,
      detail: {},
    }));

    render({
      length: run.sequence ? run.sequence.length : 0,
      protein_type: run.category,
      mode: run.mode,
      plan: { position_count: 0, candidate_count: normalized.length, exclusions: [], protected: {} },
      candidates: normalized,
      combinations: [],
      summary: run.summary || {},
      hints: [],
      warnings: ["这是从数据库载入的历史批次，扫描透明度信息与组合候选未持久化。"],
      rulepack: {},
      design_run_id: run.id,
      structure: null,
    });
  }

  /* ==================================================================== */
  /* 人工指定突变（验证 MVP 工作流）                                        */
  /* ==================================================================== */

  const AA_LIST = "ACDEFGHIKLMNPQRSTVWY".split("");

  //: 位点 -> 人工勾选的候选氨基酸集合。**空集合语义是"未指定"**，
  //: 此时服务端会按残基类别套用默认集合（含丙氨酸扫描），而不是"不突变"。
  const manualSelection = new Map();

  /** 当前序列（去掉 FASTA 头与非字母字符）。仅用于界面显示野生型残基。 */
  function currentSequence() {
    return el("sequence")
      .value.split(/\r?\n/)
      .filter((line) => !line.trim().startsWith(">"))
      .join("")
      .replace(/[^A-Za-z]/g, "")
      .toUpperCase();
  }

  /**
   * 解析人工输入的位点串。
   *
   * 容忍中英文逗号、分号、顿号与空格混用（实验人员从 Excel 或文献里
   * 复制粘贴时格式很不统一）。非法项**直接忽略**——这里只是界面辅助，
   * 原始输入仍会原样送给服务端做权威校验，服务端对非法位点是明确报错的。
   */
  function parsePositions(raw) {
    const out = [];
    String(raw || "")
      .split(/[\s,;，；、|]+/)
      .forEach((token) => {
        const text = token.trim();
        if (!text) return;
        const value = Number(text);
        if (Number.isInteger(value) && value >= 1 && out.indexOf(value) === -1) {
          out.push(value);
        }
      });
    return out.sort((a, b) => a - b);
  }

  /** 渲染每位点的氨基酸选择器（20 格点选）。 */
  function renderAaEditor() {
    const host = el("manual-aa-editor");
    if (!host) return;
    const targets = parsePositions(el("manual-targets").value);
    const sequence = currentSequence();

    if (!targets.length) {
      host.innerHTML =
        '<div class="empty" style="padding:12px">填写目标位点后，这里会列出每位点的氨基酸选择器</div>';
      return;
    }

    host.innerHTML = targets
      .map((position) => {
        const wild = sequence[position - 1] || "?";
        const selected = manualSelection.get(position) || new Set();
        const chips = AA_LIST.map((aa) => {
          const active = selected.has(aa) ? " active" : "";
          // 野生型残基标记出来：勾它在语义上等于"没有突变"，服务端会拒绝
          const wildFlag = aa === wild ? " wild" : "";
          return `<span class="chip${active}${wildFlag}" data-position="${position}" data-aa="${aa}">${aa}</span>`;
        }).join("");
        return `<div class="manual-aa-row">
            <span class="manual-aa-pos">位点 ${position}</span>
            <span class="manual-aa-wt" title="野生型残基">${wild === "?" ? "—" : fmt.escape(wild)}</span>
            <span class="manual-aa-chips">${chips}</span>
            <button class="btn btn-sm" type="button" data-clear="${position}">清空</button>
          </div>`;
      })
      .join("");

    host.querySelectorAll(".manual-aa-chips .chip").forEach((chip) =>
      chip.addEventListener("click", () => {
        const position = Number(chip.dataset.position);
        const aa = chip.dataset.aa;
        if (!manualSelection.has(position)) manualSelection.set(position, new Set());
        const set = manualSelection.get(position);
        if (set.has(aa)) {
          set.delete(aa);
          chip.classList.remove("active");
        } else {
          set.add(aa);
          chip.classList.add("active");
        }
        markPreviewStale();
      })
    );

    host.querySelectorAll("button[data-clear]").forEach((button) =>
      button.addEventListener("click", () => {
        manualSelection.delete(Number(button.dataset.clear));
        renderAaEditor();
        markPreviewStale();
      })
    );
  }

  /**
   * 参数一改动就把已预览的清单标记为过期。
   *
   * 必要性：导出的文件是**按预览时的参数**在服务端重新生成的。若用户改了位点
   * 却直接点导出，会拿到一份"和屏幕上显示的不一致"的文件——这类不一致极难察觉，
   * 因此这里主动作废预览并禁用导出，强制重新预览。
   */
  function markPreviewStale() {
    if (!state.manualPreview) return;
    state.manualPreview = null;
    el("btn-manual-fasta").disabled = true;
    el("btn-manual-csv").disabled = true;
    el("manual-summary").textContent = "参数已改动，请重新预览";
  }

  /** 收集人工模式参数；校验不通过返回 null。 */
  function manualPayload() {
    const sequence = el("sequence").value.trim();
    if (!sequence) {
      global.App.toast("请先输入序列", "warn");
      return null;
    }
    const targets = parsePositions(el("manual-targets").value);
    if (!targets.length) {
      global.App.toast("人工模式至少需要填写一个目标位点", "warn");
      return null;
    }

    const substitutions = {};
    manualSelection.forEach((set, position) => {
      if (set.size) substitutions[String(position)] = Array.from(set).sort();
    });

    const note = el("manual-note").value.trim();
    const notes = {};
    if (note) targets.forEach((position) => (notes[String(position)] = note));

    return {
      sequence,
      name: el("manual-name").value.trim() || "Protein",
      protein_type: state.proteinType,
      target_positions: targets,
      locked_positions: parsePositions(el("manual-locked").value),
      substitutions: Object.keys(substitutions).length ? substitutions : null,
      mutation_notes: Object.keys(notes).length ? notes : null,
      include_wild_type: true,
    };
  }

  /** 预览：调用规划接口，展示"平台将评估哪些突变"以及被拒位点。 */
  async function previewManual() {
    const payload = manualPayload();
    if (!payload) return;
    const button = el("btn-manual-preview");
    button.disabled = true;
    button.textContent = "规划中…";
    try {
      const plan = await global.API.post("/design/manual/plan", payload);
      state.manualPreview = plan;
      renderManualPlan(plan);
    } catch (err) {
      global.App.toastError(err);
    } finally {
      button.disabled = false;
      button.textContent = "预览突变清单";
    }
  }

  function renderManualPlan(plan) {
    const rows = plan.mutations || [];
    const blocked = plan.blocked || [];
    const rejected = plan.rejected_substitutions || [];

    const header = `<tr>
        <th>sequence_id</th><th>突变</th><th>位点</th>
        <th>野生 → 突变</th><th>长度</th><th>备注</th>
      </tr>`;

    // 野生型固定排第一：需求要求所有指标都相对野生型对比，清单里也要能一眼看到基准
    const wildRow = `<tr class="wt-row">
        <td class="mono">${fmt.escape(plan.name)}_WT</td>
        <td class="label">野生型（基准对照）</td>
        <td>—</td><td>—</td>
        <td>${plan.length}</td>
        <td>所有突变均与此对比</td>
      </tr>`;

    const body = rows
      .map(
        (item) => `<tr>
        <td class="mono" title="${fmt.escape(item.sequence_id)}">${fmt.escape(item.sequence_id)}</td>
        <td class="label">${fmt.escape(item.label)}</td>
        <td>${item.position}</td>
        <td>${fmt.escape(item.wild_type)} → ${fmt.escape(item.mutant)}</td>
        <td>${item.sequence_length}</td>
        <td>${fmt.escape(item.note || "—")}</td>
      </tr>`
      )
      .join("");

    let verdict;
    if (blocked.length) {
      verdict = `<div class="manual-conflict">
        <div class="title">${blocked.length} 个目标位点被拒绝，已从清单中排除</div>
        ${blocked
          .map(
            (item) =>
              `位点 ${item.position}${item.residue ? `（${fmt.escape(item.residue)}）` : ""}：${fmt.escape(
                item.reason
              )}`
          )
          .join("<br>")}
        <div style="margin-top:5px">提交与导出都会被拒绝，请先修正位点或锁定设置。</div>
      </div>`;
    } else {
      verdict = `<div class="manual-ok">全部目标位点通过校验：${rows.length} 条突变 + 1 条野生对照 = ${
        rows.length + 1
      } 条 FASTA 记录（含野生型共 ${rows.length + 1} 条序列）。</div>`;
    }

    const rejectedNote = rejected.length
      ? `<div class="manual-conflict" style="border-color:rgba(245,158,11,0.35);background:rgba(245,158,11,0.08)">
          <div class="title" style="color:var(--warn)">${rejected.length} 条替换被跳过</div>
          ${rejected
            .map((item) => `位点 ${item.position} → ${fmt.escape(item.mutant)}：${fmt.escape(item.reason)}`)
            .join("<br>")}
        </div>`
      : "";

    el("manual-preview").innerHTML =
      `<div class="manual-list"><table>${header}${wildRow}${body}</table></div>${verdict}${rejectedNote}`;

    const usable = !blocked.length && rows.length > 0;
    el("btn-manual-fasta").disabled = !usable;
    el("btn-manual-csv").disabled = !usable;
    el("manual-summary").textContent = usable
      ? `${rows.length} 条突变 · 覆盖 ${plan.position_count} 个位点 · 含野生对照 ${rows.length + 1} 条`
      : `未通过校验（${blocked.length} 个位点被拒）`;
  }

  /**
   * 导出突变清单（FASTA / CSV）。
   *
   * 用 POST + Blob 而不是 ``window.open(url)``：导出请求体里带的是嵌套的
   * 位点与替换结构，塞进 URL 既难看又有长度限制。文件名以服务端
   * ``Content-Disposition`` 为准，保证与后端生成的命名一致。
   */
  async function downloadManual(format) {
    const payload = manualPayload();
    if (!payload) return;
    try {
      const response = await fetch(`/api/design/manual/export?format=${format}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.message || `导出失败（HTTP ${response.status}）`);
      }
      const disposition = response.headers.get("content-disposition") || "";
      const matched = /filename="?([^";]+)"?/.exec(disposition);
      const filename = matched
        ? matched[1]
        : `${payload.name}_mutations.${format === "fasta" ? "fasta" : "csv"}`;

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      global.App.toast(`已导出 ${filename}`, "ok");
    } catch (err) {
      global.App.toastError(err);
    }
  }

  /* ------------------------------------------------------------------ */
  function init(deps) {
    state.deps = deps;
    el("btn-run").addEventListener("click", run);
    el("sequence").addEventListener("input", () => {
      el("seq-len").textContent = `${fmt.sequenceLength(el("sequence").value)} aa`;
      if (state.mode === "manual") renderAaEditor();
    });
    el("btn-sample").addEventListener("click", () => {
      el("sequence").value = global.API.CONST.SAMPLE_SEQUENCE;
      el("seq-len").textContent = `${global.API.CONST.SAMPLE_SEQUENCE.length} aa`;
      if (state.mode === "manual") renderAaEditor();
    });
    el("min-score").addEventListener("input", applyFilter);
    el("btn-export-csv").addEventListener("click", () => exportRun("csv"));
    el("btn-export-md").addEventListener("click", () => exportRun("markdown"));

    // 人工模式交互
    el("manual-targets").addEventListener("input", () => {
      renderAaEditor();
      markPreviewStale();
    });
    ["manual-locked", "manual-name", "manual-note"].forEach((id) =>
      el(id).addEventListener("input", markPreviewStale)
    );
    el("btn-manual-preview").addEventListener("click", previewManual);
    el("btn-manual-fasta").addEventListener("click", () => downloadManual("fasta"));
    el("btn-manual-csv").addEventListener("click", () => downloadManual("csv"));

    window.addEventListener("resize", () => global.Charts.resizeAll());
  }

  global.Design = { init, bindCatalog, loadStoredRun, render };
})(window);
