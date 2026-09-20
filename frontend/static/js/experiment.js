/* ==========================================================================
   实验与迭代页逻辑
   --------------------------------------------------------------------------
   三块能力：
   1. 批量导入（列名自动映射 + 逐行校验 + 预演）；
   2. 预测-实测对比（Spearman 秩相关为主指标 + 线性校准后的误差）；
   3. 属性头训练与版本回滚。
   ========================================================================== */

(function (global) {
  "use strict";

  const { get, post, fmt } = global.API;
  const App = global.App;
  const el = (id) => document.getElementById(id);

  let sequences = [];
  let currentRecords = [];

  /* ------------------------------------------------------------------ */
  /* 模板与属性                                                           */
  /* ------------------------------------------------------------------ */
  async function loadTemplate() {
    try {
      const columns = await get("/experiment/template");
      el("template-body").innerHTML = `
        <div class="table-wrap" style="max-height:340px">
          <table class="data">
            <thead><tr><th>列名</th><th>说明</th><th>示例</th><th>必填</th></tr></thead>
            <tbody>${columns
              .map(
                (column) => `<tr style="cursor:default">
                  <td class="mono">${fmt.escape(column.name)}<div class="text-xs text-2">${fmt.escape(
                  column.label
                )}</div></td>
                  <td class="text-xs">${fmt.escape(column.description)}</td>
                  <td class="mono text-xs">${fmt.escape(column.example)}</td>
                  <td>${column.required ? '<span class="badge warn">必填</span>' : '<span class="badge">选填</span>'}</td>
                </tr>`
              )
              .join("")}</tbody>
          </table>
        </div>
        <div class="text-xs text-2 mt-2">
          列名支持中英文别名自动识别（如「突变体 / mutant / variant」都会映射到 mutation）。
          无法识别的列会在导入结果中列出，不会静默丢弃。
        </div>
      `;
    } catch (err) {
      App.toastError(err);
    }
  }

  async function loadSequences() {
    try {
      const page = await get("/sequences", { limit: 200, with_sequence: false });
      sequences = page.items || [];
      const options = sequences
        .map(
          (item) =>
            `<option value="${item.id}">${fmt.escape(item.name)}（${item.length} aa · ${
              item.extra && item.extra.design_count ? item.extra.design_count + " 批次" : "无设计"
            }）</option>`
        )
        .join("");
      el("compare-sequence").innerHTML = options || '<option value="">（项目内暂无序列）</option>';
      // 导入面板也要选序列：记录必须带上 sequence_id 才能参与预测-实测对比，
      // 否则对比接口按 sequence_id 过滤时一条都取不到。
      const importSelect = document.getElementById("import-sequence");
      if (importSelect) {
        const previous = importSelect.value;
        importSelect.innerHTML =
          '<option value="">（按文件内「序列/样品名称」自动匹配）</option>' + options;
        if (previous) importSelect.value = previous;
      }
    } catch (err) {
      App.toastError(err);
    }
  }

  async function loadProperties() {
    try {
      const data = await get("/model/properties", { min_records: 1 });
      const properties = data.properties || [];
      el("record-property").innerHTML =
        '<option value="">全部属性</option>' +
        properties
          .map(
            (item) =>
              `<option value="${item.property_name}">${fmt.escape(item.property_name)}（${item.record_count}）</option>`
          )
          .join("");

      el("train-property").innerHTML =
        properties
          .map(
            (item) =>
              `<option value="${item.property_name}">${fmt.escape(item.property_name)} · ${
                item.distinct_samples || item.record_count
              } 个可用样本</option>`
          )
          .join("") || '<option value="">（暂无数据）</option>';

      if (properties.length) {
        const first = properties[0];
        el("train-hint").innerHTML = `当前样本最多的是 <b>${fmt.escape(
          first.property_name
        )}</b>（${first.distinct_samples || first.record_count} 个样本）。建议 ≥15 个后再训练。`;
      }
      return properties;
    } catch (err) {
      App.toastError(err);
      return [];
    }
  }

  /* ------------------------------------------------------------------ */
  /* 导入                                                                */
  /* ------------------------------------------------------------------ */
  async function ingestFile(file) {
    if (!file) return;
    const dryRun = el("dry-run").checked;
    el("import-sub").textContent = `正在${dryRun ? "校验" : "导入"} ${file.name}…`;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("project_id", String(App.currentProjectId() || 1));
    formData.append("dry_run", String(dryRun));
    formData.append("deduplicate", String(el("dedup").checked));
    // 关联序列：不传的话记录 sequence_id 为 NULL，预测-实测对比会一条都取不到。
    // 留空则交给后端按文件内的「序列/样品名称」列自动匹配。
    const targetSequence = document.getElementById("import-sequence");
    if (targetSequence && targetSequence.value) {
      formData.append("sequence_id", targetSequence.value);
    }

    try {
      const response = await fetch("/api/experiment/ingest", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.message || `HTTP ${response.status}`);
      }
      renderIngest(payload, file.name, dryRun);
      el("import-sub").textContent = `${file.name} · ${
        dryRun ? "已校验（未写库）" : "已导入"
      }`;
      if (!dryRun) {
        App.toast(`已导入 ${payload.inserted_ids.length} 条实验记录`, "ok");
        await Promise.all([loadRecords(), loadProperties(), loadVersions()]);
      } else {
        App.toast(`校验完成：${payload.report.accepted_rows} 行可用`, "info");
      }
    } catch (err) {
      el("import-sub").textContent = "导入失败";
      App.toast("导入失败：" + err.message, "danger", 9000);
    }
  }

  function renderIngest(payload, filename, dryRun) {
    const report = payload.report || {};
    const mapping = report.detected_columns || {};
    const errors = report.errors || [];

    el("import-result").innerHTML = `
      <div class="grid grid-4" style="gap:10px">
        ${App.render.stat("总行数", fmt.int(report.total_rows), { plain: true, small: true })}
        ${App.render.stat("可用行", fmt.int(report.accepted_rows), { plain: true, small: true })}
        ${App.render.stat("异常行", fmt.int(report.rejected_rows), {
          plain: true, small: true
        })}
        ${App.render.stat(dryRun ? "预演" : "已入库", dryRun ? "未写库" : fmt.int((payload.inserted_ids || []).length), {
          plain: true, small: true
        })}
      </div>

      ${
        report.duplicate_rows
          ? App.render.alert(`检测到 ${report.duplicate_rows} 条与库中完全一致的重复记录，已跳过。`, "info")
          : ""
      }
      ${(report.warnings || []).map((warning) => App.render.alert(fmt.escape(warning), "warn")).join("")}

      <details class="acc mt-2"><summary>列名映射（${Object.keys(mapping).length} 列）</summary>
        <div class="acc-body">
          ${Object.entries(mapping)
            .map(
              ([source, target]) =>
                `<div style="padding:2px 0"><span class="mono text-primary">${fmt.escape(
                  source
                )}</span> → <span class="mono">${fmt.escape(target)}</span></div>`
            )
            .join("")}
          ${
            (report.unmapped_columns || []).length
              ? `<div class="text-warn mt-1">未映射：${report.unmapped_columns
                  .map(fmt.escape)
                  .join("、")}</div>`
              : ""
          }
        </div>
      </details>

      ${
        errors.length
          ? `<details class="acc" open><summary class="text-warn">异常行（${errors.length} 条，最多显示 200）</summary>
               <div class="acc-body">
                 ${errors
                   .slice(0, 200)
                   .map(
                     (error) =>
                       `<div style="padding:2px 0">第 <b class="mono">${error.row}</b> 行 ·
                         <span class="mono">${fmt.escape(error.field || "-")}</span> ·
                         ${fmt.escape(error.message)}</div>`
                   )
                   .join("")}
               </div></details>`
          : App.render.alert("所有数据行均通过校验", "ok")
      }

      ${
        (payload.preview || []).length
          ? `<details class="acc"><summary>数据预览（前 ${payload.preview.length} 条）</summary>
               <div class="acc-body">
                 <table class="data"><thead><tr>
                   <th>行</th><th>突变</th><th>属性</th><th class="mono">实测值</th>
                   <th>单位</th><th>条件</th></tr></thead>
                 <tbody>${payload.preview
                   .map(
                     (item) => `<tr style="cursor:default">
                       <td class="mono">${item.row}</td>
                       <td class="mutation">${fmt.escape(item.mutation || "野生型")}</td>
                       <td><span class="badge ${item.unknown_property ? "warn" : ""}">${fmt.escape(
                       item.property_name
                     )}</span></td>
                       <td class="num">${fmt.num(item.measured_value, 4)}</td>
                       <td class="text-xs">${fmt.escape(item.unit || "—")}</td>
                       <td class="text-xs">${fmt.escape(item.condition || "—")}</td>
                     </tr>`
                   )
                   .join("")}</tbody></table>
               </div></details>`
          : ""
      }

      ${
        dryRun && report.accepted_rows
          ? `<div class="row mt-2"><button class="btn btn-primary btn-sm" id="btn-confirm-import">
               确认导入这 ${report.accepted_rows} 条记录</button>
               <span class="text-xs text-2">取消「仅校验」后重新选择文件，或直接点此按钮</span></div>`
          : ""
      }
    `;

    const confirmButton = el("btn-confirm-import");
    if (confirmButton) {
      confirmButton.addEventListener("click", () => {
        el("dry-run").checked = false;
        App.toast("已切换到导入模式，请重新选择文件", "info");
      });
    }
  }

  /* ------------------------------------------------------------------ */
  /* 对比                                                                */
  /* ------------------------------------------------------------------ */
  async function runCompare() {
    const sequenceId = el("compare-sequence").value;
    if (!sequenceId) {
      App.toast("请先选择序列（可在「序列与结构」页保存序列）", "warn");
      return;
    }
    const button = el("btn-compare");
    button.disabled = true;
    button.textContent = "分析中…";
    el("compare-body").innerHTML = `<div class="text-2 text-sm">正在为每个突变体重算预测值…${
      el("compare-esm").checked ? "（含 ESM-2 精算，较慢）" : ""
    }</div>`;

    try {
      const result = await post(
        "/experiment/compare",
        null,
        {
          sequence_id: Number(sequenceId),
          use_esm: el("compare-esm").checked,
        }
      );
      renderCompare(result);
    } catch (err) {
      el("compare-body").innerHTML = App.render.empty("对比失败", err.message);
      App.toastError(err);
    } finally {
      button.disabled = false;
      button.textContent = "开始对比";
    }
  }

  function renderCompare(result) {
    const properties = result.properties || [];
    if (!properties.length) {
      el("compare-body").innerHTML = `
        ${App.render.alert(fmt.escape(result.overall_verdict || "没有可配对的记录"), "warn")}
        ${
          (result.unmatched_detail || []).length
            ? `<details class="acc"><summary>未配对记录（${
                result.unmatched_detail.length
              } 条）</summary><div class="acc-body">
                 ${result.unmatched_detail
                   .map(
                     (item) =>
                       `<div style="padding:2px 0"><span class="mono">${fmt.escape(
                         item.property_name || "-"
                       )}</span> · ${fmt.escape(item.mutation || "野生型")} · ${fmt.escape(item.reason)}</div>`
                   )
                   .join("")}</div></details>`
            : ""
        }
      `;
      return;
    }

    el("compare-body").innerHTML = `
      <div class="grid grid-4 mb-2" style="gap:10px">
        ${App.render.stat("配对成功", fmt.int(result.matched_pairs), { plain: true, small: true })}
        ${App.render.stat("未配对", fmt.int(result.unmatched_records), { plain: true, small: true })}
        ${App.render.stat("覆盖属性", fmt.int(properties.length), { plain: true, small: true })}
        ${App.render.stat("总记录", fmt.int(result.total_records), { plain: true, small: true })}
      </div>

      ${App.render.alert(fmt.escape(result.overall_verdict), "info")}
      ${(result.methodology && result.methodology.caveat
        ? App.render.alert("方法学提示：" + fmt.escape(result.methodology.caveat), "warn")
        : "")}
      ${(result.recommendations || []).map((item) => App.render.alert(fmt.escape(item), "info")).join("")}

      ${properties
        .map(
          (item, index) => `
        <div class="panel mt-2">
          <div class="panel-head">
            <div class="panel-title"><span class="dot"></span>${fmt.escape(item.label)}
              <span class="panel-sub">${item.n_pairs} 对样本</span></div>
            <span class="badge ${
              item.spearman_rho === null
                ? ""
                : Math.abs(item.spearman_rho) >= 0.6
                ? "ok"
                : Math.abs(item.spearman_rho) >= 0.3
                ? "warn"
                : "danger"
            }">Spearman ρ = ${fmt.num(item.spearman_rho, 3)}</span>
          </div>
          <div class="panel-body">
            <div class="grid grid-2" style="gap:16px">
              <div class="chart" id="compare-scatter-${index}" style="height:280px"></div>
              <div>
                <div class="grid grid-2" style="gap:10px">
                  ${App.render.stat("Pearson r", fmt.num(item.pearson_r, 3), { plain: true, small: true })}
                  ${App.render.stat("校准后 R²", fmt.num(item.r2, 3), { plain: true, small: true })}
                  ${App.render.stat("MAE", fmt.num(item.mae, 3), {
                    unit: item.unit || "", plain: true, small: true
                  })}
                  ${App.render.stat("RMSE", fmt.num(item.rmse, 3), {
                    unit: item.unit || "", plain: true, small: true
                  })}
                </div>
                <div class="text-xs text-2 mt-2">${fmt.escape(item.verdict)}</div>
                <div class="text-xs text-2 mt-1">
                  线性校准：实测 ≈ ${fmt.num(item.calibration.slope, 4)} × 评分 +
                  ${fmt.num(item.calibration.intercept, 2)}
                </div>
                ${
                  item.worst_offsets && item.worst_offsets.length
                    ? `<details class="acc mt-2"><summary>偏差最大的记录</summary><div class="acc-body">
                         ${item.worst_offsets
                           .map(
                             (point) =>
                               `<div style="padding:2px 0"><span class="mono">${fmt.escape(
                                 point.mutation
                               )}</span> · 评分 ${fmt.num(point.predicted_score, 1)} → 实测 ${fmt.num(
                                 point.measured_value,
                                 3
                               )} · 残差 <b class="${
                                 Math.abs(point.residual) > (item.rmse || 0) ? "text-warn" : ""
                               }">${fmt.num(point.residual, 3)}</b></div>`
                           )
                           .join("")}
                       </div></details>`
                    : ""
                }
              </div>
            </div>
          </div>
        </div>`
        )
        .join("")}
    `;

    properties.forEach((item, index) => {
      global.Charts.scatter(
        el(`compare-scatter-${index}`),
        item.points || [],
        {
          xLabel: "平台评分",
          yLabel: `${item.label}${item.unit ? "（" + item.unit + "）" : ""}`,
          trend: item.calibration ? { slope: item.calibration.slope, intercept: item.calibration.intercept } : null,
        }
      );
    });
  }

  /* ------------------------------------------------------------------ */
  /* 记录与版本                                                           */
  /* ------------------------------------------------------------------ */
  async function loadRecords() {
    try {
      const property = el("record-property").value;
      const page = await get("/experiment/records", {
        limit: 200,
        property_name: property || undefined,
      });
      currentRecords = page.items || [];
      if (!currentRecords.length) {
        el("records-table").innerHTML = App.render.empty(
          "暂无实验记录",
          "下载模板填写后在上方导入，或使用「确认导入」按钮"
        );
        return;
      }
      el("records-table").innerHTML = `
        <table class="data">
          <thead><tr>
            <th>#</th><th>突变</th><th>属性</th><th class="mono">实测值</th>
            <th>单位</th><th>条件</th><th>重复</th><th>操作人</th><th>录入时间</th><th></th>
          </tr></thead>
          <tbody>${currentRecords
            .map(
              (record) => `<tr style="cursor:default">
                <td class="mono text-2">${record.id}</td>
                <td class="mutation">${fmt.escape(record.mutation || "野生型")}</td>
                <td><span class="badge">${fmt.escape(record.property_name)}</span></td>
                <td class="num">${fmt.num(record.measured_value, 4)}</td>
                <td class="text-xs">${fmt.escape(record.unit || "—")}</td>
                <td class="text-xs">${fmt.escape(record.condition || "—")}</td>
                <td class="mono text-xs">${record.replicate ?? "—"}</td>
                <td class="text-xs">${fmt.escape(record.operator || "—")}</td>
                <td class="text-xs text-2">${fmt.datetime(record.created_at)}</td>
                <td><button class="btn btn-sm btn-danger" data-delete="${record.id}">删除</button></td>
              </tr>`
            )
            .join("")}</tbody>
        </table>`;
      el("records-table")
        .querySelectorAll("[data-delete]")
        .forEach((button) =>
          button.addEventListener("click", async () => {
            try {
              await global.API.del(`/experiment/records/${button.dataset.delete}`);
              App.toast("已删除", "ok");
              await Promise.all([loadRecords(), loadProperties()]);
            } catch (err) {
              App.toastError(err);
            }
          })
        );
    } catch (err) {
      App.toastError(err);
    }
  }

  async function loadVersions() {
    try {
      const page = await get("/model/versions", { limit: 100 });
      const items = page.items || [];
      el("version-sub").textContent = `${items.length} 个版本 · ${
        items.filter((item) => item.is_active).length
      } 个已激活`;
      if (!items.length) {
        el("version-table").innerHTML = App.render.empty(
          "暂无模型版本",
          "先导入实验数据，再点击「全量训练」"
        );
        return;
      }
      el("version-table").innerHTML = `
        <table class="data">
          <thead><tr>
            <th>版本</th><th>属性</th><th>算法</th>
            <th class="mono">样本</th><th class="mono">R²</th><th class="mono">Spearman</th>
            <th class="mono">CV R²</th><th>状态</th><th>创建时间</th><th></th>
          </tr></thead>
          <tbody>${items
            .map((item) => {
              const metrics = item.metrics || {};
              return `<tr style="cursor:default">
                <td class="mono text-primary">${fmt.escape(item.version)}</td>
                <td class="text-xs">${fmt.escape(item.property_name)}</td>
                <td class="text-xs">${fmt.escape(item.algo)}</td>
                <td class="num">${fmt.int(item.n_samples)}</td>
                <td class="num">${fmt.num(metrics.r2, 3)}</td>
                <td class="num">${fmt.num(metrics.spearman, 3)}</td>
                <td class="num text-2">${fmt.num(metrics.cv_r2, 3)}</td>
                <td>${
                  item.is_active
                    ? '<span class="badge primary">已激活</span>'
                    : '<span class="badge">历史</span>'
                }</td>
                <td class="text-xs text-2">${fmt.datetime(item.created_at)}</td>
                <td>${
                  item.is_active
                    ? ""
                    : `<button class="btn btn-sm" data-activate="${item.id}">回滚至此</button>`
                }</td>
              </tr>`;
            })
            .join("")}</tbody>
        </table>`;

      el("version-table")
        .querySelectorAll("[data-activate]")
        .forEach((button) =>
          button.addEventListener("click", async () => {
            try {
              await post(`/model/versions/${button.dataset.activate}/activate`);
              App.toast("已切换激活版本", "ok");
              await loadVersions();
            } catch (err) {
              App.toastError(err);
            }
          })
        );
    } catch (err) {
      App.toastError(err);
    }
  }

  async function train(incremental) {
    const propertyName = el("train-property").value;
    if (!propertyName) {
      App.toast("没有可训练的属性（请先导入实验数据）", "warn");
      return;
    }
    el("train-result").innerHTML = App.render.alert("训练作业已提交，正在处理…", "info");

    try {
      const accepted = incremental
        ? await post(`/model/incremental?property_name=${encodeURIComponent(propertyName)}`)
        : await post("/model/train", {
            property_name: propertyName,
            algo: el("train-algo").value,
            min_samples: Number(el("train-min").value) || 8,
            test_ratio: 0.25,
            cv_folds: 5,
          });

      const result = await global.API.pollJob(accepted.job_id, {
        onProgress: (job) => {
          el("train-result").innerHTML = App.render.alert(
            `${fmt.escape(job.stage || "训练中")} · ${Math.round((job.progress || 0) * 100)}%`,
            "info"
          );
        },
      });

      const metrics = result.metrics || {};
      el("train-result").innerHTML = `
        ${App.render.alert(
          `训练完成：版本 <b>${fmt.escape(result.version.version)}</b> 已激活（${
            result.version.n_samples
          } 个样本）`,
          "ok"
        )}
        <div class="grid grid-2" style="gap:10px">
          ${App.render.stat("R²", fmt.num(metrics.r2, 3), { plain: true, small: true })}
          ${App.render.stat("Spearman ρ", fmt.num(metrics.spearman, 3), { plain: true, small: true })}
          ${App.render.stat("MAE", fmt.num(metrics.mae, 3), { plain: true, small: true })}
          ${App.render.stat("CV R²", fmt.num(metrics.cv_r2, 3), { plain: true, small: true })}
        </div>
        ${(metrics.notes || [])
          .map((note) => App.render.alert(fmt.escape(note), "warn"))
          .join("")}
        ${
          result.incremental_equivalent_to_full_refit
            ? App.render.alert("Ridge 增量训练与全量重训在数值上等价。", "info")
            : ""
        }
      `;
      App.toast("训练完成", "ok");
      await Promise.all([loadVersions(), loadProperties()]);
    } catch (err) {
      el("train-result").innerHTML = App.render.alert(
        fmt.escape(err.message || "训练失败"),
        "danger"
      );
      App.toastError(err);
    }
  }

  /* ------------------------------------------------------------------ */
  /* 事件绑定与初始化                                                     */
  /* ------------------------------------------------------------------ */
  function bindEvents() {
    el("btn-choose").addEventListener("click", () => el("file-input").click());
    el("file-input").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) ingestFile(file);
      event.target.value = "";
    });

    const zone = el("drop-zone");
    ["dragenter", "dragover"].forEach((type) =>
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.style.borderColor = "var(--primary)";
        zone.style.background = "rgba(0,224,184,0.05)";
      })
    );
    ["dragleave", "drop"].forEach((type) =>
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.style.borderColor = "var(--glass-border)";
        zone.style.background = "";
      })
    );
    zone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (file) ingestFile(file);
    });

    el("btn-compare").addEventListener("click", runCompare);
    el("btn-refresh-records").addEventListener("click", loadRecords);
    el("record-property").addEventListener("change", loadRecords);
    el("btn-train").addEventListener("click", () => train(false));
    el("btn-incremental").addEventListener("click", () => train(true));

    document.addEventListener("project-changed", async () => {
      await Promise.all([loadSequences(), loadRecords(), loadProperties(), loadVersions()]);
    });
  }

  async function init() {
    bindEvents();
    await Promise.all([
      loadTemplate(),
      loadSequences(),
      loadRecords(),
      loadProperties(),
      loadVersions(),
    ]);
  }

  global.Experiment = { init };
})(window);
