/* ==========================================================================
   图表封装（ECharts 优先，Chart.js 兜底）
   --------------------------------------------------------------------------
   所有图表共用同一套深空实验室主题，页面只需给出容器与数据。
   若 echarts 未加载（企业内网未拷贝 vendor 目录），会退化为文字提示而不是白屏。
   ========================================================================== */

(function (global) {
  "use strict";

  const COLORS = {
    primary: "#00e0b8",
    cyan: "#22d3ee",
    violet: "#7c5cff",
    ok: "#22c55e",
    warn: "#f59e0b",
    danger: "#f43f5e",
    info: "#38bdf8",
    text: "#9aa9be",
    textDim: "#5c6b82",
    grid: "rgba(122,162,200,0.10)",
    axis: "rgba(122,162,200,0.22)",
  };

  /* AlphaFold 官方 pLDDT 配色（与结构查看器保持一致，便于对照） */
  const PLDDT_BANDS = [
    { min: 90, color: "#0053d6", label: "极高 (≥90)" },
    { min: 70, color: "#65cbf3", label: "可信 (70-90)" },
    { min: 50, color: "#ffdb13", label: "较低 (50-70)" },
    { min: 0, color: "#ff7d45", label: "极低 (<50)" },
  ];

  function plddtColor(value) {
    for (const band of PLDDT_BANDS) {
      if (value >= band.min) return band.color;
    }
    return PLDDT_BANDS[PLDDT_BANDS.length - 1].color;
  }

  const available = () => typeof global.echarts !== "undefined";
  const chartJsAvailable = () => typeof global.Chart !== "undefined";

  const instances = new WeakMap();

  function ensure(dom, message) {
    if (!dom) return null;
    if (available()) {
      let instance = instances.get(dom);
      if (!instance || instance.isDisposed()) {
        instance = global.echarts.init(dom, null, { renderer: "canvas" });
        instances.set(dom, instance);
      }
      return instance;
    }
    if (!dom.dataset.fallbackShown) {
      dom.dataset.fallbackShown = "1";
      dom.innerHTML = `<div class="chart-fallback">${
        message || "图表库未就绪：请执行 python scripts/fetch_vendor_assets.py 获取本地图表库"
      }</div>`;
    }
    return null;
  }

  function baseOption(extra = {}) {
    return Object.assign(
      {
        textStyle: { fontFamily: "inherit", color: COLORS.text },
        animationDuration: 620,
        animationEasing: "cubicOut",
        tooltip: {
          backgroundColor: "rgba(13,20,32,0.95)",
          borderColor: "rgba(0,224,184,0.28)",
          borderWidth: 1,
          textStyle: { color: "#e9f1fa", fontSize: 12 },
          extraCssText: "backdrop-filter: blur(8px); border-radius: 9px;",
        },
      },
      extra
    );
  }

  function resizeAll() {
    if (!available()) return;
    document.querySelectorAll("div").forEach((dom) => {
      const instance = instances.get(dom);
      if (instance && !instance.isDisposed()) instance.resize();
    });
  }

  /* ------------------------------------------------------------------ */
  /* 雷达图：性质综合评估                                                 */
  /* ------------------------------------------------------------------ */
  function radar(dom, items, options = {}) {
    const instance = ensure(dom);
    if (!instance) return null;
    const max = options.max || 100;

    instance.setOption(
      baseOption({
        radar: {
          indicator: items.map((item) => ({ name: item.label, max })),
          radius: options.radius || "66%",
          center: ["50%", "54%"],
          splitNumber: 4,
          axisName: { color: COLORS.text, fontSize: 11.5 },
          axisLine: { lineStyle: { color: COLORS.grid } },
          splitLine: { lineStyle: { color: COLORS.grid } },
          splitArea: {
            areaStyle: {
              color: ["rgba(0,224,184,0.030)", "rgba(34,211,238,0.018)"],
            },
          },
        },
        series: [
          {
            type: "radar",
            symbolSize: 5,
            emphasis: { focus: "series", scale: 1.14 },
            data: [
              {
                value: items.map((item) => item.score),
                name: options.name || "当前蛋白",
                lineStyle: { color: COLORS.primary, width: 2 },
                itemStyle: { color: COLORS.primary },
                areaStyle: {
                  color: {
                    type: "radial",
                    x: 0.5,
                    y: 0.5,
                    r: 0.8,
                    colorStops: [
                      { offset: 0, color: "rgba(0,224,184,0.42)" },
                      { offset: 1, color: "rgba(34,211,238,0.10)" },
                    ],
                  },
                },
                label: {
                  show: true,
                  color: "#e9f1fa",
                  fontSize: 10.5,
                  formatter: (params) => Math.round(params.value),
                },
              },
            ],
          },
        ],
      }),
      true
    );
    return instance;
  }

  /* ------------------------------------------------------------------ */
  /* 折线图：净电荷曲线 / 置信度沿序列分布                                 */
  /* ------------------------------------------------------------------ */
  function line(dom, xValues, series, options = {}) {
    const instance = ensure(dom);
    if (!instance) return null;

    instance.setOption(
      baseOption({
        grid: { left: 48, right: 20, top: 26, bottom: 34, containLabel: false },
        xAxis: {
          type: options.xType || "value",
          name: options.xLabel || "",
          nameTextStyle: { color: COLORS.textDim, fontSize: 10.5 },
          axisLine: { lineStyle: { color: COLORS.axis } },
          axisLabel: { color: COLORS.textDim, fontSize: 10.5 },
          splitLine: { show: false },
        },
        yAxis: {
          type: "value",
          name: options.yLabel || "",
          nameTextStyle: { color: COLORS.textDim, fontSize: 10.5 },
          axisLine: { lineStyle: { color: COLORS.axis } },
          axisLabel: { color: COLORS.textDim, fontSize: 10.5 },
          splitLine: { lineStyle: { color: COLORS.grid } },
        },
        series: series.map((item, index) => ({
          name: item.name,
          type: "line",
          smooth: options.smooth !== false,
          showSymbol: item.showSymbol === true,
          symbolSize: 3,
          lineStyle: { width: item.width || 1.8, color: item.color },
          itemStyle: { color: item.color },
          areaStyle: item.area
            ? {
                color: {
                  type: "linear",
                  x: 0,
                  y: 0,
                  x2: 0,
                  y2: 1,
                  colorStops: [
                    { offset: 0, color: (item.color || COLORS.primary) + "55" },
                    { offset: 1, color: (item.color || COLORS.primary) + "00" },
                  ],
                },
              }
            : undefined,
          data:
            options.xType === "category"
              ? item.data
              : item.data.map((value, position) => [xValues[position], value]),
          markLine: item.markLine,
        })),
      }),
      true
    );
    return instance;
  }

  /* ------------------------------------------------------------------ */
  /* 散点图：预测 vs 实测（含对角参考线）                                  */
  /* ------------------------------------------------------------------ */
  function scatter(dom, points, options = {}) {
    const instance = ensure(dom);
    if (!instance) return null;

    const values = points.map((point) => point.measured_value);
    const lo = Math.min(...values, 0);
    const hi = Math.max(...values, 1);

    instance.setOption(
      baseOption({
        grid: { left: 56, right: 22, top: 26, bottom: 46 },
        tooltip: Object.assign(baseOption().tooltip, {
          formatter: (params) => {
            const point = points[params.dataIndex];
            if (!point) return "";
            return `<b>${point.mutation}</b><br/>评分 ${point.predicted_score}<br/>实测 ${point.measured_value}${
              point.unit ? " " + point.unit : ""
            }<br/>校准后预测 ${point.calibrated_prediction}<br/>残差 ${point.residual}`;
          },
        }),
        xAxis: {
          type: "value",
          name: options.xLabel || "平台评分",
          nameLocation: "middle",
          nameGap: 26,
          nameTextStyle: { color: COLORS.textDim, fontSize: 10.5 },
          axisLine: { lineStyle: { color: COLORS.axis } },
          axisLabel: { color: COLORS.textDim, fontSize: 10.5 },
          splitLine: { lineStyle: { color: COLORS.grid } },
        },
        yAxis: {
          type: "value",
          name: options.yLabel || "实测值",
          nameLocation: "middle",
          nameGap: 40,
          nameTextStyle: { color: COLORS.textDim, fontSize: 10.5 },
          axisLine: { lineStyle: { color: COLORS.axis } },
          axisLabel: { color: COLORS.textDim, fontSize: 10.5 },
          splitLine: { lineStyle: { color: COLORS.grid } },
        },
        series: [
          {
            type: "scatter",
            symbolSize: 12,
            data: points.map((point) => [point.predicted_score, point.measured_value]),
            itemStyle: {
              color: {
                type: "radial",
                x: 0.5,
                y: 0.5,
                r: 0.6,
                colorStops: [
                  { offset: 0, color: COLORS.cyan },
                  { offset: 1, color: COLORS.primary },
                ],
              },
              borderColor: "rgba(0,224,184,0.5)",
              borderWidth: 1.5,
              shadowBlur: 12,
              shadowColor: "rgba(0,224,184,0.6)",
            },
            label: {
              show: points.length <= 24,
              formatter: (params) => points[params.dataIndex].mutation,
              position: "top",
              color: COLORS.text,
              fontSize: 10,
            },
            markLine: options.trend
              ? {
                  silent: true,
                  symbol: "none",
                  lineStyle: { color: COLORS.violet, type: "dashed", width: 1.6 },
                  data: [
                    [
                      { coord: [lo, options.trend.slope * lo + options.trend.intercept] },
                      { coord: [hi, options.trend.slope * hi + options.trend.intercept] },
                    ],
                  ],
                  label: { formatter: "线性校准线", color: COLORS.violet, fontSize: 10 },
                }
              : undefined,
          },
        ],
      }),
      true
    );
    return instance;
  }

  /* ------------------------------------------------------------------ */
  /* 柱状图：组成 / 分布                                                  */
  /* ------------------------------------------------------------------ */
  function bar(dom, categories, values, options = {}) {
    const instance = ensure(dom);
    if (!instance) return null;

    instance.setOption(
      baseOption({
        grid: { left: 50, right: 18, top: 22, bottom: options.rotate ? 62 : 34 },
        tooltip: Object.assign(baseOption().tooltip, {
          trigger: "axis",
          axisPointer: { type: "shadow" },
        }),
        xAxis: {
          type: "category",
          data: categories,
          axisLine: { lineStyle: { color: COLORS.axis } },
          axisLabel: {
            color: COLORS.textDim,
            fontSize: 10.5,
            rotate: options.rotate || 0,
            interval: 0,
          },
          axisTick: { show: false },
        },
        yAxis: {
          type: "value",
          axisLine: { lineStyle: { color: COLORS.axis } },
          axisLabel: { color: COLORS.textDim, fontSize: 10.5 },
          splitLine: { lineStyle: { color: COLORS.grid } },
        },
        series: [
          {
            type: "bar",
            data: values.map((value, index) => ({
              value,
              itemStyle: {
                color: options.colors
                  ? options.colors[index % options.colors.length]
                  : {
                      type: "linear",
                      x: 0,
                      y: 0,
                      x2: 0,
                      y2: 1,
                      colorStops: [
                        { offset: 0, color: COLORS.primary },
                        { offset: 1, color: "rgba(34,211,238,0.14)" },
                      ],
                    },
                borderRadius: [4, 4, 0, 0],
              },
            })),
            barMaxWidth: 34,
            label: options.showLabel
              ? { show: true, position: "top", color: COLORS.text, fontSize: 10.5 }
              : undefined,
          },
        ],
      }),
      true
    );
    return instance;
  }

  /* ------------------------------------------------------------------ */
  /* 评分分解（水平条形，可正可负）                                        */
  /* ------------------------------------------------------------------ */
  function contributions(dom, dimensions) {
    const instance = ensure(dom);
    if (!instance) return null;

    const labels = dimensions.map((item) => item.label);
    const values = dimensions.map((item) => item.contribution);

    instance.setOption(
      baseOption({
        grid: { left: 96, right: 62, top: 14, bottom: 26 },
        tooltip: Object.assign(baseOption().tooltip, {
          formatter: (params) => {
            const item = dimensions[params.dataIndex];
            return `<b>${item.label}</b><br/>维度得分 ${Math.round(item.raw)}<br/>权重 <b>${(
              item.weight * 100
            ).toFixed(1)}%</b><br/>对总分贡献 ${
              item.contribution >= 0 ? "+" : ""
            }${item.contribution.toFixed(2)}<br/><span style="color:${COLORS.textDim}">${
              item.rationale || ""
            }</span>`;
          },
        }),
        xAxis: {
          type: "value",
          name: "对总分的贡献",
          nameTextStyle: { color: COLORS.textDim, fontSize: 10.5 },
          axisLine: { lineStyle: { color: COLORS.axis } },
          axisLabel: { color: COLORS.textDim, fontSize: 10.5 },
          splitLine: { lineStyle: { color: COLORS.grid } },
        },
        yAxis: {
          type: "category",
          data: labels,
          inverse: true,
          axisLine: { lineStyle: { color: COLORS.axis } },
          axisLabel: { color: COLORS.text, fontSize: 11.5 },
          axisTick: { show: false },
        },
        series: [
          {
            type: "bar",
            data: values.map((value) => ({
              value,
              itemStyle: {
                color:
                  value >= 0
                    ? {
                        type: "linear",
                        x: 0,
                        y: 0,
                        x2: 1,
                        y2: 0,
                        colorStops: [
                          { offset: 0, color: "rgba(0,224,184,0.3)" },
                          { offset: 1, color: COLORS.primary },
                        ],
                      }
                    : {
                        type: "linear",
                        x: 0,
                        y: 0,
                        x2: 1,
                        y2: 0,
                        colorStops: [
                          { offset: 0, color: COLORS.danger },
                          { offset: 1, color: "rgba(244,63,94,0.3)" },
                        ],
                      },
                borderRadius: 3,
              },
            })),
            barMaxWidth: 16,
            label: {
              show: true,
              position: "right",
              color: COLORS.text,
              fontSize: 11,
              formatter: (params) =>
                (params.value >= 0 ? "+" : "") + Number(params.value).toFixed(2),
            },
          },
        ],
      }),
      true
    );
    return instance;
  }

  /* ------------------------------------------------------------------ */
  /* 环形图：平均 pLDDT                                                   */
  /* ------------------------------------------------------------------ */
  function gauge(dom, value, options = {}) {
    const instance = ensure(dom);
    if (!instance) return null;
    const color = plddtColor(value);

    instance.setOption(
      baseOption({
        series: [
          {
            type: "gauge",
            startAngle: 220,
            endAngle: -40,
            min: 0,
            max: 100,
            radius: "88%",
            center: ["50%", "58%"],
            progress: { show: true, width: 13, roundCap: true, itemStyle: { color } },
            axisLine: {
              lineStyle: { width: 13, color: [[1, "rgba(122,162,200,0.14)"]] },
              roundCap: true,
            },
            pointer: { show: false },
            axisTick: { show: false },
            splitLine: { show: false },
            axisLabel: { show: false },
            anchor: { show: false },
            title: {
              show: true,
              offsetCenter: [0, "34%"],
              color: COLORS.textDim,
              fontSize: 11,
            },
            detail: {
              valueAnimation: true,
              offsetCenter: [0, "-2%"],
              color: "#e9f1fa",
              fontSize: 27,
              fontWeight: 600,
              fontFamily: "monospace",
              formatter: (val) => val.toFixed(1),
            },
            data: [{ value, name: options.label || "平均 pLDDT" }],
          },
        ],
      }),
      true
    );
    return instance;
  }

  /* ------------------------------------------------------------------ */
  /* 语义色辅助                                                           */
  /* ------------------------------------------------------------------ */
  function riskCounts(dom, counts) {
    const instance = ensure(dom);
    if (!instance) return null;
    const data = [
      { name: "低风险", value: counts.low || 0, itemStyle: { color: COLORS.ok } },
      { name: "中风险", value: counts.medium || 0, itemStyle: { color: COLORS.warn } },
      { name: "高风险", value: counts.high || 0, itemStyle: { color: COLORS.danger } },
    ].filter((item) => item.value > 0);

    instance.setOption(
      baseOption({
        tooltip: Object.assign(baseOption().tooltip, { trigger: "item" }),
        legend: {
          bottom: 0,
          textStyle: { color: COLORS.text, fontSize: 11 },
          itemWidth: 9,
          itemHeight: 9,
        },
        series: [
          {
            type: "pie",
            radius: ["48%", "72%"],
            center: ["50%", "44%"],
            avoidLabelOverlap: true,
            itemStyle: { borderColor: "rgba(6,9,17,0.9)", borderWidth: 2 },
            label: {
              show: true,
              color: COLORS.text,
              fontSize: 11,
              formatter: "{b}\n{c}",
            },
            data,
          },
        ],
      }),
      true
    );
    return instance;
  }

  global.Charts = {
    COLORS,
    PLDDT_BANDS,
    plddtColor,
    available,
    chartJsAvailable,
    radar,
    line,
    scatter,
    bar,
    contributions,
    gauge,
    riskCounts,
    resizeAll,
  };
})(window);
