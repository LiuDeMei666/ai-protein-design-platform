/* ==========================================================================
   3D 结构查看器（3Dmol.js 封装 + 无依赖 Canvas 降级）
   --------------------------------------------------------------------------
   核心职责：
   1. 按 pLDDT 着色渲染 PDB（与 AlphaFold 官方配色一致，便于文献对照）；
   2. 把突变热点在结构上高亮，并与候选表联动；
   3. **缺少 3Dmol 时不白屏**：退化为服务端数据驱动的 Canvas 残基轨道图，
      仍然把逐残基置信度与热点位置完整呈现出来。
   ========================================================================== */

(function (global) {
  "use strict";

  const has3Dmol = () => typeof global.$3Dmol !== "undefined";

  /* AlphaFold 官方 pLDDT 配色 */
  const BANDS = [
    { min: 90, color: "#0053d6", label: "极高 ≥90" },
    { min: 70, color: "#65cbf3", label: "可信 70-90" },
    { min: 50, color: "#ffdb13", label: "较低 50-70" },
    { min: 0, color: "#ff7d45", label: "极低 <50" },
  ];

  function bandColor(plddt) {
    for (const band of BANDS) {
      if (plddt >= band.min) return band.color;
    }
    return BANDS[BANDS.length - 1].color;
  }

  class Viewer {
    constructor(container) {
      this.container = container;
      this.viewer = null;
      this.pdbText = "";
      this.style = "cartoon";
      this.spinning = false;
      this.highlight = [];
      this.plddt = [];
      this.fallbackCanvas = null;
    }

    /* ---------------- 渲染入口 ---------------- */
    render(pdbText, options = {}) {
      this.pdbText = pdbText || "";
      this.plddt = options.plddt || [];
      if (!this.pdbText) {
        this.showOverlay("没有可渲染的结构数据");
        return false;
      }

      if (!has3Dmol()) {
        this.renderFallback(options);
        return false;
      }

      this.hideOverlay();
      this.container.innerHTML = '<div class="viewer-3d" style="width:100%;height:100%"></div>';
      const host = this.container.querySelector(".viewer-3d");
      const $3Dmol = global.$3Dmol;

      this.viewer = $3Dmol.createViewer(host, {
        backgroundColor: "#05070d",
        antialias: true,
      });
      this.viewer.addModel(this.pdbText, "pdb");
      this.applyStyle();
      this.viewer.zoomTo();
      this.viewer.render();

      // 入场：自旋两周后停稳，给"结构正在呈现"的仪式感
      this.viewer.spin("y", 0.9);
      setTimeout(() => {
        if (this.viewer && !this.spinning) {
          this.viewer.spin(false);
          this.viewer.render();
        }
      }, 2100);

      if (this.highlight.length) this.setHighlight(this.highlight);
      return true;
    }

    /* ---------------- 样式 ---------------- */
    applyStyle() {
      if (!this.viewer) return;
      const band = (atom) => bandColor(atom.b || 0);

      if (this.style === "surface") {
        this.viewer.setStyle({}, { cartoon: { colorfunc: band, opacity: 0.55 } });
        this.viewer.addSurface(global.$3Dmol.SurfaceType.VDW, {
          opacity: 0.55,
          colorfunc: band,
        });
      } else if (this.style === "stick") {
        this.viewer.removeAllSurfaces();
        this.viewer.setStyle({}, { stick: { radius: 0.14, colorfunc: band } });
      } else if (this.style === "line") {
        this.viewer.removeAllSurfaces();
        this.viewer.setStyle({}, { line: { linewidth: 2.4, colorfunc: band } });
      } else {
        this.viewer.removeAllSurfaces();
        this.viewer.setStyle({}, { cartoon: { colorfunc: band } });
      }
      this.viewer.render();
    }

    setStyle(style) {
      this.style = style;
      this.applyStyle();
    }

    toggleSpin() {
      if (!this.viewer) return false;
      this.spinning = !this.spinning;
      this.viewer.spin(this.spinning ? "y" : false, 0.9);
      this.viewer.render();
      return this.spinning;
    }

    /* ---------------- 突变热点高亮 ---------------- */
    /**
     * @param {Array<{residue:number,label?:string,score?:number}>} items 残基号（1-based）
     */
    setHighlight(items) {
      this.highlight = items || [];
      if (!this.viewer) return;
      const $3Dmol = global.$3Dmol;

      this.viewer.removeAllShapes();
      this.viewer.setStyle({}, { cartoon: { colorfunc: (atom) => bandColor(atom.b || 0) } });

      const pick = (residue) => ({ resi: residue });

      this.viewer.setStyle({}, { cartoon: { colorfunc: (atom) => bandColor(atom.b || 0) } });
      for (const item of this.highlight) {
        const selected = pick(item.residue);
        this.viewer.setStyle(selected, {
          cartoon: { color: "#f43f5e" },
          stick: { radius: 0.22, color: "#f43f5e" },
        });
        // 呼吸式光晕圈，便于在整条链上快速定位
        this.viewer.addSphere({
          center: this._residueCenter(selected),
          radius: 1.5,
          color: "#f43f5e",
          opacity: 0.26,
          wireframe: true,
        });
      }
      this.viewer.render();

      if (this.highlight.length) {
        try {
          this.viewer.zoomTo(
            this.highlight.length === 1
              ? pick(this.highlight[0].residue)
              : this.highlight.map((item) => pick(item.residue))
          );
          this.viewer.render();
        } catch (_) {
          /* 某些 3Dmol 版本对多选择器支持不完整，忽略即可 */
        }
      }
    }

    _residueCenter(selection) {
      try {
        const atoms = this.viewer.selectedAtoms(selection);
        if (!atoms.length) return { x: 0, y: 0, z: 0 };
        const sum = atoms.reduce(
          (acc, atom) => ({ x: acc.x + atom.x, y: acc.y + atom.y, z: acc.z + atom.z }),
          { x: 0, y: 0, z: 0 }
        );
        return { x: sum.x / atoms.length, y: sum.y / atoms.length, z: sum.z / atoms.length };
      } catch (_) {
        return { x: 0, y: 0, z: 0 };
      }
    }

    resetView() {
      if (!this.viewer) return;
      this.viewer.zoomTo();
      this.viewer.render();
    }

    /* ---------------- 降级：Canvas 残基轨道 ---------------- */
    /**
     * 3Dmol 不可用时的兜底：用服务端返回的逐残基 pLDDT 画一条彩色轨道，
     * 并把突变热点标出来。核心信息（置信度分布 + 热点位置）不丢失。
     */
    renderFallback(options = {}) {
      const plddt = this.plddt || [];
      const highlights = options.highlight || [];

      if (!plddt.length) {
        this.showOverlay(
          "3D 渲染不可用（缺少 3Dmol.js），且未获得逐残基置信度数据。<br/>" +
            "请在服务端执行 <code>python scripts/fetch_vendor_assets.py</code> 后刷新。"
        );
        return;
      }

      this.hideOverlay();
      this.container.innerHTML = `
        <div style="padding:14px 16px">
          <div class="alert warn mb-2">
            <span class="ico">!</span>
            <span>3Dmol.js 未加载，已切换为 <b>逐残基置信度轨道图</b>（Canvas 降级视图）。
            交互式三维结构不可用，但置信度分布与突变热点位置完整保留。
            如需 3D 视图，请执行 <code>python scripts/fetch_vendor_assets.py</code>。</span>
          </div>
          <canvas id="fallback-track" style="width:100%;height:190px;display:block"></canvas>
          <div class="track-legend">
            ${BANDS.map(
              (band) =>
                `<span><i style="background:${band.color};width:11px;height:4px;display:inline-block;border-radius:2px"></i>${band.label}</span>`
            ).join("")}
            <span><i style="background:#f43f5e;width:11px;height:4px;display:inline-block;border-radius:2px"></i>突变热点</span>
          </div>
        </div>
      `;

      this.fallbackCanvas = this.container.querySelector("#fallback-track");
      this._drawFallbackTrack(plddt, highlights);
    }

    _drawFallbackTrack(plddt, highlights) {
      const canvas = this.fallbackCanvas;
      if (!canvas) return;
      const ratio = global.devicePixelRatio || 1;
      const width = canvas.clientWidth || 800;
      const height = 190;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      const ctx = canvas.getContext("2d");
      ctx.scale(ratio, ratio);
      ctx.clearRect(0, 0, width, height);

      const padLeft = 44;
      const padRight = 14;
      const padTop = 14;
      const plotWidth = width - padLeft - padRight;
      const plotHeight = height - padTop - 34;
      const step = plotWidth / plddt.length;

      // 网格与刻度
      ctx.strokeStyle = "rgba(122,162,200,0.14)";
      ctx.fillStyle = "rgba(154,169,190,0.75)";
      ctx.font = "10px monospace";
      ctx.lineWidth = 1;
      for (const value of [0, 25, 50, 75, 100]) {
        const y = padTop + plotHeight - (value / 100) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(padLeft + plotWidth, y);
        ctx.stroke();
        ctx.fillText(String(value), 8, y + 3);
      }
      ctx.save();
      ctx.translate(13, padTop + plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("pLDDT", -18, 0);
      ctx.restore();

      // 柱状轨道
      for (let index = 0; index < plddt.length; index += 1) {
        const value = Math.max(0, Math.min(100, plddt[index]));
        const barHeight = (value / 100) * plotHeight;
        ctx.fillStyle = bandColor(value);
        ctx.fillRect(padLeft + index * step, padTop + plotHeight - barHeight, Math.max(1, step - 0.4), barHeight);
      }

      // 突变热点标记
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 1.6;
      for (const item of highlights) {
        const index = (item.residue || 1) - 1;
        if (index < 0 || index >= plddt.length) continue;
        const x = padLeft + index * step + step / 2;
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, padTop + plotHeight);
        ctx.stroke();
      }

      // 横轴：残基号
      ctx.fillStyle = "rgba(154,169,190,0.75)";
      const ticks = Math.min(10, plddt.length);
      for (let t = 0; t <= ticks; t += 1) {
        const index = Math.round((t / ticks) * (plddt.length - 1));
        const x = padLeft + index * step;
        ctx.fillText(String(index + 1), x - 8, height - 12);
      }
    }

    /* ---------------- 覆盖层 ---------------- */
    showOverlay(html) {
      let overlay = this.container.querySelector(".viewer-overlay");
      if (!overlay) {
        overlay = document.createElement("div");
        overlay.className = "viewer-overlay";
        this.container.appendChild(overlay);
      }
      overlay.innerHTML = `<div class="spinner lg"></div><div>${html}</div>`;
      overlay.classList.remove("hidden");
    }

    showLoading(text) {
      this.showOverlay(text || "正在预测三维结构…");
    }

    hideOverlay() {
      const overlay = this.container.querySelector(".viewer-overlay");
      if (overlay) overlay.remove();
    }

    destroy() {
      if (this.viewer) {
        try {
          this.viewer.clear();
        } catch (_) {}
        this.viewer = null;
      }
    }
  }

  global.Viewer3D = { Viewer, has3Dmol, bandColor, BANDS };
})(window);
