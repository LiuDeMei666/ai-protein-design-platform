/* ==========================================================================
   API 客户端：统一请求封装、错误提示、作业轮询、格式化工具
   --------------------------------------------------------------------------
   约定：
   * 所有请求都走 apiFetch，后端返回的统一错误体 {error, message, detail}
     会被转成带 code 的 ApiError，页面只需 catch 一次。
   * 后端把长耗时任务做成"提交作业 + 轮询"，因此这里提供 submitJob/pollJob，
     页面不再各自实现轮询逻辑。
   ========================================================================== */

(function (global) {
  "use strict";

  const BASE = "/api";
  const TERMINAL = ["success", "failed", "cancelled"];

  /* ---------------- 错误类型 ---------------- */
  class ApiError extends Error {
    constructor(message, { code = "unknown", status = 0, detail = null } = {}) {
      super(message);
      this.name = "ApiError";
      this.code = code;
      this.status = status;
      this.detail = detail;
    }
  }

  class JobError extends ApiError {
    constructor(job) {
      super(job.error || "作业执行失败", {
        code: "job_failed",
        detail: { jobId: job.id, stage: job.stage, status: job.status },
      });
      this.name = "JobError";
      this.job = job;
    }
  }

  /* ---------------- 基础请求 ---------------- */
  async function apiFetch(path, { method = "GET", body, formData, timeoutMs = 600000 } = {}) {
    const url = path.startsWith("http") ? path : BASE + path;
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    if (formData) options.body = formData;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    options.signal = controller.signal;

    let response;
    try {
      response = await fetch(url, options);
    } catch (err) {
      clearTimeout(timer);
      if (err.name === "AbortError") {
        throw new ApiError(`请求超时（${Math.round(timeoutMs / 1000)}s）`, { code: "timeout" });
      }
      throw new ApiError("无法连接后端服务，请确认服务已启动", { code: "network_error" });
    }
    clearTimeout(timer);

    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (_) {
        payload = { message: text.slice(0, 500) };
      }
    }

    if (!response.ok) {
      const message =
        (payload && (payload.message || payload.detail?.message)) ||
        `请求失败（HTTP ${response.status}）`;
      throw new ApiError(message, {
        code: (payload && payload.error) || "http_error",
        status: response.status,
        detail: payload && payload.detail,
      });
    }
    return payload;
  }

  const get = (path, params) => {
    const query = params
      ? "?" +
        new URLSearchParams(
          Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")
        ).toString()
      : "";
    return apiFetch(path + query);
  };
  const post = (path, body, params) => {
    const query = params
      ? "?" +
        new URLSearchParams(
          Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")
        ).toString()
      : "";
    return apiFetch(path + query, { method: "POST", body });
  };
  const upload = (path, formData) => apiFetch(path, { method: "POST", formData });
  const del = (path) => apiFetch(path, { method: "DELETE" });

  /* ---------------- 作业 ---------------- */
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function submitJob(kind, payload, params) {
    const accepted = await post(`/${kind}/${payload ? "run" : "predict"}`, payload, params);
    return accepted.job_id;
  }

  /**
   * 轮询作业直到终态。
   * @param {string} jobId
   * @param {{onProgress?: Function, intervalMs?: number, timeoutMs?: number}} options
   * @returns {Promise<object>} 作业结果
   */
  async function pollJob(jobId, options = {}) {
    const { onProgress, intervalMs = 1500, timeoutMs = 3600000 } = options;
    const startedAt = Date.now();
    let lastProgress = -1;

    for (;;) {
      const job = await get(`/jobs/${jobId}`);
      if (onProgress && job.progress !== lastProgress) {
        lastProgress = job.progress;
        onProgress(job);
      }
      if (TERMINAL.includes(job.status)) {
        if (job.status !== "success") throw new JobError(job);
        return job.result;
      }
      if (Date.now() - startedAt > timeoutMs) {
        throw new ApiError("作业轮询超时，请到工作台查看作业状态", {
          code: "poll_timeout",
          detail: { jobId },
        });
      }
      await sleep(intervalMs);
    }
  }

  /** 提交并轮询（最常用的一步调用）。 */
  async function runJob(path, payload, params, options = {}) {
    const accepted = await post(path, payload, params);
    return pollJob(accepted.job_id, options);
  }

  async function cancelJob(jobId) {
    return post(`/jobs/${jobId}/cancel`);
  }

  /* ---------------- 格式化工具 ---------------- */
  const fmt = {
    /**
     * 统计序列长度（自动忽略 FASTA 头行与所有空白）。
     *
     * 为什么不直接用 `value.replace(/\s/g, "").length`：
     * 粘贴 FASTA 时头行里的字母会被算进去——实测 `>sp|Q39967|ALL5_HEVBR
     * Major latex allergen Hev b 5 OS=Hevea brasiliensis OX=3981` 会把
     * 151 aa 的序列显示成 200 多，与后端清洗后的真实长度对不上。
     * 后端已统一剥离头行，前端计数必须保持一致，否则用户会以为序列被截断了。
     */
    sequenceLength(text) {
      if (!text) return 0;
      return String(text)
        .split(/\r?\n/)
        .filter((line) => !line.trim().startsWith(">"))
        .join("")
        .replace(/\s/g, "").length;
    },
    /** 0-100 分 -> 语义色（越高越好） */
    scoreColor(score) {
      if (score === null || score === undefined) return "var(--text-2)";
      if (score >= 70) return "var(--ok)";
      if (score >= 50) return "var(--cyan)";
      if (score >= 33) return "var(--warn)";
      return "var(--danger)";
    },
    scoreClass(score) {
      if (score === null || score === undefined) return "";
      if (score >= 70) return "";
      if (score >= 50) return "mid";
      return "low";
    },
    riskBadge(risk) {
      const map = {
        low: ['<span class="badge ok">低风险</span>', "ok"],
        medium: ['<span class="badge warn">中风险</span>', "warn"],
        high: ['<span class="badge danger">高风险</span>', "danger"],
      };
      return (map[risk] || ['<span class="badge">未知</span>', ""])[0];
    },
    num(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(value)) return "—";
      return Number(value).toFixed(digits);
    },
    int(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "—";
      return Number(value).toLocaleString("zh-CN");
    },
    percent(value, digits = 1) {
      if (value === null || value === undefined) return "—";
      return (Number(value) * 100).toFixed(digits) + "%";
    },
    datetime(value) {
      if (!value) return "—";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      const pad = (n) => String(n).padStart(2, "0");
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(
        date.getHours()
      )}:${pad(date.getMinutes())}`;
    },
    relativeTime(value) {
      if (!value) return "—";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      const diff = (Date.now() - date.getTime()) / 1000;
      if (diff < 60) return "刚刚";
      if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
      if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
      if (diff < 2592000) return `${Math.floor(diff / 86400)} 天前`;
      return fmt.datetime(value).slice(0, 10);
    },
    duration(seconds) {
      if (seconds === null || seconds === undefined) return "—";
      if (seconds < 60) return `${Number(seconds).toFixed(1)}s`;
      const m = Math.floor(seconds / 60);
      const s = Math.round(seconds % 60);
      return `${m}m ${s}s`;
    },
    escape(value) {
      if (value === null || value === undefined) return "";
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    },
    /** 把序列按固定宽度折行，便于等宽展示 */
    wrapSequence(sequence, width = 60) {
      const lines = [];
      for (let i = 0; i < sequence.length; i += width) {
        lines.push(sequence.slice(i, i + width));
      }
      return lines.join("\n");
    },
  };

  /* ---------------- 常量 ---------------- */
  const CONST = {
    PROTEIN_TYPES: [
      { key: "generic", label: "通用蛋白" },
      { key: "collagen", label: "胶原蛋白" },
      { key: "protease", label: "重组蛋白酶" },
      { key: "protein_a", label: "蛋白 A" },
    ],
    DESIGN_MODES: [
      { key: "single", label: "单点突变", desc: "输出 Top-N 单点候选" },
      { key: "combination", label: "多点组合", desc: "束搜索 + 上位效应精评" },
      { key: "local", label: "局部优化", desc: "限定区段内只做保守替换" },
    ],
    HOST_SYSTEMS: [
      { key: "ecoli", label: "大肠杆菌 (K-12)" },
      { key: "bacillus", label: "枯草芽孢杆菌" },
      { key: "yeast", label: "毕赤酵母" },
      { key: "saccharomyces", label: "酿酒酵母" },
      { key: "cho", label: "CHO 细胞" },
    ],
    JOB_KIND_LABEL: { structure: "结构", property: "性质", design: "设计", train: "训练" },
    JOB_KIND_SHORT: { structure: "3D", property: "PH", design: "MU", train: "ML" },
    SAMPLE_SEQUENCE:
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
  };

  global.API = {
    ApiError,
    JobError,
    get,
    post,
    upload,
    del,
    submitJob,
    pollJob,
    runJob,
    cancelJob,
    fmt,
    CONST,
  };
})(window);
