# AI 辅助蛋白设计平台

面向重组蛋白（工业酶、胶原蛋白、蛋白药物）研发场景的**计算机辅助蛋白分子智能设计平台**。

研发人员在浏览器中提交目标蛋白序列，即可获得三维结构、关键理化性质预测、
以及经过多维打分排序的突变改造方案；实验完成后将实测数据回录平台，
形成"**计算设计 → 实验验证 → 数据回流 → 模型进化**"的闭环。

---

## 快速开始

```bash
# 1) 创建环境（若已存在可跳过）
conda create -n dbz python=3.11 -y
conda activate dbz
export PYTHONNOUSERSITE=1        # 必须：避免被用户级 pip 劫持

# 2) 安装依赖
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 获取模型权重（必须走镜像，HF 直连不可用）
python scripts/download_esm2.py --fallback

# 4) 获取前端第三方库（多镜像回退）
python scripts/fetch_vendor_assets.py

# 5) 获取参考数据（标准序列 + 宿主密码子表）
python scripts/fetch_seed_sequences.py
python scripts/fetch_codon_usage.py

# 6) 启动
bash run_server.sh
```

打开 `http://<服务器IP>:8848` ｜ 接口文档 `http://<服务器IP>:8848/docs`

---

## 参与开发（团队协作）

仓库托管在 GitHub。克隆下来只有约 3 MB——**模型权重不在仓库里**（约 5.2 GB），
按上面「快速开始」第 3 步用脚本拉取。

### 什么进仓库，什么不进

这是本项目最容易踩的坑，**不要绕过 `.gitignore` 手工 `git add` 大文件**：

| ✅ 进仓库 | 原因 |
|---|---|
| `backend/` `frontend/static/` `scripts/` `tests/` `configs/` | 源码 |
| `frontend/vendor/` | 第三方 JS 库，**离线环境没法现拉** |
| `data/seeds/` | 参考序列与密码子表，离线必需；且换了会影响预测结果，需要版本化 |
| `data/validation/case_*.json` | 标准案例的验收基准快照，用来判断"这次改动有没有改变结果" |

| ❌ 不进仓库 | 原因 |
|---|---|
| `models/`（约 5.2 GB） | 可再生；且 GitHub 单文件限 100 MB，**一旦进过历史就永久膨胀** |
| `data/db/` | SQLite 业务库：二进制无法合并，且承载企业真实录入数据 |
| `data/cache/` `data/uploads/` `logs/*` | 运行时产物 |
| `.env` | 机器相关配置与可选密钥（模板见 `.env.example`） |

CI 的 `repo-guard` 作业会**自动拦住**超过 5 MB 的文件与上述目录，不必自己盯。

### 分支与提交

```
main            ← 随时可交付、随时能起服务；只通过 PR 合入
 └─ feat/xxx    ← 特性分支，短生命周期
 └─ fix/xxx
```

* `main` 必须始终能跑起来，合并前 CI 必须绿。
* 提交信息写**为什么**而不是**改了什么**。例如「修复 `.track` 类名冲突导致开关滑块被撑满」
  优于「改 css」。
* PR 模板（`.github/PULL_REQUEST_TEMPLATE.md`）会在开 PR 时自动带出，里面是本项目的自查清单。

### 分工建议

模块边界清晰，多数工作可以并行：

| 可独立并行 | 需要先协调（改前说一声） |
|---|---|
| `services/property/` 理化性质 | `backend/app/schemas/` 前后端契约 |
| `services/structure/` 结构预测 | `configs/default.yaml` 全局配置 |
| `services/design/` 突变设计 | `services/property/metric.py` 的指标标签表 |
| 单个前端页面（一个 html + 对应 js） | `frontend/static/css/app.css` 全局样式 |

前端是**零构建、无模块化框架**的：`app.js` 与 `app.css` 为所有页面共用，没有隔离机制。
因此约定 **一个 PR 只动一个页面**；改全局样式单独开 PR，并在描述里说明影响面。

### CI 会做什么、不做什么

`.github/workflows/ci.yml` 共三个作业，约一两分钟：

1. **仓库边界守卫** —— 超 5 MB 文件、运行时目录混入，直接失败。
2. **语法检查** —— Python 语法、Shell 脚本语法（并检测 CRLF）、YAML 可解析。
3. **测试** —— `pytest -m "not slow" -rs`。

**刻意不安装 torch / transformers / safetensors / peft（合计约 2.5 GB）**，所以
**依赖 ESM-2 权重的用例在 CI 上不会真正执行**，需要在部署机上补齐：

```bash
python -m pytest -m slow     # 只跑需要 GPU 与权重的用例
python -m pytest -rs         # 全跑；缺权重时那批会 skip，-rs 打印跳过原因
```

> `slow` 标记由 `tests/conftest.py` 按"是否依赖 `esm_available` 夹具"**自动**打上，
> 新增用例时无需手工标记，因此 `-m "not slow"` 永远准确等价于"不依赖模型的那批"。

---

## 核心能力

| 模块 | 能力 |
| --- | --- |
| **序列与结构** | FASTA 校验清洗、ESM Atlas 三维结构预测、pLDDT 着色 3D 视图、长序列自动分片（显式标注） |
| **性质预测** | 9 项 0-100 评分：热/碱/酸稳定性、溶解性、聚集风险、表达量趋势、修饰位点、免疫原性、蛋白酶抗性 |
| **突变设计** | 全位点 × 19 氨基酸零样本扫描；单点/组合/局部优化三种模式；**五维可解释评分卡** |
| **实验回流** | CSV/Excel 批量导入（列名自动映射 + 逐行校验）、预测-实测对比、属性头全量/增量训练与版本回滚 |

### 三类专用规则包

| 蛋白类型 | 规则包行为 |
| --- | --- |
| **胶原蛋白** | 保护三股螺旋 Gly 位；优先推荐 Y 位羟脯氨酸与交联位点、X/Y 位链间离子作用 |
| **重组蛋白酶** | 按序列模体（非硬编码编号）保护催化三联体与氧负离子洞；保护前导肽区；按嗜热偏好推荐 |
| **蛋白 A** | 检测串联 Ig 结合域；保护跨重复保守位点；优先消除 Asn 脱酰胺位点以提升耐碱性 |

---

## 设计原则

1. **不静默失败**：结构降级、分片、缓存命中、位点排除……所有状态都显式标注在接口与界面上。
2. **可解释性优先**：每条候选携带逐维度分数、中文依据与告警标签，**禁止只给一个总分**；
   各维度贡献之和**精确等于**总分（零残差分解）。
3. **诚实的边界声明**：不把 0-100 评分伪装成 ΔΔG；不把筛查级免疫原性启发式说成绝对亲和力预测；
   仅在蛋白序列时明确说明无法计算真正的 CAI。
4. **数据来自真实来源**：密码子表取自 Kazusa 官方数据库并记录参考集规模；
   标准序列取自 UniProt；**绝不内置人工编造的序列或频率**。
5. **离线可用**：第三方库全部 vendor 本地；无 GPU、无结构预测时仍能给出完整 9 项性质指标。

---

## 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | Python 3.11 + FastAPI + Pydantic v2 + SQLAlchemy 2.0 + SQLite(WAL) |
| 算法 | PyTorch 2.5.1 + transformers（ESM-2 650M/35M）+ scikit-learn + numpy/scipy + biopython |
| 结构预测 | ESM Atlas 在线 REST（主）/ 本地 ESMFold（后备）/ stub 占位（离线兜底） |
| 前端 | **零构建** 原生 HTML/CSS/ES Module + 本地 vendor 的 ECharts 与 3Dmol.js |
| 测试 | pytest + Playwright（端到端 UI 验证） |

> **前端为什么零构建**：本机 Node 版本为 v12.22.9，无法运行 Vite/现代打包链，
> 因此采用原生多页应用 + 本地 vendor 静态库。这对企业内网离线部署反而更有利。

---

## 目录结构

```
danbaizhi/
├── run_server.sh                # 启动脚本
├── requirements.txt             # 依赖清单（版本已锁定）
├── .env.example                 # 环境变量样例
├── .gitignore / .gitattributes  # 仓库边界与换行符规范
├── .github/                     # CI（3 个作业）与 PR 模板
├── configs/default.yaml         # 平台参数（评分权重、规则包、阈值）
├── backend/app/
│   ├── core/                    # 配置 / 日志 / 异常
│   ├── db/                      # ORM 模型与幂等初始化（含轻量 schema 迁移）
│   ├── schemas/                 # Pydantic 请求响应模型
│   ├── services/
│   │   ├── sequence/            # 序列校验、特征工具、残基属性表
│   │   ├── structure/           # Provider 抽象、域切分、PDB 解析、结构分析
│   │   ├── embedding/           # ESM-2 嵌入与掩码边缘打分
│   │   ├── property/            # 9 项性质预测引擎
│   │   ├── design/              # 突变设计（扫描/打分/规则包/组合/可解释/人工指定）
│   │   └── experiment/          # 数据导入与预测-实测对比
│   ├── ml/                      # 属性头、训练编排、模型版本注册
│   ├── jobs/                    # 进程内异步作业队列与执行器
│   └── api/routes/              # REST 路由（55 个端点）
├── frontend/                    # 零构建前端（5 个页面）
│   ├── vendor/                  # 本地化第三方库（离线可用，进仓库）
│   └── static/                  # 设计系统与页面逻辑
├── data/
│   ├── seeds/                   # 标准序列 + Kazusa 密码子表
│   ├── templates/               # 录入模板与示例
│   ├── validation/              # 验收基准快照（case_*.json 入库；ui/ 截图不入库）
│   ├── cache/                   # ✗ 不入库：结构 / 嵌入 / PDB 缓存
│   └── db/platform.db           # ✗ 不入库：SQLite 数据库
├── models/                      # ✗ 不入库：ESM-2 权重（约 5.2 GB，用脚本拉取）
├── scripts/                     # 数据获取、案例运行、UI 验证
├── docs/                        # 交付文档
├── logs/                        # ✗ 不入库：仅留 .gitkeep 固定目录
└── tests/                       # 单元与接口测试
```

---

## 常用脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/download_esm2.py` | 下载 ESM-2 权重（hf-mirror 镜像，支持 `--list` 检查） |
| `scripts/fetch_vendor_assets.py` | 多镜像回退获取前端库（支持 `--verify` 校验） |
| `scripts/fetch_seed_sequences.py` | UniProt 拉取胶原/工业酶/蛋白A 标准序列 |
| `scripts/fetch_codon_usage.py` | Kazusa 拉取 5 个宿主的密码子使用表 |
| `scripts/seed_demo_data.py` | 初始化演示数据（仅真实参考序列，**不含合成实验数据**） |
| `scripts/run_case_collagen.py` | 胶原蛋白标准化测试案例 |
| `scripts/run_case_industrial_enzyme.py` | 工业酶标准化测试案例 |
| `scripts/run_case_protein_a.py` | 蛋白 A 标准化测试案例 |
| `scripts/verify_ui.py` | Playwright 端到端 UI 验证（含截图与控制台错误检查） |
| `scripts/make_templates.py` | 生成 Excel 交付物（录入模板 + 测试案例矩阵） |
| `scripts/make_docs.py` | Markdown 交付文档转 Word |

---

## 交付文档

每份文档同时提供 Markdown 源文件与排版规范的 Word 版本（`docs/*.docx`）。

| 文档 | 内容 |
| --- | --- |
| [部署文档](docs/部署文档.md) | 环境要求、安装步骤、**离线部署流程**、配置说明、常见问题排查 |
| [平台使用手册](docs/平台使用手册.md) | 面向实验人员的操作指引、结果解读要点、术语速查 |
| [算法原理说明](docs/算法原理说明.md) | 每项输出的算法来源、计算方式与**适用边界** |
| [验证报告](docs/验证报告_胶原蛋白与工业酶.md) | 标准化测试案例的定量结果与一致性校验 |

## 交付表格

| 文件 | 用途 |
| --- | --- |
| `data/templates/实验数据录入模板.xlsx` | 发放给企业生物技术团队的实测数据录入表（含填写说明、属性名对照、列定义） |
| `data/validation/标准化测试案例矩阵.xlsx` | 标准化测试案例的定量结果矩阵（由案例脚本产出，不手工填写） |
| `data/validation/ui/*.png` | 五个页面的端到端验证截图 |
| `data/validation/case_*.json` | 三个测试案例的完整结构化结果 |

---

## 验证状态

```bash
# 后端接口端到端
python scripts/verify_ui.py

# 标准化测试案例
python scripts/run_case_industrial_enzyme.py
python scripts/run_case_protein_a.py
python scripts/run_case_collagen.py

# 单元与接口测试
pytest tests/ -v
```

验证结果详见《验证报告》。

---

## 许可与数据来源

| 资源 | 许可 / 来源 |
| --- | --- |
| ESM-2 模型权重 | MIT（Meta AI） |
| ESM Atlas 预测服务 | 输出结构为 CC-BY-4.0，**学术与商业用途均可** |
| ECharts / 3Dmol.js / Chart.js | Apache-2.0 / BSD-3-Clause / MIT（见 `frontend/vendor/LICENSES.md`） |
| 密码子使用表 | Kazusa Codon Usage Database |
| 标准蛋白序列 | UniProt |

---

## 已知局限

* 平台评分是 **0-100 的相对排序指标**，不是 ΔΔG（kcal/mol）等绝对物理量；
* ESM-2 的 ΔlogP 与实验 ΔΔG 单调相关，但**尚未做 kcal/mol 校准**
  （企业数据积累后可基于属性头完成）；
* 超长序列分片预测时，**片段之间的相对空间取向未经建模**；
* 免疫原性评估是**筛查级启发式**，不是 NetMHCIIpan 的替代品；
* 仅有蛋白序列时**无法计算真正的 CAI**（需要 DNA 密码子），平台给出可严格推导的代理特征。

这些局限在接口响应、界面与文档中都有显式标注，不做隐藏。
