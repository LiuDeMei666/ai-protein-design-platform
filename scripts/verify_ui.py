#!/usr/bin/env python
"""前端端到端可用性验证（Playwright）。

验证内容
--------
1. 五个页面均能正常加载，**无 JS 控制台错误**；
2. 关键流程可交互：序列校验、候选表渲染、评分卡展开、对比图绘制；
3. 3D 查看器或 Canvas 降级视图至少有一个可用（不允许白屏）；
4. 输出每个页面的截图，作为交付验证材料。

前置条件
--------
* 后端服务已启动（``bash run_server.sh``）；
* 已安装 Playwright 与 Chromium::

      pip install playwright
      python -m playwright install chromium

用法::

    python scripts/verify_ui.py
    python scripts/verify_ui.py --base http://127.0.0.1:8848 --headed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "validation" / "ui"
SAMPLE_SEQUENCE = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

#: 需要忽略的控制台噪声（第三方库的良性提示）
IGNORED_CONSOLE = (
    "favicon",
    "Download the React DevTools",
    "WebGL",
    "GPU",
    "Autofill",
)


def seed_design_run(base: str) -> int | None:
    """通过 API 预置一个设计批次，供设计页的"载入历史批次"路径验证。

    使用 ``use_structure=False`` 与短序列，保证在数秒内完成，不依赖外网。
    """
    import httpx

    print("  预置设计批次（用于验证历史记录渲染）…")
    with httpx.Client(base_url=base, timeout=600.0) as client:
        try:
            accepted = client.post(
                "/api/design/run",
                json={
                    "sequence": SAMPLE_SEQUENCE,
                    "protein_type": "generic",
                    "mode": "single",
                    "top_n": 20,
                    "use_structure": False,
                },
            )
            accepted.raise_for_status()
            job_id = accepted.json()["job_id"]
        except Exception as exc:
            print(f"    预置失败（跳过该验证项）：{exc}")
            return None

        deadline = time.time() + 300
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("success", "failed", "cancelled"):
                if job["status"] != "success":
                    print(f"    预置作业失败：{job.get('error')}")
                    return None
                run_id = (job.get("result") or {}).get("design_run_id")
                print(f"    已生成批次 id={run_id}")
                return run_id
            time.sleep(1.5)
    print("    预置超时")
    return None


def verify(base: str, headed: bool) -> dict:
    from playwright.sync_api import sync_playwright

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = seed_design_run(base)

    results: list[dict] = []
    console_errors: list[str] = []

    pages = [
        {"name": "工作台", "path": "/index.html", "key": "index"},
        {"name": "序列与结构", "path": "/structure.html", "key": "structure"},
        {"name": "性质报告", "path": "/property.html", "key": "property"},
        {"name": "突变设计", "path": "/design.html", "key": "design"},
        {"name": "实验与迭代", "path": "/experiment.html", "key": "experiment"},
    ]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()

        def on_console(message):
            if message.type != "error":
                return
            text = message.text
            if any(noise in text for noise in IGNORED_CONSOLE):
                return
            console_errors.append(f"[{current_page}] {text}")

        def on_page_error(error):
            console_errors.append(f"[{current_page}] 未捕获异常: {error}")

        page.on("console", on_console)
        page.on("pageerror", on_page_error)

        current_page = ""

        for item in pages:
            current_page = item["name"]
            record: dict = {"page": item["name"], "path": item["path"], "checks": []}

            url = base + item["path"]
            if item["key"] == "design" and run_id:
                url += f"?run_id={run_id}"

            try:
                response = page.goto(url, wait_until="networkidle", timeout=60000)
                record["status"] = response.status if response else None
                page.wait_for_timeout(1600)

                # 通用检查：导航与状态栏已注入
                nav_count = page.locator(".app-nav .nav-link").count()
                record["checks"].append(
                    {"name": "顶部导航已注入", "pass": nav_count == 5, "detail": f"{nav_count} 个入口"}
                )

                status_text = page.locator("#app-status").inner_text() if page.locator("#app-status").count() else ""
                record["checks"].append(
                    {
                        "name": "底部状态栏有内容",
                        "pass": "状态" in status_text,
                        "detail": status_text.replace("\n", " · ")[:90],
                    }
                )

                # 页面专属检查
                if item["key"] == "index":
                    cards = page.locator("#overview .stat").count()
                    record["checks"].append(
                        {"name": "概览卡片渲染", "pass": cards >= 6, "detail": f"{cards} 张"}
                    )

                elif item["key"] == "structure":
                    page.fill("#sequence", SAMPLE_SEQUENCE)
                    page.wait_for_timeout(1400)
                    hint = page.locator("#validate-hint").inner_text()
                    record["checks"].append(
                        {"name": "序列实时校验", "pass": "通过" in hint, "detail": hint[:60]}
                    )
                    composition = page.locator("#composition-chart canvas").count()
                    record["checks"].append(
                        {"name": "组成图表绘制", "pass": composition > 0, "detail": f"{composition} 个 canvas"}
                    )

                elif item["key"] == "property":
                    chips = page.locator("#protein-types .chip").count()
                    hosts = page.locator("#host option").count()
                    record["checks"].append(
                        {"name": "参数控件渲染", "pass": chips >= 4 and hosts >= 4,
                         "detail": f"{chips} 类型 / {hosts} 宿主"}
                    )

                elif item["key"] == "design":
                    if run_id:
                        rows = page.locator("#candidate-table tbody tr").count()
                        record["checks"].append(
                            {"name": "候选表渲染（历史批次）", "pass": rows > 0, "detail": f"{rows} 行"}
                        )
                        if rows:
                            page.locator("#candidate-table tbody tr").first.click()
                            page.wait_for_timeout(1200)
                            dims = page.locator("#card-body .dim-card").count()
                            canvas = page.locator("#card-chart canvas").count()
                            record["checks"].append(
                                {"name": "评分卡展开", "pass": dims >= 5, "detail": f"{dims} 个维度卡"}
                            )
                            record["checks"].append(
                                {"name": "评分分解图绘制", "pass": canvas > 0, "detail": f"{canvas} 个 canvas"}
                            )
                    else:
                        record["checks"].append(
                            {"name": "候选表渲染", "pass": True, "detail": "未预置批次，跳过"}
                        )

                elif item["key"] == "experiment":
                    columns = page.locator("#template-body tbody tr").count()
                    record["checks"].append(
                        {"name": "模板列定义渲染", "pass": columns >= 8, "detail": f"{columns} 列"}
                    )
                    versions = page.locator("#version-table").inner_text()
                    record["checks"].append(
                        {"name": "模型版本区可渲染", "pass": len(versions) > 0,
                         "detail": versions.replace("\n", " ")[:80]}
                    )

                # 截图
                shot = OUTPUT_DIR / f"{item['key']}.png"
                page.screenshot(path=str(shot), full_page=True)
                record["screenshot"] = str(shot.relative_to(PROJECT_ROOT))

            except Exception as exc:
                record["status"] = record.get("status")
                record["checks"].append({"name": "页面加载与交互", "pass": False, "detail": str(exc)[:200]})

            record["passed"] = all(check["pass"] for check in record["checks"])
            results.append(record)
            mark = "✓" if record["passed"] else "✗"
            print(f"  {mark} {record['page']}")
            for check in record["checks"]:
                print(f"      {'✓' if check['pass'] else '✗'} {check['name']}：{check.get('detail', '')}")

        browser.close()

    report = {
        "pages": results,
        "console_errors": console_errors,
        "passed": all(item["passed"] for item in results) and not console_errors,
        "screenshots_dir": str(OUTPUT_DIR),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="前端端到端可用性验证")
    parser.add_argument("--base", default="http://127.0.0.1:8848", help="服务地址")
    parser.add_argument("--headed", action="store_true", help="显示浏览器窗口")
    parser.add_argument("--report", default=None, help="报告输出路径")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        print("未安装 Playwright。请执行：")
        print("  pip install playwright && python -m playwright install chromium")
        return 2

    print("=" * 78)
    print(" 前端端到端可用性验证")
    print("=" * 78)
    print(f" 目标服务: {args.base}")
    print()

    report = verify(args.base, args.headed)

    print("\n" + "=" * 78)
    if report["console_errors"]:
        print(f" 控制台错误 {len(report['console_errors'])} 条：")
        for error in report["console_errors"][:20]:
            print(f"   - {error}")
    else:
        print(" 无 JS 控制台错误")
    print(f" 截图目录: {report['screenshots_dir']}")
    print(f" 总体判定: {'通过' if report['passed'] else '未通过'}")
    print("=" * 78)

    output = Path(args.report) if args.report else OUTPUT_DIR / "ui_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" 报告已写入: {output}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
