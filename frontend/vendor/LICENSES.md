# 第三方前端库许可声明

本目录下的静态资源为以下开源库的官方发布构建，版权归各自作者所有。

| 文件 | 库 | 版本 | 许可 | 用途 |
| --- | --- | --- | --- | --- |
| `echarts.min.js` | 雷达图 / 折线图 / 散点图 / 柱状图 / 轨道图 | 5.4.3 | Apache-2.0 | https://echarts.apache.org/ |
| `3Dmol-min.js` | PDB 三维结构渲染与 pLDDT 着色 | 2.4.0 | BSD-3-Clause | https://3dmol.csb.pitt.edu/ |
| `chartjs.min.js` | 轻量备用图表库（ECharts 不可用时降级） | 4.4.1 | MIT | https://www.chartjs.org/ |

## 部署说明

1. 本目录整体拷贝到内网服务器的 `frontend/vendor/` 即可离线使用。
2. 若某个文件缺失，前端会自动降级（例如缺少 3Dmol 时改用服务端解析的
   Canvas 残基轨道图），不会白屏。
3. 重新获取：`python scripts/fetch_vendor_assets.py`
