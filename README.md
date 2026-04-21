# MatchAnything

轻量化整理后的本地运行版，默认聚焦两种官方推理模型：
`matchanything_eloftr` 和 `matchanything_roma`。

研究性 pipeline、评测脚本、额外 matcher/extractor 与 C++ API 测试已迁到 `legacy/`；默认运行面只保留本地安装、资源准备、UI、API 和基础 CLI。

## 1. 创建环境

```bash
conda env create -f environment.yml
conda activate matchanything
pip install -e .
```

## 2. 准备资源

先检查运行环境与模型状态：

```bash
python -m matchanything doctor
python -m matchanything setup --check
```

如果模型权重尚未下载，执行：

```bash
python -m matchanything setup
```

默认缓存目录：

- `MATCHANYTHING_HOME=~/.cache/matchanything`
- `MATCHANYTHING_MODELS_DIR=$MATCHANYTHING_HOME/models`

也可以在启动时显式传入：

```bash
python -m matchanything setup --models-dir /path/to/models
python -m matchanything ui --models-dir /path/to/models
python -m matchanything api --models-dir /path/to/models
```

## 3. 启动方式

启动 Gradio UI：

```bash
python -m matchanything ui
```

启动 FastAPI：

```bash
python -m matchanything api --host 0.0.0.0 --port 8001
```

本地 CLI 匹配一对图片：

```bash
python -m matchanything match tests/data/02928139_3448003521.jpg tests/data/17295357_9106075285.jpg --output /tmp/match.json
```

兼容入口仍然保留：

- `app.py`
- `import imcui`

它们现在只转发到新的 `matchanything` 运行时，不再在 import 时自动下载或启动服务。
