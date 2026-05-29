# RW_Tool

置顶、可缩放的屏幕区域 OCR 工具。OCR 结果模糊匹配 `desc.json` 图鉴，并在独立图标条显示小动物与置信度。

## 环境

- Python 3.10+
- PyQt6（见 `requirements.txt`，Windows 固定 6.6.x）
- `rapidocr-onnxruntime`、`mss`、`rapidfuzz`

Windows 若出现 `DLL load failed while importing QtCore`：启动时会通过 `rw_tool/qt6_bootstrap.py` 预置 DLL 路径；仍失败请重装 PyQt6 或安装 VC++ 运行库。

可选 OCR 后端：在 `config.ini` 的 `[engine] backend = easyocr`（需额外 `pip install easyocr`）。

## 使用

```bash
pip install -r requirements.txt
python main.py
```

### 交互

- **默认锁定**：主框与图标条鼠标穿透，不影响游戏操作
- **Caps Lock 开启**：临时解锁，可拖动/缩放；主框右上角出现关闭按钮
- **Esc**：关闭（窗口有焦点时）；**Ctrl+Q**：退出
- 几何与位置写入 `window_state.json`，下次启动恢复

### 图鉴与图标

- 小动物描述：`desc.json`（`pets[].name` / `description`）
- 图标：`img/{名称}.png`

## 配置（config.ini）

| 段 | 常用项 |
|----|--------|
| `[ocr]` | `frequency_hz`、`ocr_layout`（strip/auto）、`preprocess_scale`、`dual_preprocess` |
| `[matcher]` | `catalog_path`、`min_score`、`min_candidate_score`、`match_top_k` |
| `[icon_panel]` | `enabled`、`show_score`（是否在图标下显示匹配概率） |
| `[window]` | 识别区尺寸、缩放热区；`match_panel_height` / `result_panel_height` 仅 icon 条关闭时有效 |

`icon_panel.enabled=true` 时，主框不再创建匹配/OCR 文本控件；`match_panel_height`、`show_ocr_raw`、`min_margin`（在 `match_top_k=1` 时）等可忽略。

## 项目结构

```
config.ini
desc.json
main.py
rw_tool/
  config.py
  frame_lock.py      # Caps Lock 锁定 / 穿透
  icon_panel_window.py
  ocr_engine.py
  overlay_window.py
  pet_catalog.py
  pet_matcher.py
scripts/test_match.py
```
