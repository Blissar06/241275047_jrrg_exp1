"""把 docs/screenshots/ 下的 PNG 拼成 10 秒演示 GIF。

设计：
  - 选 8 张代表性图，每张展示 ~1 秒（10 帧 × 100ms）
  - 顶部加 Tab 指示条（高亮当前所属 Tab）
  - 帧间加 3 帧淡入淡出（平滑过渡）
  - 输出 docs/screenshots/demo.gif

帧规划：
   00s  [概览] 01_overview_nav
   01s  [概览] 12_compare_radar_multi 第一份
   02s  [行情] 03_market_apy_history
   03s  [行情] 05_market_gas_timeline
   04s  [行情] 06_market_apy_heatmap
   05s  [风险] 07_risk_drawdown
   06s  [仓位] 09_position_timeline
   07s  [成本] 10_cost_stacked
   08s  [对比] 11_compare_nav_overlay
   09s  [对比] 12_compare_radar_multi
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SHOTS = ROOT / "docs" / "screenshots"
OUT = SHOTS / "demo.gif"

# 统一画布尺寸：宽 1280，高 720（720p，PPT 友好）
CANVAS_W, CANVAS_H = 1280, 720

# 顶部 Tab 指示条高度
BANNER_H = 60

# 各 Tab 名称（按 ui/app.py 一致）
TABS = ["概览", "行情", "风险", "仓位", "成本", "多策略对比"]

# 帧规划：(展示秒，所属 Tab 索引，文件名，副标题)
FRAMES = [
    (0,  "概览",         "01_overview_nav.png",          "NAV 实际 vs 理论最优 + 调仓菱形标记"),
    (1,  "概览",         "02_overview_radar_single.png", "单策略归因雷达"),
    (2,  "行情",         "03_market_apy_history.png",    "3 池历史 APY 多线"),
    (3,  "行情",         "05_market_gas_timeline.png",   "Gas spike × 5 一目了然"),
    (4,  "行情",         "06_market_apy_heatmap.png",    "池 × 时间 APY 热力图"),
    (5,  "风险",         "07_risk_drawdown.png",         "真实数据回撤水下图"),
    (6,  "仓位",         "09_position_timeline.png",     "Gantt 持仓时间线"),
    (7,  "成本",         "10_cost_stacked.png",          "摩擦三分量按时间堆积"),
    (8,  "多策略对比",   "11_compare_nav_overlay.png",   "5 个预设 NAV 叠加"),
    (9,  "多策略对比",   "12_compare_radar_multi.png",   "摩擦雷达 + 调仓空窗 bar"),
]


def _try_font(size: int) -> ImageFont.ImageFont:
    """尽量找到中文字体；找不到回退到默认。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",            # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT_TAB = _try_font(20)
FONT_TAB_ACTIVE = _try_font(22)
FONT_TITLE = _try_font(18)
FONT_FOOTER = _try_font(14)


def _draw_banner(canvas: Image.Image, active_tab: str, subtitle: str) -> None:
    """在画布顶部画 Tab 指示条。"""
    draw = ImageDraw.Draw(canvas)

    # 背景
    draw.rectangle([(0, 0), (CANVAS_W, BANNER_H)], fill=(30, 38, 56))

    # Tab 按钮，水平等距
    n = len(TABS)
    btn_w = 140
    total_w = btn_w * n + (n - 1) * 8
    x = (CANVAS_W - total_w) // 2
    y = 8
    for tab in TABS:
        is_active = (tab == active_tab)
        bg = (220, 60, 90) if is_active else (50, 60, 84)
        fg = "white" if is_active else (200, 200, 210)
        draw.rounded_rectangle(
            [(x, y), (x + btn_w, y + 36)],
            radius=8,
            fill=bg,
        )
        # 居中文字
        bbox = draw.textbbox((0, 0), tab, font=FONT_TAB_ACTIVE if is_active else FONT_TAB)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (x + (btn_w - tw) // 2, y + (36 - th) // 2 - 2),
            tab,
            font=FONT_TAB_ACTIVE if is_active else FONT_TAB,
            fill=fg,
        )
        x += btn_w + 8

    # 底部细线
    draw.line([(0, BANNER_H - 1), (CANVAS_W, BANNER_H - 1)], fill=(80, 90, 110), width=1)


def _draw_footer(canvas: Image.Image, subtitle: str, frame_idx: int, total: int) -> None:
    """底部副标题 + 进度条。"""
    draw = ImageDraw.Draw(canvas)

    # 副标题
    bbox = draw.textbbox((0, 0), subtitle, font=FONT_TITLE)
    tw = bbox[2] - bbox[0]
    draw.text(
        ((CANVAS_W - tw) // 2, CANVAS_H - 50),
        subtitle,
        font=FONT_TITLE,
        fill=(60, 60, 60),
    )

    # 进度条
    bar_w = 600
    bar_x = (CANVAS_W - bar_w) // 2
    bar_y = CANVAS_H - 18
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + 4)], fill=(220, 220, 220))
    progress = int(bar_w * (frame_idx + 1) / total)
    draw.rectangle([(bar_x, bar_y), (bar_x + progress, bar_y + 4)], fill=(220, 60, 90))


def _fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """等比缩放图片到不超过 max_w × max_h，背景白色填充。"""
    w, h = img.size
    ratio = min(max_w / w, max_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    return resized


def _compose_frame(image_path: Path, active_tab: str, subtitle: str,
                   frame_idx: int, total: int) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), color=(245, 246, 248))

    # 加载并贴图（留白边）
    img = Image.open(image_path).convert("RGB")
    content_area_h = CANVAS_H - BANNER_H - 70   # 上 banner + 下 footer
    content_area_w = CANVAS_W - 80               # 左右 padding
    fitted = _fit_image(img, content_area_w, content_area_h)
    paste_x = (CANVAS_W - fitted.width) // 2
    paste_y = BANNER_H + (content_area_h - fitted.height) // 2 + 10
    canvas.paste(fitted, (paste_x, paste_y))

    _draw_banner(canvas, active_tab, subtitle)
    _draw_footer(canvas, subtitle, frame_idx, total)
    return canvas


def main():
    print("=== Building demo.gif ===")

    # 每张图展示约 1 秒
    frame_duration_ms = 100         # 100ms / 帧 → 10 帧/秒
    hold_frames = 10                # 每张图保持 10 帧 = 1 秒
    fade_frames = 0                 # 简化：先不做淡入淡出（GIF 文件大）

    frames = []
    total_frames = len(FRAMES) * hold_frames
    counter = 0

    for sec, tab, fname, subtitle in FRAMES:
        path = SHOTS / fname
        if not path.exists():
            print(f"  [skip] {fname} 不存在")
            continue
        print(f"  [{sec:02d}s] {tab}: {fname}")
        frame = _compose_frame(path, tab, subtitle, counter, total_frames)
        # 复用同一帧多次（GIF 压缩相同帧很高效）
        for _ in range(hold_frames):
            frames.append(frame)
            counter += 1

    if not frames:
        print("  [error] 没有可用帧")
        return

    print(f"  共 {len(frames)} 帧 × {frame_duration_ms}ms = {len(frames) * frame_duration_ms / 1000:.1f} 秒")
    print(f"  保存中...")

    # 用 P 模式（256 调色板）减小体积
    frames_p = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in frames]

    frames_p[0].save(
        OUT,
        save_all=True,
        append_images=frames_p[1:],
        optimize=True,
        duration=frame_duration_ms,
        loop=0,    # 0 = 无限循环
        disposal=2,
    )
    size_kb = OUT.stat().st_size / 1024
    print(f"  → {OUT.relative_to(ROOT)}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
