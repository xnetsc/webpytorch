"""The README's chart, as SVG in English and Chinese: the decision model in the browser
against native MLX on the same machine and the same requests.

    python3 scripts/readme_charts.py

The figures are the measurements recorded in PROGRESS.md (M5 MacBook Air, Chrome, WebGPU):
MLX's, taken as the reference while the WebGPU path was optimised (2026-10-07, "Like for like
first"), and the WebGPU path's after it. Change them there first, then here.
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")

# Decision model, 22 layers, F16, milliseconds per request: (MLX native, webtorch low, high).
DECIDE = [
    (("1 question", "back to back"), ("一道题", "连续发"), 7.6, 15.6, 15.6),
    (("1 question", "2 s apart"), ("一道题", "间隔 2 秒"), 35.8, 32.0, 38.0),
    (("3 questions", "back to back"), ("三道题", "连续发"), 19.0, 43.9, 43.9),
    (("3 questions", "2 s apart"), ("三道题", "间隔 2 秒"), 48.2, 64.0, 70.0),
]

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', "
        "'Microsoft YaHei', 'Noto Sans CJK SC', Helvetica, Arial, sans-serif")
INK, MUTED, GRID, CARD, EDGE = "#1f2328", "#59636e", "#e6e9ee", "#ffffff", "#d0d7de"
OURS, MLX = "#2f6fec", "#a3aab3"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=14, fill=INK, anchor="start", weight="normal"):
    return ('<text x="%.1f" y="%.1f" font-size="%d" fill="%s" text-anchor="%s" '
            'font-weight="%s">%s</text>' % (x, y, size, fill, anchor, weight, esc(s)))


def card(w, h, body, title, machine, subtitle):
    """A chart on its own white card: title, the machine it was measured on, what is shown."""
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
            'font-family="%s">\n<rect x="0.5" y="0.5" width="%d" height="%d" rx="12" fill="%s" '
            'stroke="%s"/>\n%s\n%s\n%s\n%s\n</svg>\n'
            % (w, h, w, h, FONT.replace("'", "&apos;"), w - 1, h - 1, CARD, EDGE,
               text(28, 40, title, 19, INK, weight="700"),
               text(28, 66, machine, 14, INK, weight="600"),
               text(28, 88, subtitle, 13, MUTED), "\n".join(body)))


def decide_chart(zh):
    w, h = 860, 488
    left, right, top, bottom = 70, 30, 130, 140
    plot_h = h - top - bottom
    span = w - left - right
    top_v = 80.0
    body = []
    for v in range(0, 81, 20):
        y = top + plot_h * (1 - v / top_v)
        body.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
                    % (left, y, w - right, y, GRID))
        body.append(text(left - 10, y + 4, str(v), 12, MUTED, "end"))
    body.append(text(left - 10, top - 14, "ms", 12, MUTED, "end"))
    group = span / len(DECIDE)
    bar = 46
    for i, (en, cn, mlx, lo, hi) in enumerate(DECIDE):
        cx = left + group * (i + 0.5)
        for j, (value, color) in enumerate(((mlx, MLX), ((lo + hi) / 2, OURS))):
            x = cx - bar - 4 + j * (bar + 8)
            y = top + plot_h * (1 - value / top_v)
            body.append('<rect x="%.1f" y="%.1f" width="%d" height="%.1f" rx="4" fill="%s"/>'
                        % (x, y, bar, top + plot_h - y, color))
            if j == 1 and hi > lo:          # the range measured across sessions
                y_lo = top + plot_h * (1 - lo / top_v)
                y_hi = top + plot_h * (1 - hi / top_v)
                body.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                            'stroke-width="2"/>' % (x + bar / 2, y_lo, x + bar / 2, y_hi, INK))
                for yy in (y_lo, y_hi):
                    body.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                                'stroke-width="2"/>' % (x + bar / 2 - 7, yy, x + bar / 2 + 7, yy,
                                                        INK))
                label = "%g~%g" % (lo, hi)
                top_y = y_hi
            else:
                label = "%.1f" % value
                top_y = y
            body.append(text(x + bar / 2, top_y - 8, label, 13, INK, "middle", "700"))
        a, b = cn if zh else en
        body.append(text(cx, top + plot_h + 24, a, 14, INK, "middle", "600"))
        body.append(text(cx, top + plot_h + 43, b, 13, MUTED, "middle"))
    lx, ly = left, h - 72
    for name, color in (("MLX（原生）" if zh else "MLX (native)", MLX),
                        ("webtorch（浏览器）" if zh else "webtorch (in the browser)", OURS)):
        body.append('<rect x="%d" y="%d" width="14" height="14" rx="3" fill="%s"/>'
                    % (lx, ly - 12, color))
        body.append(text(lx + 22, ly, name, 13, INK))
        lx += 210
    notes = (("MLX 的优势来自苹果 GPU 里的矩阵硬件，WebGPU 目前用不了。",
              "MLX 只能跑在苹果芯片上，webtorch 在英伟达、AMD、Intel 的显卡上也能跑。") if zh else
             ("MLX's lead comes from matrix hardware in Apple's GPU that WebGPU cannot use today.",
              "MLX runs only on Apple silicon; webtorch also runs on NVIDIA, AMD and Intel GPUs."))
    for k, note in enumerate(notes):
        body.append(text(28, h - 38 + 19 * k, note, 12, MUTED))
    title = ("决策模型：webtorch 对比原生 MLX" if zh else
             "Decision model: webtorch against native MLX")
    machine = ("测试机器：MacBook Air（M5，24 GB）" if zh else
               "Test machine: MacBook Air (M5, 24 GB)")
    sub = ("两边同样的请求 · 22 层编码器，F16 · webtorch 跑在 Chrome 里 · 毫秒，越低越好" if zh else
           "The same requests on both · 22-layer encoder, F16 · webtorch in Chrome · "
           "milliseconds, lower is better")
    return card(w, h, body, title, machine, sub)


if __name__ == "__main__":
    for zh in (False, True):
        tag = "zh" if zh else "en"
        path = os.path.join(ROOT, "vs-mlx-%s.svg" % tag)
        with open(path, "w", encoding="utf-8") as f:
            f.write(decide_chart(zh))
        print(path)
