"""Generate charts for hallucination detection report."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Chinese font setup ──────────────────────────────────────
_CHINESE_FONTS = [
    "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Noto Sans SC", "Noto Sans CJK",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
]
_font_family = None

for _name in _CHINESE_FONTS:
    for _f in fm.fontManager.ttflist:
        if _name.lower() in _f.name.lower():
            _font_family = _f.name
            break
    if _font_family:
        break

if _font_family:
    plt.rcParams["font.family"] = _font_family
    plt.rcParams["font.sans-serif"] = [_font_family]
else:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False

# ── Color palette ───────────────────────────────────────────
COLORS = ["#EF5350", "#FF7043", "#FFA726", "#FFCA28",
          "#66BB6A", "#42A5F5", "#AB47BC", "#8D6E63", "#78909C"]


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def type_pie_chart(type_counts: dict, output_path: str, title: str = "幻觉类型分布"):
    """Generate pie chart of hallucination type distribution."""
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    colors = COLORS[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.0f%%",
        startangle=140, colors=colors, pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(10)

    ax.legend(wedges, [f"{l} ({s})" for l, s in zip(labels, sizes)],
              title="幻觉类型", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def type_bar_chart(type_counts: dict, output_path: str, title: str = "幻觉类型统计"):
    """Generate horizontal bar chart of hallucination type counts."""
    items = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    colors = COLORS[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("数量", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def severity_bar_chart(severity_counts: dict, output_path: str, title: str = "严重度分布"):
    """Generate bar chart of severity distribution."""
    order = ["Critical", "High", "Medium", "Low", "None"]
    labels = [s for s in order if s in severity_counts and severity_counts[s] > 0]
    values = [severity_counts[s] for s in labels]

    sev_colors = {"Critical": "#EF5350", "High": "#FF7043",
                  "Medium": "#FFA726", "Low": "#66BB6A", "None": "#78909C"}
    colors = [sev_colors.get(l, "#78909C") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white", width=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("数量", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def nli_bar_chart(nli_counts: dict, output_path: str, title: str = "NLI 判定分布"):
    """Generate bar chart of NLI status distribution."""
    order = ["ENTAILED", "CONTRADICTED", "UNMENTIONED"]
    labels = [s for s in order if s in nli_counts]
    values = [nli_counts.get(s, 0) for s in labels]

    nli_colors = {"ENTAILED": "#66BB6A", "CONTRADICTED": "#EF5350", "UNMENTIONED": "#FFA726"}

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=[nli_colors[l] for l in labels],
                  edgecolor="white", width=0.45)
    ax.set_ylabel("Claim 数量", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_charts(type_counts: dict, severity_counts: dict,
                        nli_counts: dict, output_dir: str = "outputs") -> dict:
    """Generate all charts and return paths relative to working dir.

    Args:
        type_counts: {type_name: count} dict for hallucination types
        severity_counts: {severity: count} dict
        nli_counts: {status: count} dict
        output_dir: directory to save charts

    Returns:
        dict mapping chart_key → relative_path
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    if type_counts:
        pie_path = os.path.join(output_dir, "type_pie.png")
        type_pie_chart(type_counts, pie_path)
        paths["type_pie"] = pie_path

        bar_path = os.path.join(output_dir, "type_bar.png")
        type_bar_chart(type_counts, bar_path)
        paths["type_bar"] = bar_path

    if severity_counts:
        sev_path = os.path.join(output_dir, "severity_bar.png")
        severity_bar_chart(severity_counts, sev_path)
        paths["severity_bar"] = sev_path

    if nli_counts:
        nli_path = os.path.join(output_dir, "nli_bar.png")
        nli_bar_chart(nli_counts, nli_path)
        paths["nli_bar"] = nli_path

    return paths
