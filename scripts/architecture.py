"""The R6 architecture as data: one node/edge list that renders both the README Mermaid diagram and slide 4.

Every edge mirrors the code path: app.py (validation, slicing) -> models/recognizer.py (per-box models, fold,
empty-box filter) -> models/ladder.py (raw read, pools, the four correction rows).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple


class Node(NamedTuple):
    key: str
    label: str
    x: float
    y: float
    primary: bool = False
    w: float = 0.10


class Edge(NamedTuple):
    src: str
    dst: str
    label: str = ""


NODES: tuple[Node, ...] = (
    Node("UI", "Browser UI\ncanvas boxes or\nstrip upload", 0.065, 0.495, w=0.09),
    Node("API", "FastAPI on 127.0.0.1\nHost allow-list, JSON only,\nsize / base64 / image checks", 0.185, 0.495, w=0.125),
    Node("SLICE", "slice_strip_image\n(strip uploads)", 0.315, 0.735),
    Node("PRE", "preprocess_box\n28x28 per box;\nempty boxes left out", 0.315, 0.495),
    Node("MLP", "Dense MLP · FP32\n(display only)", 0.44, 0.735),
    Node("CNN32", "CharCNN · FP32\n+ Grad-CAM", 0.44, 0.495),
    Node("CNN8", "CharCNN · INT8\nstatic PTQ", 0.44, 0.235),
    Node("FOLD32", "47 -> 36 fold\nFP32 lattice", 0.555, 0.495, w=0.09),
    Node("FOLD8", "47 -> 36 fold\nINT8 lattice", 0.555, 0.235, w=0.09),
    Node("R1", "Row 1: raw FP32 read", 0.68, 0.735, w=0.115),
    Node("POOL32", "Candidate pool (FP32 read)\nSymSpell vs 22k lexicon,\ntop 20 by vis - 2.0 x wED", 0.68, 0.495, w=0.125),
    Node("POOL8", "Candidate pool (INT8 read)\nsame fixed rule", 0.68, 0.235, w=0.125),
    Node("NG", "Row 2: interpolated\nn-gram fusion", 0.81, 0.61, w=0.105),
    Node("B32", "Row 3: BERT · FP32\nspan-PLL fusion", 0.81, 0.405, w=0.105),
    Node("B8", "Row 4: BERT · INT8 dynamic\nspan-PLL fusion\nPRIMARY app output (R5)", 0.81, 0.235, primary=True, w=0.115),
    Node("OUT", "Numeric guard (rows 2-4)\n-> JSON -> UI:\n4-model table,\nGrad-CAM, top-2 badges", 0.935, 0.495, w=0.11),
)

EDGES: tuple[Edge, ...] = (
    Edge("UI", "API"), Edge("API", "SLICE", "strip"), Edge("API", "PRE", "boxes"), Edge("SLICE", "PRE"),
    Edge("PRE", "MLP"), Edge("PRE", "CNN32"), Edge("PRE", "CNN8"),
    Edge("CNN32", "FOLD32"), Edge("CNN8", "FOLD8"),
    Edge("FOLD32", "R1"), Edge("FOLD32", "POOL32"), Edge("FOLD8", "POOL8"),
    Edge("POOL32", "NG"), Edge("POOL32", "B32"), Edge("POOL8", "B8"),
    Edge("MLP", "OUT", "top-2 badges"), Edge("R1", "OUT"), Edge("NG", "OUT"), Edge("B32", "OUT"), Edge("B8", "OUT"),
)

FOOTNOTE = ("Everything runs on the CPU on 127.0.0.1. Models load once at startup from weights/*.pt, weights/ngram_lm.pkl, "
            "weights/bert_int8_dynamic.pt and the Hugging Face cache (bert-base-uncased).")


def mermaid(nodes: Sequence[Node] = NODES, edges: Sequence[Edge] = EDGES) -> str:
    """Mermaid flowchart text for the README (same nodes and edges as slide 4)."""
    lines = ["flowchart LR"]
    for node in nodes:
        lines.append(f'  {node.key}["{node.label.replace(chr(10), "<br/>")}"]')
    for edge in edges:
        lines.append(f"  {edge.src} -->|{edge.label}| {edge.dst}" if edge.label else f"  {edge.src} --> {edge.dst}")
    lines.append("  style B8 stroke-width:3px")
    return "\n".join(lines)


def _border(node: Node, toward: Node, box_h: float) -> tuple[float, float]:
    """Point where the line from node's centre toward the other node leaves node's box."""
    dx, dy = toward.x - node.x, toward.y - node.y
    scale = 1.0 / max(abs(dx) / (node.w / 2), abs(dy) / (box_h / 2), 1e-9)
    return node.x + dx * scale, node.y + dy * scale


def draw(ax: Any, box_h: float = 0.15) -> None:
    """Draw the same graph on a matplotlib axes in [0, 1] x [0, 1]; arrows start and end at the box borders."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    pos = {n.key: n for n in NODES}
    for edge in EDGES:
        a, b = pos[edge.src], pos[edge.dst]
        start, end = _border(a, b, box_h), _border(b, a, box_h)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#1f4fa3", lw=1.2, shrinkA=1, shrinkB=1, zorder=3))
        if edge.label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.018, edge.label, fontsize=7.5, color="#4a5466", ha="center", zorder=4,
                    bbox={"fc": "#fbf8f1", "ec": "none", "pad": 0.5})
    for node in NODES:
        ax.add_patch(FancyBboxPatch((node.x - node.w / 2, node.y - box_h / 2), node.w, box_h, boxstyle="round,pad=0.004",
                                    fc="#fff4cc" if node.primary else "#ffffff", ec="#1f4fa3", lw=2.2 if node.primary else 1.2, zorder=2))
        ax.text(node.x, node.y, node.label, ha="center", va="center", fontsize=7.2, color="#1d2433", zorder=4)
    ax.text(0.5, 0.07, FOOTNOTE, ha="center", fontsize=9.5, color="#4a5466")
