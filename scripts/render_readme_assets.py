"""Render deterministic SVG figures used by the repository README."""

# ruff: noqa: E501

from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"

INK = "#17242b"
MUTED = "#627078"
PAPER = "#f7f4eb"
WHITE = "#fffdf8"
TEAL = "#0b6d73"
TEAL_LIGHT = "#dcefed"
RUST = "#b0522f"
RUST_LIGHT = "#f4dfd3"
GOLD = "#dda12c"
GREEN = "#2f7a4c"
LINE = "#d8d3c6"


def write_svg(name: str, body: str, *, width: int, height: int) -> None:
    ASSETS.mkdir(exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: {INK}; }}
.title {{ font-family: Georgia, serif; font-size: 32px; font-weight: 700; }}
.subtitle {{ font-size: 15px; fill: {MUTED}; }}
.label {{ font-size: 14px; font-weight: 700; }}
.small {{ font-size: 12px; fill: {MUTED}; }}
.value {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: 13px; }}
</style>
<rect width="100%" height="100%" fill="{PAPER}" rx="24"/>
{body}
</svg>
"""
    (ASSETS / name).write_text(svg)


def card(x: int, y: int, width: int, height: int, *, fill: str = WHITE, stroke: str = LINE) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )


def disk(x: float, y: float, radius: float, color: str) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>'


def overview() -> None:
    body = [
        '<text x="56" y="62" class="title">One Event Can Produce Many Videos</text>',
        '<text x="56" y="92" class="subtitle">The audit separates pixel reuse, trajectory reuse, local generalization, and independent events.</text>',
        card(54, 128, 260, 150, fill=TEAL_LIGHT, stroke=TEAL),
        '<text x="78" y="160" class="label">Training exposure</text>',
        '<line x1="82" y1="220" x2="286" y2="220" stroke="#718187" stroke-width="3"/>',
        disk(125, 220, 17, TEAL),
        disk(242, 220, 17, GOLD),
        '<path d="M143 205 C175 173 207 173 225 205" fill="none" stroke="#718187" stroke-width="2" stroke-dasharray="5 5"/>',
        '<text x="78" y="258" class="small">source trajectory S</text>',
        '<line x1="330" y1="202" x2="404" y2="202" stroke="#718187" stroke-width="3"/>',
        '<polygon points="404,202 392,195 392,209" fill="#718187"/>',
        card(418, 146, 250, 112, fill=WHITE, stroke=TEAL),
        '<text x="448" y="187" class="label">Continued training</text>',
        '<text x="448" y="216" class="small">matched updates and checkpoints</text>',
        '<line x1="684" y1="202" x2="754" y2="202" stroke="#718187" stroke-width="3"/>',
        '<polygon points="754,202 742,195 742,209" fill="#718187"/>',
        card(768, 128, 378, 150, fill=RUST_LIGHT, stroke=RUST),
        '<text x="792" y="160" class="label">Evaluation populations</text>',
        '<text x="792" y="193" class="small">same trajectory · unseen appearance</text>',
        '<text x="792" y="219" class="small">nearby trajectory · new event</text>',
        '<text x="792" y="245" class="small">unrestricted independent event</text>',
        '<text x="56" y="338" class="title">The Quantity We Measure</text>',
        card(54, 372, 1092, 156),
        '<text x="82" y="414" class="label">Source-specific generalization gap</text>',
        '<text x="82" y="455" style="font-family:ui-monospace,monospace;font-size:22px">(gain on unseen rerenders) − (gain on independent events)</text>',
        '<text x="82" y="495" class="subtitle">A positive value means the training policy benefits more from reused physical events than from new ones.</text>',
        '<rect x="54" y="554" width="1092" height="2" fill="#d8d3c6"/>',
        '<text x="56" y="590" class="small">Controlled lineage provides the ground truth. It is not inferred from model outputs.</text>',
    ]
    write_svg("overview.svg", "\n".join(body), width=1200, height=620)


def memory_chart() -> None:
    result = json.loads((RESULTS / "memory_positive_control.json").read_text())["aggregate"]
    values = [
        ("Source-family gain", result["mean_source_gain"], TEAL),
        ("Independent-population gain", result["mean_population_gain"], RUST),
        ("50% evaluation-reuse effect", result["mean_evaluation_reuse_effect"], GOLD),
    ]
    maximum = 0.04
    body = [
        '<text x="52" y="58" class="title">Memory Positive Control</text>',
        '<text x="52" y="88" class="subtitle">Nearest-trajectory retrieval · 64 independent simulation blocks</text>',
    ]
    for index, (label, value, color) in enumerate(values):
        y = 145 + index * 82
        width = 650 * value / maximum
        body.extend(
            [
                f'<text x="52" y="{y}" class="label">{escape(label)}</text>',
                f'<rect x="290" y="{y - 22}" width="650" height="28" rx="14" fill="#e7e3d8"/>',
                f'<rect x="290" y="{y - 22}" width="{width:.2f}" height="28" rx="14" fill="{color}"/>',
                f'<text x="958" y="{y}" class="value">{value:.6f}</text>',
            ]
        )
    threshold_x = 290 + 650 * (1 / 64) / maximum
    body.extend(
        [
            f'<line x1="{threshold_x:.2f}" y1="112" x2="{threshold_x:.2f}" y2="330" stroke="{INK}" stroke-width="2" stroke-dasharray="5 5"/>',
            f'<text x="{threshold_x + 8:.2f}" y="325" class="small">pre-specified 1/64 threshold</text>',
            card(52, 388, 996, 90, fill=TEAL_LIGHT, stroke=TEAL),
            f'<text x="78" y="427" class="label" fill="{GREEN}">Gate: PASS</text>',
            '<text x="78" y="457" class="subtitle">95% lower bound: 0.016127 · threshold: 0.015625</text>',
        ]
    )
    write_svg("memory-positive-control.svg", "\n".join(body), width=1100, height=520)


def phyworld_chart() -> None:
    result = json.loads((RESULTS / "phyworld_state_baselines.json").read_text())
    rows = result["results"]
    maximum = 0.55
    body = [
        '<text x="52" y="58" class="title">PhyWorld Collision Baselines</text>',
        '<text x="52" y="88" class="subtitle">Observe 4 state frames · predict 28 · lower normalized center error is better</text>',
        f'<rect x="730" y="42" width="14" height="14" rx="3" fill="{RUST}"/>',
        '<text x="752" y="54" class="small">constant velocity</text>',
        f'<rect x="900" y="42" width="14" height="14" rx="3" fill="{TEAL}"/>',
        '<text x="922" y="54" class="small">nearest trajectory</text>',
    ]
    for index, row in enumerate(rows):
        y = 150 + index * 122
        constant = row["constant_velocity"]["mean_error"]
        nearest = row["nearest_trajectory"]["mean_error"]
        body.extend(
            [
                f'<text x="52" y="{y}" class="label">{escape(row["name"])}</text>',
                f'<rect x="260" y="{y - 35}" width="{620 * constant / maximum:.2f}" height="25" rx="12" fill="{RUST}"/>',
                f'<text x="{278 + 620 * constant / maximum:.2f}" y="{y - 16}" class="value">{constant:.4f}</text>',
                f'<rect x="260" y="{y + 5}" width="{620 * nearest / maximum:.2f}" height="25" rx="12" fill="{TEAL}"/>',
                f'<text x="{278 + 620 * nearest / maximum:.2f}" y="{y + 24}" class="value">{nearest:.4f}</text>',
            ]
        )
    body.extend(
        [
            card(52, 510, 996, 72, fill=TEAL_LIGHT, stroke=TEAL),
            '<text x="76" y="540" class="label">Finding</text>',
            '<text x="76" y="566" class="subtitle">Nearest-trajectory retrieval beats constant velocity on all three evaluated splits.</text>',
        ]
    )
    write_svg("phyworld-baselines.svg", "\n".join(body), width=1100, height=620)


def main() -> None:
    overview()
    memory_chart()
    phyworld_chart()
    print("Rendered README assets in assets/")


if __name__ == "__main__":
    main()
