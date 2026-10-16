# -*- coding: utf-8 -*-
"""Mấy hình SVG trang trí trang chủ + CSS chung.

Tách ra cho gọn, bản chất vẫn là code copy từ file cũ sang.
"""
import math


def one_line(html):
    return " ".join(line.strip() for line in html.strip().splitlines())


def _curve(t):
    # cột sống nhìn nghiêng: gù ngực rồi ưỡn thắt lưng
    return (175 - 34 * math.sin(2 * math.pi * t) + 14 * (1 - t) ** 3, 70 + 330 * t)


def _tangent(t, e=1e-3):
    (x0, y0), (x1, y1) = _curve(t - e), _curve(t + e)
    return x1 - x0, y1 - y0


def spine_svg(mode):
    n = 19
    parts = []
    for i in range(n):
        t = i / (n - 1)
        x, y = _curve(t)
        dx, dy = _tangent(t)
        ang = -math.degrees(math.atan2(dx, dy))
        w, h = 24 + 12 * t, 14 + 5 * t
        parts.append(
            f'<rect class="v" style="animation-delay:{i * 55}ms" x="{-w / 2:.1f}" y="{-h / 2:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" rx="5" transform="translate({x:.1f} {y:.1f}) rotate({ang:.1f})"/>'
        )
    body = "".join(parts)
    if mode == "hero":
        ex, ey = 192, 40
        sx, sy = 183, 110
        hx, hy = 194, 354
        body = (
            f'<circle cx="200" cy="36" r="27" fill="none" stroke="#8FB4D9" stroke-width="2"/>' + body +
            f'<g class="lm">'
            f'<line x1="{ex}" y1="{ey}" x2="{ex}" y2="{hy + 14}" stroke="#E7EEF7" stroke-width="1.5" stroke-dasharray="4 5"/>'
            f'<line x1="{ex}" y1="{ey}" x2="{sx}" y2="{sy}" stroke="#FFB547" stroke-width="2.5"/>'
            f'<line x1="{sx}" y1="{sy}" x2="{sx + 70}" y2="{sy}" stroke="#FFB547" stroke-width="1.5" stroke-dasharray="4 4"/>'
            f'<path d="M {sx + 34} {sy} A 34 34 0 0 0 {sx + 34 * math.cos(math.radians(-83)):.1f} '
            f'{sy + 34 * math.sin(math.radians(-83)):.1f}" fill="none" stroke="#FFB547" stroke-width="2"/>'
            f'<circle cx="{ex}" cy="{ey}" r="6" fill="#FFB547"/><circle cx="{sx}" cy="{sy}" r="6" fill="#FFB547"/>'
            f'<circle cx="{hx}" cy="{hy}" r="6" fill="#FFB547"/>'
            f'<text x="236" y="44" class="tx">Tai</text><text x="216" y="132" class="tx">Vai</text>'
            f'<text x="216" y="360" class="tx">Hông</text>'
            f'<text x="{sx + 44}" y="{sy - 24}" class="tx s">góc tai–vai</text></g>'
        )
        return (
            f'<svg class="art" viewBox="0 0 360 430" role="img" '
            f'aria-label="Sơ đồ cột sống nhìn nghiêng với các mốc tai, vai, hông">{body}</svg>'
        )
    labels = ["T2", "T4", "T6", "T8", "T10", "T12"]
    ts = [0.10 + i * (0.45 / 5) for i in range(6)]
    marks = []
    for lab, t in zip(labels, ts):
        x, y = _curve(t)
        dx, dy = _tangent(t)
        ln = math.hypot(dx, dy)
        nx, ny = -dy / ln, dx / ln
        marks.append((lab, t, x + nx * 30, y + ny * 30, dx / ln, dy / ln))
    g = ""
    for lab, t, mx, my, ux, uy in marks:
        if lab in ("T2", "T12"):
            g += (
                f'<line x1="{mx - ux * 60:.1f}" y1="{my - uy * 60:.1f}" '
                f'x2="{mx + ux * 60:.1f}" y2="{my + uy * 60:.1f}" '
                f'stroke="#E08A00" stroke-width="2" stroke-dasharray="5 4"/>'
            )
    for lab, t, mx, my, ux, uy in marks:
        g += (
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="8" fill="#FFB547" stroke="#fff" stroke-width="2"/>'
            f'<text x="{mx - 16:.1f}" y="{my + 5:.1f}" text-anchor="end" class="tk">{lab}</text>'
        )
    return (
        f'<svg class="art" viewBox="0 0 360 430" role="img" '
        f'aria-label="Sơ đồ 6 nhãn dán trên lưng từ T2 đến T12">{body}{g}</svg>'
    )
