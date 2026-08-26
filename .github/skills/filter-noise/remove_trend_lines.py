# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "opencv-python-headless>=4.10",
#   "scipy>=1.14",
#   "scikit-image>=0.24",
#   "scikit-learn>=1.5",
#   "pillow>=10.4",
# ]
# ///
"""Claim an identified fitted or regression trend line as a non-destructive noise mask layer.

Unlike its sibling filters, this one never decides on geometry alone: a straight or smooth line is
not enough evidence, because some supported figures carry a genuine line series that looks exactly
as tidy. A line is claimed only once the agent identifies it, by legend entry or by an explicit
region, and the reason is recorded alongside the pixels.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))

import cv2
import numpy as np
from scipy import ndimage

from pdkit import Overlay, Result, base_parser, finish, open_workspace, palette_colour, save_mask
from pdkit.ink import ink_mask
from pdkit.masks import load_mask, rebuild_combined

STAGE = "remove-trend-lines"
LAYER = "trend-lines"
DEFAULT_COLOUR_TOLERANCE = 20.0
DEFAULT_RADIUS = 4
DEFAULT_BAND = 2.5
MIN_LINE_PIXELS = 25
MIN_FIT_POINTS = 20
MIN_STRAIGHT_COVERAGE = 0.5
_SELECT_COLOUR_PREFIX = re.compile(r"^(\d{1,3}):(\d{1,3}):(\d{1,3}),(.*)$", re.DOTALL)


def main() -> int:
    parser = base_parser("Remove an identified fitted or regression trend line.")
    parser.add_argument(
        "--legend-entry",
        type=int,
        action="append",
        default=[],
        metavar="INDEX",
        help="identify a legend entry as a trend line, repeatable",
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="X,Y,WIDTH,HEIGHT,REASON",
        help="identify a trend line by an explicit region and the reason it is one, repeatable; "
        "prefix the reason with R:G:B, to scope it to one colour within the region",
    )
    parser.add_argument(
        "--colour-tolerance",
        type=float,
        default=DEFAULT_COLOUR_TOLERANCE,
        help="LAB colour distance a pixel may sit from the identified colour (default %(default)s)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_RADIUS,
        help="disk radius that separates a marker blob from line ink; 0 disables (default %(default)s)",
    )
    parser.add_argument(
        "--straight",
        action="store_true",
        help="tighten every claim to a robustly fitted straight line band",
    )
    parser.add_argument(
        "--band",
        type=float,
        default=DEFAULT_BAND,
        help="perpendicular half-width kept around the fitted line with --straight (default %(default)s)",
    )
    parser.add_argument("--drop", action="store_true", help="drop this layer and rebuild the combined mask")
    args = parser.parse_args()

    workdir, extraction, image = open_workspace(args)
    result = Result(stage=STAGE)
    if args.drop:
        return _drop(workdir, extraction, image, result)

    area = extraction.data.get("plot_area") or {}
    if not all(key in area for key in ("x", "y", "width", "height")):
        result.fail("no plot area recorded; run locate_plot_area.py first")
        return finish(workdir, extraction, STAGE, Path(__file__).name, result, ["trend lines none"])

    if not args.legend_entry and not args.select:
        result.fail("no trend line identified; supply --legend-entry, --select, or --drop")
        return finish(workdir, extraction, STAGE, Path(__file__).name, result, ["trend lines none"])

    height, width = image.shape[:2]
    area_roi, _ = _roi_mask((height, width), area)

    lines = [
        _from_legend(image, extraction, index, area_roi, args.colour_tolerance, args.radius, result)
        for index in args.legend_entry
    ]
    for position, value in enumerate(args.select):
        parsed = _parse_select(value, result)
        if parsed is None:
            continue
        box, reason, colour_rgb = parsed
        lines.append(
            _from_select(image, f"select-{position}", box, reason, colour_rgb, args.colour_tolerance, args.radius, result)
        )

    combined = np.zeros((height, width), dtype=bool)
    recorded = []
    for entry in lines:
        mask = entry.pop("mask", None)
        pixels = int(mask.sum()) if mask is not None else 0
        resolved = mask is not None and pixels >= MIN_LINE_PIXELS
        if resolved and args.straight:
            tightened, coverage = _tighten_straight(mask, args.band)
            if coverage >= MIN_STRAIGHT_COVERAGE:
                mask = tightened
                pixels = int(mask.sum())
                entry["straight_coverage"] = round(coverage, 3)
            else:
                result.warn(
                    f"{entry['id']}: ink did not resemble a single straight line "
                    f"(best line spanned {coverage:.0%} of it); kept the untightened claim",
                    region=_int_box(entry.get("region")),
                )
        if not resolved:
            result.warn(
                f"{entry['id']}: too little matching ink to claim a trend line; reported as unresolved",
                region=_int_box(entry.get("region")),
            )
        elif mask is not None:
            combined |= mask
        entry["pixels"] = pixels
        entry["resolved"] = resolved
        entry["confidence"] = round(_line_confidence(entry), 3) if resolved else 0.0
        recorded.append(entry)

    combined = _exclude_existing(workdir, extraction, combined)
    combined = _spare(combined, extraction)
    path = workdir.masks / f"{LAYER}.png"
    pixels = save_mask(path, combined)
    extraction.add_mask_layer(LAYER, "noise", workdir.relative(path), pixels)
    extraction.data["trend_lines"] = {
        "colour_tolerance": args.colour_tolerance,
        "radius": args.radius,
        "straight": bool(args.straight),
        "band": args.band if args.straight else None,
        "lines": recorded,
    }
    combined_total = rebuild_combined(workdir, extraction, image.shape[:2])

    resolved_lines = [entry for entry in recorded if entry["resolved"]]
    result.confidence = round(float(np.mean([e["confidence"] for e in resolved_lines])), 3) if resolved_lines else 0.15
    overlay = _draw(image, workdir, extraction, recorded).save(workdir.overlay_path(STAGE))
    summary = [
        f"claims      {len(resolved_lines)} of {len(recorded)} identification(s) resolved",
        f"layer       {pixels} px   {workdir.relative(path)}",
        f"combined    {combined_total} px over {len(extraction.mask_layers('noise'))} noise layers",
        f"straight    {'band ' + format(args.band, 'g') + ' px' if args.straight else 'off'}",
    ]
    for entry in recorded:
        state = "resolved" if entry["resolved"] else "unresolved"
        summary.append(f"  {entry['id']}: {state}, {entry['pixels']} px - {entry['reason']}")
    return finish(workdir, extraction, STAGE, Path(__file__).name, result, summary, overlay)


def _from_legend(image, extraction, index, roi, tolerance, radius, result):
    entries = (extraction.data.get("legend") or {}).get("entries") or []
    identifier = f"legend-{index}"
    if not (0 <= index < len(entries)):
        message = f"legend entry {index} does not exist; run find_legend.py first"
        result.warn(message)
        return _blank(identifier, "legend", index, None, message)

    entry = entries[index]
    swatch = entry.get("swatch") or {}
    style = entry.get("style") or {}
    label = entry.get("label")
    colour_lab = style.get("colour_lab")
    colour_rgb = style.get("colour_rgb")
    sampled = False
    if not colour_lab:
        colour_lab, colour_rgb = _sample_colour(image, swatch)
        sampled = True
    if colour_lab is None:
        message = f"legend entry {index}: no colour could be resolved from its swatch"
        result.warn(message, region=_int_box(swatch))
        return _blank(identifier, "legend", index, swatch, message, label=label)

    mask = _strip_blobs(_colour_mask(image, colour_lab, tolerance, roi), radius)
    bits = [f"legend entry {index}" + (f" labelled '{label}'" if label else " (no label recorded)")]
    bits.append("colour sampled directly from the swatch" if sampled else "colour and style read from the profiled legend entry")
    return {
        "id": identifier,
        "source": "legend",
        "legend_index": index,
        "region": swatch,
        "colour_lab": [round(float(value), 2) for value in colour_lab],
        "colour_rgb": [int(value) for value in colour_rgb] if colour_rgb is not None else None,
        "line_style": style.get("line_style"),
        "label": label,
        "reason": "; ".join(bits),
        "mask": mask,
    }


def _from_select(image, identifier, box, reason, colour_rgb_hint, tolerance, radius, result):
    int_box = _int_box(box)
    if not reason:
        message = "explicit region has no identifying reason; declined as ambiguous"
        result.warn(f"{identifier}: {message}", region=int_box)
        return _blank(identifier, "select", None, box, message)

    roi, _ = _roi_mask(image.shape[:2], box)
    if colour_rgb_hint is not None:
        bgr = np.array(colour_rgb_hint[::-1], dtype=np.uint8).reshape(1, 1, 3)
        colour_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        colour_rgb = np.asarray(colour_rgb_hint)
        mask = _strip_blobs(_colour_mask(image, colour_lab, tolerance, roi), radius)
        reason = f"colour {tuple(colour_rgb_hint)} supplied explicitly: {reason}"
    else:
        colour_lab, colour_rgb = _sample_colour(image, box)
        mask = _strip_blobs(ink_mask(image) & roi, radius)
    return {
        "id": identifier,
        "source": "select",
        "legend_index": None,
        "region": box,
        "colour_lab": [round(float(value), 2) for value in colour_lab] if colour_lab is not None else None,
        "colour_rgb": [int(value) for value in colour_rgb] if colour_rgb is not None else None,
        "line_style": None,
        "label": None,
        "reason": f"explicit region selection: {reason}",
        "mask": mask,
    }


def _blank(identifier, source, legend_index, region, reason, label=None):
    return {
        "id": identifier,
        "source": source,
        "legend_index": legend_index,
        "region": region,
        "colour_lab": None,
        "colour_rgb": None,
        "line_style": None,
        "label": label,
        "reason": reason,
        "mask": None,
    }


def _parse_select(value, result):
    parts = value.split(",", 4)
    if len(parts) < 4:
        result.warn(f"ignored malformed --select {value!r}; expected x,y,width,height,reason")
        return None
    try:
        x, y, width, height = (float(part) for part in parts[:4])
    except ValueError:
        result.warn(f"ignored malformed --select {value!r}; x,y,width,height must be numbers")
        return None
    rest = parts[4].strip() if len(parts) > 4 else ""
    colour_rgb = None
    match = _SELECT_COLOUR_PREFIX.match(rest)
    if match:
        colour_rgb = tuple(int(group) for group in match.groups()[:3])
        rest = match.group(4).strip()
    return {"x": x, "y": y, "width": width, "height": height}, rest, colour_rgb


def _sample_colour(image, box):
    crop = _crop(image, box)
    if crop.size == 0:
        return None, None
    mask = ink_mask(crop)
    pixels = crop[mask] if mask.any() else crop.reshape(-1, 3)
    if pixels.size == 0:
        return None, None
    bgr = np.median(pixels, axis=0)
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    rgb = bgr[::-1].astype(int)
    return lab, rgb


def _colour_mask(image, colour_lab, tolerance, roi):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    distance = np.linalg.norm(lab_image - np.asarray(colour_lab, dtype=np.float32), axis=2)
    return (distance <= tolerance) & roi


def _strip_blobs(mask, radius):
    """Drop anything that survives an opening at marker scale, keeping thinner line ink.

    A marker's ring or disk is filled first, so an open marker outline is not mistaken for a line
    the way it would be if only the raw ink were opened.
    """
    if radius < 1 or not mask.any():
        return mask
    filled = ndimage.binary_fill_holes(mask)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0
    blobs = cv2.dilate(opened.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return mask & ~blobs


def _tighten_straight(mask, band, trials=300, seed=0):
    """Find the best-supported straight line through the claim by RANSAC, and keep its inliers.

    A regression line drawn over its own scatter cloud can be a minority of the candidate pixels,
    so a least-squares fit gets dragged toward the cloud instead of the line. Sampling many
    candidate lines from pairs of points and keeping the one whose inliers span the most of the
    claim survives that noise the way a single robust fit does not.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < MIN_FIT_POINTS:
        return mask, 0.0

    points = np.column_stack([xs, ys]).astype(np.float64)
    diagonal = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    if diagonal < 1:
        return mask, 0.0

    rng = np.random.default_rng(seed)
    pool_size = min(points.shape[0], 5000)
    pool = points if points.shape[0] <= pool_size else points[rng.choice(points.shape[0], pool_size, replace=False)]

    best_inliers, best_span = None, -1.0
    for _ in range(trials):
        first, second = rng.choice(pool.shape[0], 2, replace=False)
        origin, other = pool[first], pool[second]
        direction = other - origin
        length = float(np.hypot(direction[0], direction[1]))
        if length < 0.3 * diagonal:
            continue  # too short a baseline to trust the heading of a line spanning the claim
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        inliers = np.abs((points - origin) @ normal) <= band
        if inliers.sum() < MIN_FIT_POINTS:
            continue
        projection = (points[inliers] - origin) @ unit
        span = float(projection.max() - projection.min())
        if span > best_span:
            best_span, best_inliers = span, inliers

    if best_inliers is None:
        return mask, 0.0
    coverage = best_span / diagonal
    tightened = np.zeros(mask.shape, dtype=bool)
    tightened[ys[best_inliers], xs[best_inliers]] = True
    return tightened, coverage


def _line_confidence(entry):
    confidence = 0.75 if entry["source"] == "legend" else 0.55
    if entry.get("label"):
        confidence += 0.1
    if entry.get("line_style") not in (None, "none"):
        confidence += 0.05
    if entry.get("straight_coverage") is not None:
        confidence += 0.05
    return min(confidence, 0.95)


def _roi_mask(shape, box):
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    x = max(0, int(round(box.get("x", 0))))
    y = max(0, int(round(box.get("y", 0))))
    right = min(width, int(round(box.get("x", 0) + box.get("width", 0))))
    bottom = min(height, int(round(box.get("y", 0) + box.get("height", 0))))
    mask[y:bottom, x:right] = True
    return mask, (x, y, right, bottom)


def _crop(image, box):
    height, width = image.shape[:2]
    x = max(0, int(round(box.get("x", 0))))
    y = max(0, int(round(box.get("y", 0))))
    right = min(width, int(round(box.get("x", 0) + box.get("width", 0))))
    bottom = min(height, int(round(box.get("y", 0) + box.get("height", 0))))
    return image[y:bottom, x:right]


def _int_box(box):
    if not box:
        return None
    return {key: int(round(box.get(key, 0))) for key in ("x", "y", "width", "height")}


def _exclude_existing(workdir, extraction, claim):
    for layer in extraction.mask_layers(kind="noise"):
        if layer.get("name") == LAYER:
            continue
        path = workdir.path / layer["file"]
        if path.exists():
            claim &= ~load_mask(path)
    return claim


def _spare(claim, extraction):
    for region in extraction.data.get("protected_regions", []):
        x, y = max(0, int(round(region.get("x", 0)))), max(0, int(round(region.get("y", 0))))
        right = min(claim.shape[1], int(round(region.get("x", 0) + region.get("width", 0))))
        bottom = min(claim.shape[0], int(round(region.get("y", 0) + region.get("height", 0))))
        claim[y:bottom, x:right] = False
    return claim


def _drop(workdir, extraction, image, result):
    path = workdir.masks / f"{LAYER}.png"
    path.unlink(missing_ok=True)
    extraction.data["mask_layers"] = [layer for layer in extraction.data["mask_layers"] if layer.get("name") != LAYER]
    extraction.data.pop("trend_lines", None)
    result.confidence = 1.0
    combined = rebuild_combined(workdir, extraction, image.shape[:2])
    overlay = _draw(image, workdir, extraction, ()).save(workdir.overlay_path(STAGE))
    return finish(workdir, extraction, STAGE, Path(__file__).name, result, [f"dropped     {LAYER}", f"combined    {combined} px"], overlay)


def _draw(image, workdir, extraction, recorded):
    overlay = Overlay(image)
    for index, layer in enumerate(extraction.mask_layers(kind="noise")):
        path = workdir.path / layer["file"]
        if path.exists():
            overlay.tint(load_mask(path), palette_colour(index), label=layer["name"])
    for region in extraction.data.get("protected_regions", []):
        overlay.rectangle(region, palette_colour(6), label=f"protected: {region.get('name', '?')}")
    for entry in recorded:
        region = entry.get("region")
        if not region:
            continue
        colour = (40, 180, 40) if entry.get("resolved") else (40, 40, 220)
        state = "claimed" if entry.get("resolved") else "unresolved"
        overlay.rectangle(region, colour, label=f"{entry['id']}: {state}", thickness=1)
    return overlay


if __name__ == "__main__":
    raise SystemExit(main())
