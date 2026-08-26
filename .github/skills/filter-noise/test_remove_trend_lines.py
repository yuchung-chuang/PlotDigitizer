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
"""Command-level regression checks for identified trend-line removal."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILLS = ROOT / ".github" / "skills"

FIGURE = "images/Screenshot 2024-09-15 131643.png"
PLOT_AREA_BOX = "88.5,26.5,591,361"
# The straight line connecting two of the four Tmax=32 circle markers.
TREND_LINE_SELECT = (
    "375,112,213,43,solid line connecting two Tmax=32 markers, distinct from the open circle markers"
)
# Real marker centres for the Tmax=32 series, read from the chart.
TMAX_32_MARKERS = [(383.26, 79.55), (383.42, 120.21), (580.28, 147.21), (383.22, 159.93)]


class RemoveTrendLinesTests(unittest.TestCase):
    def test_declines_without_any_identification(self) -> None:
        with tempfile.TemporaryDirectory(prefix="plot-digitizer-ticket23-") as temporary:
            workdir = Path(temporary)
            self._locate_plot_area(workdir)
            completed = self._run_raw(
                "filter-noise", "remove_trend_lines.py", "--image", FIGURE, "--workdir", str(workdir)
            )
            self.assertEqual(completed.returncode, 1, completed.stdout)
            self.assertIn("no trend line identified", completed.stdout)

    def test_select_without_reason_is_declined_as_unresolved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="plot-digitizer-ticket23-") as temporary:
            workdir = Path(temporary)
            self._locate_plot_area(workdir)
            self._run(
                "filter-noise",
                "remove_trend_lines.py",
                "--image",
                FIGURE,
                "--workdir",
                str(workdir),
                "--select",
                "375,112,213,43",
            )

            extraction = self._extraction(workdir)
            lines = extraction["trend_lines"]["lines"]
            self.assertEqual(len(lines), 1)
            self.assertFalse(lines[0]["resolved"])
            self.assertEqual(lines[0]["pixels"], 0)
            layer = self._layer(extraction, "trend-lines")
            self.assertEqual(layer["pixels"], 0)

    def test_explicit_selection_claims_the_line_and_spares_its_markers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="plot-digitizer-ticket23-") as temporary:
            workdir = Path(temporary)
            self._locate_plot_area(workdir)
            self._run(
                "filter-noise",
                "remove_trend_lines.py",
                "--image",
                FIGURE,
                "--workdir",
                str(workdir),
                "--select",
                TREND_LINE_SELECT,
            )

            extraction = self._extraction(workdir)
            lines = extraction["trend_lines"]["lines"]
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0]["resolved"], lines[0]["reason"])
            self.assertGreater(lines[0]["pixels"], 30)

            layer = self._layer(extraction, "trend-lines")
            self.assertEqual(layer["kind"], "noise")
            self.assertGreater(layer["pixels"], 0)

            mask = self._load_mask(workdir / layer["file"])
            for x, y in TMAX_32_MARKERS:
                self.assertEqual(
                    _disk_sum(mask, x, y, radius=6),
                    0,
                    f"marker at ({x}, {y}) lost pixels to the trend-line mask",
                )

    def test_drop_removes_the_layer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="plot-digitizer-ticket23-") as temporary:
            workdir = Path(temporary)
            self._locate_plot_area(workdir)
            self._run(
                "filter-noise",
                "remove_trend_lines.py",
                "--image",
                FIGURE,
                "--workdir",
                str(workdir),
                "--select",
                TREND_LINE_SELECT,
            )
            self._run("filter-noise", "remove_trend_lines.py", "--image", FIGURE, "--workdir", str(workdir), "--drop")

            extraction = self._extraction(workdir)
            self.assertIsNone(self._layer(extraction, "trend-lines", required=False))
            self.assertNotIn("trend_lines", extraction)
            self.assertFalse((workdir / "masks" / "trend-lines.png").exists())

    def _locate_plot_area(self, workdir: Path) -> None:
        self._run(
            "analyse-axis",
            "locate_plot_area.py",
            "--image",
            FIGURE,
            "--workdir",
            str(workdir),
            "--box",
            PLOT_AREA_BOX,
        )

    def _extraction(self, workdir: Path) -> dict:
        return json.loads((workdir / "extraction.json").read_text(encoding="utf-8"))

    def _layer(self, extraction: dict, name: str, required: bool = True) -> dict | None:
        layer = next((entry for entry in extraction["mask_layers"] if entry.get("name") == name), None)
        if required:
            self.assertIsNotNone(layer, f"no mask layer named {name!r}")
        return layer

    def _load_mask(self, path: Path):
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image, f"could not read mask {path}")
        return image > 0

    def _run(self, group: str, script: str, *arguments: str) -> None:
        completed = self._run_raw(group, script, *arguments)
        if completed.returncode:
            self.fail(f"{script} exited {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")

    def _run_raw(self, group: str, script: str, *arguments: str) -> subprocess.CompletedProcess:
        command = [sys.executable, str(SKILLS / group / script), *arguments]
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _disk_sum(mask, x: float, y: float, radius: int) -> int:
    height, width = mask.shape[:2]
    total = 0
    cx, cy = int(round(x)), int(round(y))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            px, py = cx + dx, cy + dy
            if 0 <= px < width and 0 <= py < height and mask[py, px]:
                total += 1
    return total


if __name__ == "__main__":
    unittest.main()
