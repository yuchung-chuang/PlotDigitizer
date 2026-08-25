# 23 — Remove fitted trend lines

**What to build:** Fitted or regression trend lines stop being mistaken for data points when a chart
contains discrete markers and a line drawn through or beside them. This is a noise-filter operation,
not line-series extraction: a genuine measured line remains data, while an explicitly identified
trend line is removed before point extraction.

The filter must be conservative. A straight or smooth line is not enough evidence by itself, because
some supported figures contain real line series. The agent supplies the trend-line identity from the
figure, legend, style profile or an explicit region/style selection; the operation records that
reason and declines an ambiguous line rather than deleting it silently.

**Blocked by:** 10 — Noise filter group, mask layers, and border removal; 06 — Extract a single
point series.

**Status:** ready-for-agent

**Validation scope:** The corpus figures with a visible fitted, regression or explicitly labelled
trend line are:

1. `images/3-2-freefall-mimimum.png` - dotted quadratic `Fit` over discrete markers.
2. `images/Inseam-v-Height-Graph.jpg` - `Linear (Men)` and `Linear (Women)` fitted lines over
      two marker groups.
3. `images/VLObject-2561-031201081203.png` - labelled `Long-term Trend` over the monthly series.
4. `images/Screenshot 2024-09-15 131309.png` - `Typical`, `Minimum` and `Best fit` reference
      lines alongside the fatigue markers.
5. `images/Screenshot 2024-09-15 131624.png` - the labelled `K_Q = -0.25HV+143` regression
      line across the clustered measurements.
6. `images/Screenshot 2024-09-15 131643.png` - fitted lines over the `Tmax=32` and `Tmax=18`
      marker groups.

All six listed figures are acceptance fixtures: the skill must remove the identified trend lines in
each one before point extraction while preserving the underlying marker data. Begin implementation
and visual tuning with `images/Screenshot 2024-09-15 131643.png` as the smallest acceptance case,
then validate the remaining five figures. Ordinary measured curves in line-series, hysteresis and
point-and-line plots are preservation controls and must not be routed through this filter unless the
agent has identified a separate trend line.

- [ ] A trend-line filter skill is discoverable and explains the distinction between fitted trend
      lines and genuine line-series data.
- [ ] The filter requires identifying evidence or an explicit selection before claiming a line;
      ambiguous linework is reported as unresolved rather than removed.
- [ ] Running the filter writes a named `trend_lines` mask layer and records its pixel count and
      identification evidence in the extraction document.
- [ ] Trend-line masks are non-destructive, can be dropped and re-run independently, and contribute
      to the combined noise mask.
- [ ] The legend, protected regions, marker centres and genuine data-line series are preserved.
- [ ] Multiple trend-line styles or colours can be removed when each is identified separately.
- [ ] The overlay renders the claimed trend lines distinctly over the original image and makes
      preservation of the discrete markers visible.
- [ ] The filter reports confidence and structured diagnostics naming an ambiguous or unresolved
      trend line and its affected pixel region.
- [ ] On every listed acceptance fixture, trend-line pixels are removed and discrete point
      extraction no longer reports the trend lines as data points.
- [ ] The trend-line filter does not remove a genuine measured line on a line-series control.

**Acceptance result:** Pending implementation and fixture validation. Each acceptance fixture must
have an inspected trend-line mask overlay and a point-extraction overlay showing the marker centres
still present after filtering.

**Relationship to later tickets.** Ticket 18 consumes the filtered masks when extracting multiple
point series, while ticket 19 owns genuine ordered line-series extraction. Ticket 20 should include
this filter only for figures where the trend-line identity is resolved.
