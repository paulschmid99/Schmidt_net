# Schmidt_net
Python toolbox for lower-hemisphere Schmidt net (equal-area) stereographic projection, including density plotting (KDE), automatic joint-set clustering, and kinematic slope stability analysis (plane, wedge, and toppling failure).
# Schmidt-Toolbox for Structural Geology & Kinematic Slope Analysis

This Python module provides a collection of utilities to work with lower-hemisphere Schmidt (Lambert equal-area) projections for:

- plotting poles, planes, lineations and density maps  
- clustering joint sets and analysing their intersections  
- kinematic stability analysis (plane, wedge, toppling, orthotilt)  
- visualising slope–discontinuity systems and their intersection lineations  

The focus is on a transparent, reproducible workflow suitable for teaching, engineering practice, and scientific publications.

---

## Dependencies

- Python 3.x  
- `numpy`  
- `matplotlib`  
- `pandas`

---

## Core Functionality Overview

### 1. Projection & Stereonet

- **`lambert_schmidt_xy(phi_deg, lam_deg, lam0_deg=90)`**  
  Forward Lambert equal-area projection (Schmidt net) from latitude/longitude-like angles to (X, Y) in the unit disc.

- **`plot_schmidt_net(grid_step=10, fine_step=5, figsize=(6, 6))`**  
  Draws a standard lower-hemisphere Schmidt net with great circles, crosshair, and azimuth labels. Returns `(fig, ax)`.

- **`trend_plunge_to_xyz(trend_deg, plunge_deg)`**  
  Converts line orientation (trend, plunge) to a 3D unit vector (x, y, z).

- **`project_trend_plunge(trend_deg, plunge_deg)`**  
  Projects a line (trend, plunge) into the Schmidt net (X, Y) using the equal-area formula.

- **`xyz_to_trend_plunge(x, y, z)`**  
  Inverse of `trend_plunge_to_xyz` (for lower hemisphere): returns trend and plunge of a 3D vector.

- **`schmidt_xy_to_xyz(X, Y)`**  
  Inverse Lambert equal-area projection: maps (X, Y) in the Schmidt net back to a 3D unit vector (x, y, z).

- **`xy_to_xyz(X, Y)`**  
  Alternative inverse projection helper used in the kinematic routines (also returns a 3D unit vector).

- **`az_from_xy(X, Y)`**  
  Computes an azimuth in the Schmidt net coordinate system (0° = N, 90° = E, …) from projected (X, Y).

---

### 2. Planes, Poles & Lineations

- **`plane_pole_from_dipdir(dipdir_deg, dip_deg)`**  
  Computes the pole of a plane from dip direction and dip.

- **`great_circle_from_plane(dipdir_deg, dip_deg, n_points=361)`**  
  Returns the great-circle trace of a plane (as X, Y arrays) in the Schmidt net, masking the upper hemisphere.

- **`plot_plane_and_pole(ax, dipdir_deg, dip_deg, ...)`**  
  Plots a plane (great circle) and its pole into an existing Schmidt net `ax`.

- **`plot_linear(ax, trend_deg, plunge_deg, ...)`**  
  Plots a linear feature (lineation) as a cross symbol in the Schmidt net.

---

### 3. Input Tables & Simple Plots

- **`load_orientation_table(path)`**  
  Reads an orientation table (CSV / Excel) with columns:  
  `type` (`'plane'` or `'linear'`), `dir` (dip direction or trend), `angle` (dip or plunge).

- **`plot_from_table(path, grid_step=10, figsize=(6, 6))`**  
  Draws planes (great circles + poles) and lineations from a table of orientations on a Schmidt net.

---

### 4. Density Mapping & Clustering

- **`kde_on_disc(xs, ys, grid_n=200, sigma=0.08, threshold=0.05)`**  
  Computes a simple 2D Gaussian kernel density estimate on the unit disc for a set of projected points.  
  Returns density grid `D` and coordinate vectors `(x, y)`. The density is normalised to 0–1 and values below `threshold` are set to `NaN`.

- **`find_clusters_from_density(D, x_vec, y_vec, min_pixels=10)`**  
  Finds connected areas (“patches”) in a density grid `D` (using an 8-neighbour flood fill).  
  Returns a label grid and the centres of the patches in (x, y), discarding clusters with fewer than `min_pixels` cells.

- **`plot_density_from_table(path, ...)`**  
  Reads `plane` and `linear` orientations from a table, projects their poles/lineations, and overlays density maps on a Schmidt net:  
  - plane poles in a red heatmap  
  - lineations in a blue heatmap  
  Both are normalised to 0–1, with dedicated colour scales (colorbars) for publication-ready interpretation.

- **`cluster_planes_from_table(path, ...)`**  
  Density-based clustering of plane poles:  
  - builds a KDE, segments density clusters,  
  - groups poles into clusters,  
  - computes mean planes (dipdir, dip) per cluster,  
  - writes a CSV with mean orientations,  
  - generates two plots (all planes + clusters; mean planes only).

---

### 5. Intersections & “Pole of Intersections”

- **`intersections_from_means_csv(path_means, ...)`**  
  Reads a CSV of mean planes (from `cluster_planes_from_table`), computes:  
  - all pairwise intersection lineations (trend, plunge),  
  - angle between planes,  
  - plots mean planes + intersection lineations on a Schmidt net,  
  - writes a CSV summarising the intersections.

- **`add_intersection_poles_with_plot(path_means, path_intersections, ...)`**  
  Reads mean planes and their intersections, computes an additional “pole” for each intersection lineation (according to your definition), and:  
  - appends these to the intersection CSV,  
  - plots mean planes, intersection lineations (bicolour crosses) and their “poles” (bicolour squares) in a single Schmidt net.

---

### 6. Kinematic Slope Stability: Critical Zones

These functions construct “critical zones” in pole space for slope stability analysis (plane, wedge, toppling, orthotilt), based on a slope plane and friction angle.

- **`plot_plane_failure_zone(slope_dipdir, slope_dip, friction_angle, ...)`**  
  Computes and plots the classic planar failure zone: poles of joints that can slide on the slope, based on dip range and azimuth tolerance.  
  Zone is shown as horizontally hatched area with dashed outline.

- **`plot_wedge_failure_marklandsche_area(slope_dipdir, slope_dip, friction_angle=None, ...)`**  
  Constructs the “Markland area” / Critical Pole Vector Zone by using “poles of lineations” derived from the slope great circle.  
  Optionally cuts out an inner friction cone. Zone is plotted with vertical hatching and dashed outline.

- **`plot_toppling_failure_zone(slope_dipdir, slope_dip, friction_angle, ...)`**  
  Computes the toppling failure zone, governed by “backward-dipping” joints that are steep enough relative to the slope and friction angle.  
  Shown as diagonally hatched red zone with dashed outline.

- **`plot_plane_failure_zone_orthotilt(slope_dipdir, slope_dip, friction_angle, max_orthotilt=20, ...)`**  
  Variant of the planar failure zone with an additional constraint on the allowable “orthogonal tilt” between a joint and a reference joint family aligned with the slope.

- **`plot_all_critical_zones(slope_dipdir, slope_dip, friction_angle, ...)`**  
  Combines any subset of the above zones (plane, orthotilt, wedge, toppling) into a single Schmidt net for an overview of the kinematic stability of a given slope.

---

### 7. Joint-by-Joint and Pairwise Kinematic Analysis

- **`plot_joint_critical_zones_from_table(path, slope_dipdir, slope_dip, ...)`**  
  For each discontinuity in a table (`name, dip, dipdir, friction_angle`):  
  - draws the slope and selected critical zones,  
  - plots the joint plane and its pole,  
  - checks whether the pole falls into any zone and prints a red warning or green “uncritical” text,  
  - produces a summary table (critical vs. uncritical joints, failure type).

- **`plot_joint_pair_critical_zones_from_table(path, slope_dipdir, slope_dip, ...)`**  
  Analyses all pairs of joints:  
  - computes the intersection lineation and its “pole”,  
  - plots slope, both joints, intersection lineation and its “pole” in the Schmidt net,  
  - tests if the “pole of the intersection” lies in any critical zone (plane, wedge, etc.),  
  - outputs detailed textual interpretation (plane vs wedge failure geometry) and a summary table.

---

### 8. Slope + Discontinuity Network Overview

- **`load_discontinuity_table(path)`**  
  Reads a table with columns: `name, dip, dipdir, friction`.

- **`plot_slope_and_discontinuities_from_table(slope_dip_deg, slope_dipdir_deg, friction_angle_deg, path, ...)`**  
  Produces a comprehensive Schmidt-plot of:  
  - the slope plane and its fall-direction tick on the rim,  
  - all joint planes with ticks on the rim,  
  - all intersection lineations between joints as bicolour crosses,  
  - arrows on the rim indicating the plunge direction of each intersection lineation,  
  - a semi-transparent green region representing “uncritical” orientations due to the global friction angle.  
  Optionally writes a CSV combining slope, joints and intersection information.

---

### 9. Colour & Symbol Helpers

Mostly used internally, but available if you want the same visual style:

- **`lighten_color(color, amount=0.5)`**  
  Mixes a Matplotlib colour with white by a given fraction.

- **`plot_bicolor_cross(ax, X, Y, color1, color2, size=0.03, linewidth=2.5)`**  
  Draws a two-colour “X” at (X, Y), used for intersection lineations.

- **`plot_bicolor_square(ax, X, Y, color1, color2, size=0.03, ...)`**  
  Draws a two-colour square symbol (two triangles) at (X, Y), used for “poles” of intersection lineations.

- **`plot_bicolor_arrow_on_rim(ax, az_deg, color1, color2, ...)`**  
  Draws a two-colour arrow on the outer rim for a given azimuth, used to show the plunge direction of intersection lineations.

---

### 10. Miscellaneous Utilities

- **`circular_diff(a, b)`**  
  Minimal signed angular difference between two azimuths (in degrees, range −180°…+180°).  
  Used throughout for azimuth comparisons without problems at 0°/360°.

---

## Example (Very Short)

```python
from schmidt_tools import (
    plot_schmidt_net,
    plot_density_from_table,
    plot_all_critical_zones
)

# 1) Basic Schmidt net
fig, ax = plot_schmidt_net()

# 2) Density plot from orientation file
plot_density_from_table("orientations.csv", save_path="density.png")

# 3) Kinematic overview for a slope
plot_all_critical_zones(
    slope_dipdir=120,
    slope_dip=45,
    friction_angle=30,
    show_plane=True,
    show_wedge=True,
    show_toppling=True,
    save_fig="critical_zones.png"
)
