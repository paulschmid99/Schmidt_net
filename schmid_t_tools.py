# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 13:46:29 2025

@author: Paul
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import os

# -------------------------------------------------------
# Lambert-Azimutal-Gleichflächenprojektion (Schmidt)
# -------------------------------------------------------

def lambert_schmidt_xy(phi_deg, lam_deg, lam0_deg=90.0):
    phi = np.deg2rad(phi_deg)
    lam = np.deg2rad(lam_deg)
    lam0 = np.deg2rad(lam0_deg)

    R = np.sqrt(2.0) / 2.0
    kp = np.sqrt(2.0 / (1.0 + np.cos(phi) * np.cos(lam - lam0)))

    x = R * kp * np.cos(phi) * np.sin(lam - lam0)
    y = R * kp * np.sin(phi)
    return x, y


def plot_schmidt_net(grid_step=10, fine_step=5, figsize=(6, 6)):
    """
    Klassisches Schmidt-Netz (Lambert equal area)
    mit Gitterlinien in grid_step-Grad-Abständen.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Randkreis
    circle = plt.Circle((0, 0), 1.0, edgecolor='black',
                        facecolor='none', linewidth=1.0)
    ax.add_artist(circle)

    # ===== 1) Breitenkreise (φ = konstant) =====
    lam_vals = np.arange(0, 180 + fine_step, fine_step)  # 0 .. 180°
    for phi_deg in range(-80, 81, grid_step):  # -80, -70, ..., +80
        x, y = lambert_schmidt_xy(phi_deg, lam_vals)
        ax.plot(x, y, color='0.8', linewidth=0.5)

    # ===== 2) Längenkreise (λ = konstant) =====
    phi_vals = np.arange(-90, 90 + fine_step, fine_step)  # -90 .. +90°
    for lam_deg in range(10, 171, grid_step):  # 10, 20, ..., 170
        x, y = lambert_schmidt_xy(phi_vals, lam_deg)
        ax.plot(x, y, color='0.8', linewidth=0.5)

    # kleines Fadenkreuz im Zentrum
    ax.plot([-0.02, 0.02], [0, 0], color='black', linewidth=1.0)
    ax.plot([0, 0], [-0.02, 0.02], color='black', linewidth=1.0)

    # Himmelsrichtungen (etwas weiter außen als die Gradzahlen)
    ax.text(0, 1.16, "N", ha='center', va='bottom', fontsize=10)
    ax.text(1.16, 0, "E", ha='left',   va='center', fontsize=10)
    ax.text(0, -1.16, "S", ha='center', va='top',   fontsize=10)
    ax.text(-1.16, 0, "W", ha='right',  va='center', fontsize=10)

    # ===== OPTIONALE Gradbeschriftung am Außenkreis (alle 10°) =====
    # -> Wenn du die Zahlen nicht willst, diese Schleife einfach auskommentieren.
    for az in range(0, 360, grid_step):  # 0°, 10°, 20°, ... 350°
        radius = 1.075    # leicht außerhalb des Randkreises
        angle = np.deg2rad(az)
        x = radius * np.sin(angle)
        y = radius * np.cos(angle)
        ax.text(x, y, f"{az}°", ha='center', va='center', fontsize=7)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis('off')

    return fig, ax


# -------------------------------------------------------
# 2. Geometrie: Trend/Plunge -> 3D-Vektor -> Schmidt-Projektion
# -------------------------------------------------------

def trend_plunge_to_xyz(trend_deg, plunge_deg):
    """
    Konvention:
    - trend_deg  : Azimut der Linie, 0°=N, 90°=E (im Uhrzeigersinn)
    - plunge_deg : Einfallswinkel nach unten, 0°=horizontal, 90°=vertikal
    Weltkoordinaten:
    - x: Osten, y: Norden, z: oben
    """
    T = np.radians(trend_deg)
    P = np.radians(plunge_deg)

    x = np.cos(P) * np.sin(T)   # Osten
    y = np.cos(P) * np.cos(T)   # Norden
    z = -np.sin(P)              # nach unten

    return x, y, z


def project_trend_plunge(trend_deg, plunge_deg):
    """
    Lambert-Gleichflächenprojektion der unteren Hemisphäre
    (klassisches Schmidt, Radius = 1)
    """
    x, y, z = trend_plunge_to_xyz(trend_deg, plunge_deg)

    # Formel: (X, Y) = (x, y) / sqrt(1 - z), z <= 0
    denom = np.sqrt(1.0 - z)
    # Numerische Sicherheit
    denom = np.where(denom == 0, 1e-9, denom)

    X = x / denom
    Y = y / denom

    return X, Y

# -------------------------------------------------------
# 3. Flächen + Pole + Lineare in die Projektion bringen
# -------------------------------------------------------

def plane_pole_from_dipdir(dipdir_deg, dip_deg):
    """
    Aus Diprichtung + Einfallswinkel eine Pol-Linie (Trend/Plunge)
    bestimmen.

    Annahme:
    - dipdir_deg : Einfallsrichtung (Dip Direction), 0°=N, 90°=E
    - dip_deg    : Einfallswinkel der Fläche (Dip), 0°=horizontal, 90°=vertikal

    Pol:
    - Trend_pole  = dipdir + 180° (zeigt "aufwärts" aus der Fläche)
    - Plunge_pole = 90° - dip
    """
    trend_pole = (dipdir_deg + 180.0) % 360.0
    plunge_pole = 90.0 - dip_deg
    return trend_pole, plunge_pole


def great_circle_from_plane(dipdir_deg, dip_deg, n_points=361):
    """
    Berechnet den Großkreis (Flächenspur) der Ebene
    mit Diprichtung / Dip als Punktwolke und projiziert
    NUR den Teil der unteren Halbkugel (ohne störende Gerade).
    """
    # Pol der Fläche (als Richtung)
    trend_pole, plunge_pole = plane_pole_from_dipdir(dipdir_deg, dip_deg)
    nx, ny, nz = trend_plunge_to_xyz(trend_pole, plunge_pole)
    n = np.array([nx, ny, nz])

    # Basisvektoren in der Ebene (orthogonal zu n)
    k_up = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(n, k_up)) < 0.9:
        a = np.cross(n, k_up)
    else:
        a = np.cross(n, np.array([1.0, 0.0, 0.0]))

    u = a / np.linalg.norm(a)
    w = np.cross(n, u)
    w = w / np.linalg.norm(w)

    # Punkte auf dem Schnittkreis: v(t) = u*cos(t) + w*sin(t)
    t = np.linspace(0, 2*np.pi, n_points)
    V = np.outer(np.cos(t), u) + np.outer(np.sin(t), w)

    x, y, z = V[:, 0], V[:, 1], V[:, 2]

    # Projektion (die gleiche wie bei project_trend_plunge)
    denom = np.sqrt(1.0 - z)
    denom[denom == 0] = 1e-9
    X = x / denom
    Y = y / denom

    # Obere Halbkugel NICHT löschen, sondern mit NaN "unterbrechen"
    mask = z <= 1e-9    # untere Halbkugel
    X[~mask] = np.nan
    Y[~mask] = np.nan

    return X, Y

def plot_plane_and_pole(ax, dipdir_deg, dip_deg,
                        plane_kwargs=None, pole_kwargs=None,
                        label_plane=None, label_pole=None):
    """
    Zeichnet:
    - Großkreis der Fläche
    - Pol der Fläche (Kreis-Symbol)
    """
    if plane_kwargs is None:
        plane_kwargs = dict(color='C0', linewidth=0.8, alpha=0.8)
    if pole_kwargs is None:
        pole_kwargs = dict(marker='o', color='C0', markersize=5,
                           linestyle='none')

    # Fläche (Großkreis)
    Xgc, Ygc = great_circle_from_plane(dipdir_deg, dip_deg)
    if label_plane is not None:
        ax.plot(Xgc, Ygc, label=label_plane, **plane_kwargs)
    else:
        ax.plot(Xgc, Ygc, **plane_kwargs)

    # Pol
    trend_pole, plunge_pole = plane_pole_from_dipdir(dipdir_deg, dip_deg)
    Xp, Yp = project_trend_plunge(trend_pole, plunge_pole)
    if label_pole is not None:
        ax.plot(Xp, Yp, label=label_pole, **pole_kwargs)
    else:
        ax.plot(Xp, Yp, **pole_kwargs)


def plot_linear(ax, trend_deg, plunge_deg, line_kwargs=None, label=None):
    """
    Zeichnet eine lineare Struktur (Trend/Plunge) als Kreuzsymbol.
    """
    if line_kwargs is None:
        line_kwargs = dict(marker='x', color='C3', markersize=6,
                           linestyle='none')

    X, Y = project_trend_plunge(trend_deg, plunge_deg)
    if label is not None:
        ax.plot(X, Y, label=label, **line_kwargs)
    else:
        ax.plot(X, Y, **line_kwargs)


# -------------------------------------------------------
# 4. Tabelle (CSV / Excel) einlesen und alles plotten
# -------------------------------------------------------

def load_orientation_table(path):
    """
    Liest eine Tabelle mit 3 Spalten:
    1. type  : 'plane' oder 'linear'
    2. dir   : Einfallsrichtung / Azimut (Dip Direction bzw. Trend)
    3. angle : Einfallswinkel / Plunge

    CSV oder Excel werden je nach Endung automatisch erkannt.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.csv', '.txt']:
        df = pd.read_csv(path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unbekanntes Dateiformat: {ext}")
    return df



def plot_from_table(path, grid_step=10, figsize=(6, 6)):
    """
    Kompletter Ablauf:
    - Schmidt-Netz zeichnen
    - Tabelle einlesen
    - 'plane' -> Fläche + Pol (Kreis)
    - 'linear' -> Linie (Kreuz)
    """
    df = load_orientation_table(path)

    fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    fig.suptitle(f"Schmidt-Netz – Daten aus '{os.path.basename(path)}'",
                 y=0.98)

    plane_done = False
    pole_done = False
    linear_done = False

    for _, row in df.iterrows():
        kind = str(row.iloc[0]).strip().lower()
        direction = float(row.iloc[1])
        angle = float(row.iloc[2])

        if kind == "plane":
            # Diprichtung + Dip
            label_plane = "Fläche" if not plane_done else None
            label_pole = "Pol" if not pole_done else None

            plot_plane_and_pole(ax,
                                dipdir_deg=direction,
                                dip_deg=angle,
                                label_plane=label_plane,
                                label_pole=label_pole)
            plane_done = True
            pole_done = True

        elif kind == "linear":
            # Trend + Plunge
            label_lin = "Linear" if not linear_done else None
            plot_linear(ax,
                        trend_deg=direction,
                        plunge_deg=angle,
                        label=label_lin)
            linear_done = True

        else:
            print(f"Warnung: unbekannter Typ '{kind}' – Zeile übersprungen.")

    # Legende außerhalb platzieren
    if plane_done or linear_done:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    return fig, ax

def kde_on_disc(xs, ys, grid_n=200, sigma=0.08, threshold=0.05):
    """
    Berechnet eine einfache 2D-Gauß-KDE im Einheitskreis.

    xs, ys   : Arrays der projizierten Punkte (X, Y), |r| <= 1
    grid_n   : Rastergröße (grid_n x grid_n)
    sigma    : "Breite" der Gausskerne (in Einheitskreis-Koordinaten)
    threshold: alles unterhalb dieses Werts wird ausgeblendet (NaN)
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if xs.size == 0:
        return None, None, None

    # Raster im Quadrat [-1,1] x [-1,1]
    x = np.linspace(-1, 1, grid_n)
    y = np.linspace(-1, 1, grid_n)
    X, Y = np.meshgrid(x, y, indexing="xy")

    D = np.zeros_like(X, dtype=float)

    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    # Für jeden Punkt einen Gauß hinzufügen
    for xi, yi in zip(xs, ys):
        D += np.exp(-((X - xi)**2 + (Y - yi)**2) * inv2s2)

    # außerhalb des Einheitskreises ausmaskieren
    mask = X**2 + Y**2 <= 1.0
    D[~mask] = np.nan

    # Falls wirklich gar nichts da ist
    if np.all(np.isnan(D)) or np.nanmax(D) == 0:
        return None, None, None

    # Normierung auf 0..1
    D /= np.nanmax(D)

    # schwache Bereiche ausblenden, damit nicht alles hell eingefärbt ist
    D[D < threshold] = np.nan

    return D, x, y

def find_clusters_from_density(D, x_vec, y_vec, min_pixels=10):
    """
    Findet zusammenhängende Dichte-Flecken in der KDE-Karte D.

    D      : 2D-Array (grid_n x grid_n), NaN = kein Cluster (weiß).
    x_vec  : 1D-Array der x-Koordinaten (Länge = grid_n)
    y_vec  : 1D-Array der y-Koordinaten (Länge = grid_n)
    min_pixels : minimale Anzahl Rasterzellen, die ein Cluster
                 haben muss (kleinere werden als Rauschen verworfen).

    Rückgabe:
    - labels_grid: 2D-Array gleicher Form wie D, Cluster-ID (0..k-1) oder -1
    - centers_xy : Array der Clusterzentren als (x, y)
    """
    if D is None:
        return None, np.zeros((0, 2))

    H = np.array(D)  # Kopie
    nrows, ncols = H.shape

    labels_grid = np.full((nrows, ncols), -1, dtype=int)
    current_label = 0
    centers_xy = []

    # 8er-Nachbarschaft
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(H[i, j]):
                continue
            if labels_grid[i, j] != -1:
                continue

            # Neuer Cluster: Floodfill / BFS
            stack = [(i, j)]
            labels_grid[i, j] = current_label
            pixels = [(i, j)]

            while stack:
                ci, cj = stack.pop()
                for di, dj in neighbors:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nrows and 0 <= nj < ncols:
                        if not np.isnan(H[ni, nj]) and labels_grid[ni, nj] == -1:
                            labels_grid[ni, nj] = current_label
                            stack.append((ni, nj))
                            pixels.append((ni, nj))

            # Clustergröße prüfen
            if len(pixels) < min_pixels:
                # zu klein -> als Rauschen verwerfen
                for pi, pj in pixels:
                    labels_grid[pi, pj] = -1
            else:
                # Maximum der Dichte als Zentrum
                vals = np.array([H[pi, pj] for (pi, pj) in pixels])
                imax = int(np.nanargmax(vals))
                ci, cj = pixels[imax]

                # Achtung: Zeile i -> y, Spalte j -> x
                xc = x_vec[cj]
                yc = y_vec[ci]
                centers_xy.append((xc, yc))

                current_label += 1

    # Labels auf 0..k-1 normalisieren
    unique = sorted([lab for lab in np.unique(labels_grid) if lab != -1])
    remap = {old: new for new, old in enumerate(unique)}
    labels_grid_remap = np.full_like(labels_grid, -1)
    for old, new in remap.items():
        labels_grid_remap[labels_grid == old] = new

    centers_xy = np.array(centers_xy)

    return labels_grid_remap, centers_xy

def plot_density_from_table(path,
                            grid_step=10,
                            grid_n=200,
                            sigma_planes=0.08,
                            sigma_linear=0.08,
                            threshold=0.05,
                            figsize=(6, 6),
                            save_path="schmidt_density.png",
                            dpi=300):
    """
    Liest eine Tabelle (type, dir, angle) und erstellt:
    - Dichte der Flächenpole (plane) in weiß->rot
    - Dichte der Lineare (linear) in weiß->blau
    ohne Großkreise der Flächen.

    Nutzt eine einfache Gauß-KDE im Einheitskreis.

    Die Dichten werden auf ihr jeweiliges Maximum normiert (0..1).

    Colorbars:
    - falls nur Flächen vorhanden:
        vertikale rote Skala rechts neben der Abbildung:
        "Normierte Orientierungdichte der Flächenpole (0–1)"
    - falls nur Lineare vorhanden:
        horizontale blaue Skala unter der Abbildung:
        "Normierte Orientierungdichte der Lineare (0–1)"
    - falls beides vorhanden:
        vertikale rote Skala rechts (Flächenpole)
        + horizontale blaue Skala unten (Lineare).
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    # Tabelle einlesen (type, dir, angle etc.)
    df = load_orientation_table(path)

    # Punkte sammeln
    X_planes, Y_planes = [], []
    X_lin, Y_lin = [], []

    for _, row in df.iterrows():
        kind = str(row.iloc[0]).strip().lower()
        direction = float(row.iloc[1])
        angle = float(row.iloc[2])

        if kind == "plane":
            # Diprichtung/Dip -> Pol (Trend/Plunge)
            trend_pole, plunge_pole = plane_pole_from_dipdir(direction, angle)
            Xp, Yp = project_trend_plunge(trend_pole, plunge_pole)
            X_planes.append(Xp)
            Y_planes.append(Yp)

        elif kind == "linear":
            # Trend/Plunge direkt
            Xl, Yl = project_trend_plunge(direction, angle)
            X_lin.append(Xl)
            Y_lin.append(Yl)

    X_planes = np.array(X_planes)
    Y_planes = np.array(Y_planes)
    X_lin = np.array(X_lin)
    Y_lin = np.array(Y_lin)

    # Schmidt-Netz als Basis
    fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    fig.suptitle(f"Dichteplots – {os.path.basename(path)}", y=0.98)

    # --- Dichte für Flächenpole ---
    Dp, xp, yp = kde_on_disc(X_planes, Y_planes,
                             grid_n=grid_n,
                             sigma=sigma_planes,
                             threshold=threshold)

    # --- Dichte für Lineare ---
    Dl, xl, yl = kde_on_disc(X_lin, Y_lin,
                             grid_n=grid_n,
                             sigma=sigma_linear,
                             threshold=threshold)

    # Extents (nehmen wir einfach vom jeweiligen Raster)
    extent_p = None
    extent_l = None
    if Dp is not None:
        extent_p = [xp[0], xp[-1], yp[0], yp[-1]]
    if Dl is not None:
        extent_l = [xl[0], xl[-1], yl[0], yl[-1]]

    # Colormaps: weiße Bereiche = keine Dichte
    cmap_red = plt.cm.Reds.copy()
    cmap_red.set_bad("white")

    cmap_blue = plt.cm.Blues.copy()
    cmap_blue.set_bad("white")

    # Normierung 0..1 für beide Karten
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # zuerst Flächenpole (rot)
    if Dp is not None and extent_p is not None:
        im_planes = ax.imshow(
            Dp,
            extent=extent_p,
            origin="lower",
            cmap=cmap_red,
            norm=norm,
            interpolation="bilinear",
            alpha=0.9
        )
    else:
        im_planes = None

    # dann Lineare (blau)
    if Dl is not None and extent_l is not None:
        im_lin = ax.imshow(
            Dl,
            extent=extent_l,
            origin="lower",
            cmap=cmap_blue,
            norm=norm,
            interpolation="bilinear",
            alpha=0.9
        )
    else:
        im_lin = None

    # Randkreis oben drüber (damit er sichtbar bleibt)
    circle = plt.Circle((0, 0), 1.0, edgecolor='black',
                        facecolor='none', linewidth=1.0)
    ax.add_artist(circle)

    # ---------------------------------------------------
    # Colorbars anlegen
    # ---------------------------------------------------
    has_planes = im_planes is not None
    has_linears = im_lin is not None

    if has_planes or has_linears:
        pos = ax.get_position()
        cbar_height = 0.03
        cbar_pad = 0.05

        # 1) Flächenpole: vertikal rechts
        if has_planes:
            # [x0, y0, width, height]
            cax_p = fig.add_axes([
                pos.x1 + 0.02,     # rechts neben der Achse
                pos.y0,
                0.03,
                pos.height
            ])
            sm_p = ScalarMappable(norm=norm, cmap=cmap_red)
            sm_p.set_array([])
            cbar_p = fig.colorbar(sm_p, cax=cax_p, orientation="vertical")
            cbar_p.set_label("Dichte der Flächenpole")

        # 2) Lineare: horizontal unterhalb der Hauptachse
        if has_linears:
            cax_l = fig.add_axes([
                pos.x0 + 0.1 * pos.width,
                pos.y0 - cbar_pad - cbar_height,
                pos.width * 0.8,
                cbar_height
            ])
            sm_l = ScalarMappable(norm=norm, cmap=cmap_blue)
            sm_l.set_array([])
            cbar_l = fig.colorbar(sm_l, cax=cax_l, orientation="horizontal")
            cbar_l.set_label("Dichte der Lineare")

    # Keine zusätzliche Legende mehr nötig – die Colorbars erklären alles

    # Hochauflösend speichern
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


# -------------------------------------------------------
# Farb-Helfer
# -------------------------------------------------------
def lighten_color(color, amount=0.5):
    """
    Farbe aufhellen (Mischung mit Weiß).
    amount: 0 = original, 1 = weiß
    """
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple((1 - amount) * c + amount * white)


BASE_COLORS = ['tab:red', 'tab:blue', 'tab:green',
               'tab:purple', 'tab:orange', 'tab:brown']

def plot_bicolor_cross(ax, X, Y, color1, color2, size=0.03, linewidth=2.5):
    """
    Zeichnet ein zweifarbiges Kreuz ('X') um den Punkt (X,Y).

    color1: Farbe der einen Diagonale
    color2: Farbe der anderen Diagonale
    size  : halbe Länge der Kreuzarme
    """
    # Diagonale 1
    ax.plot([X - size, X + size],
            [Y - size, Y + size],
            color=color1,
            linewidth=linewidth)

    # Diagonale 2
    ax.plot([X - size, X + size],
            [Y + size, Y - size],
            color=color2,
            linewidth=linewidth)

def plot_bicolor_square(ax, X, Y, color1, color2,
                        size=0.03, edgecolor='k', linewidth=1.0):
    """
    Zeichnet ein zweifarbiges Quadrat um den Punkt (X,Y).

    color1: Farbe eines Dreiecks im Quadrat
    color2: Farbe des anderen Dreiecks
    size  : halbe Kantenlänge des Quadrats
    """
    from matplotlib.patches import Polygon, Rectangle

    # Eckpunkte des Quadrats
    x0, x1 = X - size, X + size
    y0, y1 = Y - size, Y + size

    # Zwei Dreiecke, die zusammen das Quadrat füllen
    # Dreieck 1: unten links -> unten rechts -> oben rechts
    tri1 = Polygon([[x0, y0], [x1, y0], [x1, y1]],
                   closed=True, facecolor=color1, edgecolor='none')

    # Dreieck 2: unten links -> oben rechts -> oben links
    tri2 = Polygon([[x0, y0], [x1, y1], [x0, y1]],
                   closed=True, facecolor=color2, edgecolor='none')

    # Schwarzer Rand um das Quadrat
    border = Rectangle((x0, y0),
                       width=2*size,
                       height=2*size,
                       fill=False,
                       edgecolor=edgecolor,
                       linewidth=linewidth)

    ax.add_patch(tri1)
    ax.add_patch(tri2)
    ax.add_patch(border)

# -------------------------------------------------------
# Vektor -> Trend/Plunge (zu unserem Konventionssystem passend)
# -------------------------------------------------------
def xyz_to_trend_plunge(x, y, z):
    """
    Inverse zu trend_plunge_to_xyz für z <= 0 (untere Hemisphäre).
    Gibt Trend (0°=N, 90°=E) und Plunge (0..90°) zurück.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    trend = np.degrees(np.arctan2(x, y)) % 360.0
    plunge = np.degrees(np.arcsin(-z))
    return trend, plunge
    

def cluster_planes_from_table(path,
                              grid_step=10,
                              figsize=(6, 6),
                              mean_linewidth=2.5,
                              mean_markersize=10,
                              save_full="schmidt_clusters_full.png",
                              save_means="schmidt_clusters_means.png",
                              save_csv="plane_clusters_means.csv",
                              dpi=300,
                              # Parameter für die Dichte-basierten Cluster
                              grid_n=200,
                              sigma=0.07,
                              threshold=0.03,
                              min_cluster_pixels=10,
                              # NEU: minimale Anzahl Pole pro Cluster
                              min_poles_per_cluster=3):
    """
    Dichte-basierte Clusterung der Flächenpole:

    - Cluster = zusammenhängende Dichteflecken (KDE > threshold),
      getrennt durch weiße Bereiche.
    - Nur Cluster mit mindestens `min_poles_per_cluster` Polen
      gelten als „echte“ Cluster.
    - Erstellt:
        (1) Plot mit allen Flächen + Clustern + Linearen
        (2) Plot mit nur gemittelten Flächen + Linearen
        (3) CSV mit den gemittelten Flächen (Diprichtung/Dip).
    """
    df = load_orientation_table(path)

    # --- 1. Flächenpole + Lineare einsammeln ---
    plane_indices = []
    plane_dipdir = []
    plane_dip = []
    pole_xyz = []
    pole_XY = []
    lin_XY = []

    for i, row in df.iterrows():
        kind = str(row.iloc[0]).strip().lower()
        direction = float(row.iloc[1])
        angle = float(row.iloc[2])

        if kind == "plane":
            plane_indices.append(i)
            plane_dipdir.append(direction)
            plane_dip.append(angle)

            # Pol der Fläche (Trend/Plunge)
            trend_pole, plunge_pole = plane_pole_from_dipdir(direction, angle)
            x, y, z = trend_plunge_to_xyz(trend_pole, plunge_pole)
            pole_xyz.append([x, y, z])

            Xp, Yp = project_trend_plunge(trend_pole, plunge_pole)
            pole_XY.append([Xp, Yp])

        elif kind == "linear":
            Xl, Yl = project_trend_plunge(direction, angle)
            lin_XY.append([Xl, Yl])

    pole_xyz = np.array(pole_xyz)
    pole_XY = np.array(pole_XY)
    plane_dipdir = np.array(plane_dipdir)
    plane_dip = np.array(plane_dip)
    lin_XY = np.array(lin_XY)

    n_poles = pole_xyz.shape[0]
    if n_poles == 0:
        print("Keine Flächen (plane) in der Tabelle gefunden.")
        return None, None, None

    # --- 2. Dichtekarte der Flächenpole berechnen ---
    D, xgrid, ygrid = kde_on_disc(
        pole_XY[:, 0], pole_XY[:, 1],
        grid_n=grid_n,
        sigma=sigma,
        threshold=threshold
    )

    if D is None:
        print("Warnung: KDE ergab keine zusammenhängenden Dichtebereiche – alle Pole in einem Cluster.")
        labels_poles = np.zeros(n_poles, dtype=int)
        centers_xy = np.zeros((1, 2))
    else:
        # --- 3. Clusterflecken in der Dichtekarte finden ---
        labels_grid, centers_xy = find_clusters_from_density(
            D, xgrid, ygrid, min_pixels=min_cluster_pixels
        )

        if centers_xy.shape[0] == 0:
            print("Warnung: keine Cluster in der Dichtekarte gefunden – alle Pole in einem Cluster.")
            labels_poles = np.zeros(n_poles, dtype=int)
            centers_xy = np.zeros((1, 2))
        else:
            # --- 4. Jedem Pol den nächstgelegenen Dichte-Zentrum zuordnen ---
            centers = centers_xy  # (k,2)
            pts = pole_XY         # (n,2)
            diff = pts[:, None, :] - centers[None, :, :]   # (n,k,2)
            dist2 = np.sum(diff**2, axis=2)                # (n,k)
            labels_poles = np.argmin(dist2, axis=1)        # (n,)

    # Anzahl aller (dichte-basierten) Cluster
    k_all = int(np.max(labels_poles)) + 1
    cluster_ids = np.arange(k_all)

    # --- 5. Mindestanzahl an Polen pro Cluster anwenden ---
    counts = np.array([np.sum(labels_poles == j) for j in cluster_ids])
    valid_mask = counts >= min_poles_per_cluster
    valid_ids = cluster_ids[valid_mask]
    noise_ids = cluster_ids[~valid_mask]

    if len(valid_ids) == 0:
        print(f"Warnung: kein Cluster mit mindestens {min_poles_per_cluster} Polen gefunden.")
        # Wir geben keine Mittelwerte zurück
        df_means = pd.DataFrame(columns=["cluster", "n_poles", "trend_pole",
                                         "plunge_pole", "dipdir", "dip"])
        return df_means, (None, None), (None, None)

    # --- 6. Mittelrichtungen / mittlere Flächen nur für gültige Cluster ---
    mean_records = []
    for j in valid_ids:
        mask = labels_poles == j
        vcls = pole_xyz[mask]
        vmean = vcls.mean(axis=0)
        norm = np.linalg.norm(vmean)
        if norm == 0:
            vmean = np.array([0.0, 0.0, -1.0])
        else:
            vmean = vmean / norm

        mx, my, mz = vmean
        trend_pole_mean, plunge_pole_mean = xyz_to_trend_plunge(mx, my, mz)

        dipdir_mean = (trend_pole_mean + 180.0) % 360.0
        dip_mean = 90.0 - plunge_pole_mean

        mean_records.append(dict(
            cluster=int(j),
            n_poles=int(mask.sum()),
            trend_pole=trend_pole_mean,
            plunge_pole=plunge_pole_mean,
            dipdir=dipdir_mean,
            dip=dip_mean
        ))

    df_means = pd.DataFrame(mean_records)

    # --- Cluster-Labels neu durchnummerieren (1..N) --- 
    # valid_ids enthält die "guten" Cluster (ohne Ausreißer)
    valid_ids_sorted = sorted(valid_ids)
    cluster_label_map = {orig: i + 1 for i, orig in enumerate(valid_ids_sorted)}

    # neue Spalte mit "schönen" Cluster-Nummern
    df_means["cluster_label"] = df_means["cluster"].map(cluster_label_map)


    # CSV mit den gemittelten Flächen speichern
    if save_csv is not None:
        df_means.to_csv(save_csv, index=False)

    # --- 7. Farben je Cluster (gültig bunt, Ausreißer grau) ---
    valid_set = set(valid_ids)
    n_colors = len(BASE_COLORS)
    cluster_colors = {}
    color_idx = 0
    for j in cluster_ids:
        if j in valid_set:
            base = BASE_COLORS[color_idx % n_colors]
            color_idx += 1
        else:
            base = '0.5'  # grau für Ausreißer
        light = lighten_color(base, amount=0.6)
        cluster_colors[j] = dict(base=base, light=light, valid=(j in valid_set))

    # --- 8. Plot 1: alle Flächenpole + Großkreise + gemittelte Flächen + Lineare ---
    fig_full, ax_full = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    fig_full.suptitle(f"Cluster der Flächenpole – {os.path.basename(path)}",
                      y=0.98)

    # Einzel-Flächen (hell) + alle Großkreise (hell)
    for j in cluster_ids:
        cols = cluster_colors[j]
        base = cols['base']
        light = cols['light']

        mask = labels_poles == j
        idxs = np.where(mask)[0]

        # Einzelpole (kleine, helle Kreise – Ausreißer dann grau)
        for idx in idxs:
            Xp, Yp = pole_XY[idx]
            ax_full.plot(Xp, Yp,
                         marker='o',
                         markersize=4,
                         linestyle='none',
                         markerfacecolor='none',
                         markeredgecolor=light)

        # Helle Großkreise für alle Flächen in diesem Cluster
        for idx in idxs:
            dipdir = plane_dipdir[idx]
            dip = plane_dip[idx]
            Xgc, Ygc = great_circle_from_plane(dipdir, dip)
            ax_full.plot(Xgc, Ygc,
                         color=light,
                         linewidth=0.5,
                         alpha=0.6)

        # Mittlere Fläche NUR für gültige Cluster
        if not cols['valid']:
            continue

        mrec = df_means[df_means["cluster"] == j].iloc[0]
        dipdir_mean = mrec['dipdir']
        dip_mean = mrec['dip']
        trend_pole_mean = mrec['trend_pole']
        plunge_pole_mean = mrec['plunge_pole']

        # Mittlerer Großkreis fett – mit Label
        Xgc_m, Ygc_m = great_circle_from_plane(dipdir_mean, dip_mean)
        label = f"Cluster {int(j)+1} – mittlere Fläche"
        ax_full.plot(Xgc_m, Ygc_m,
                     color=base,
                     linewidth=mean_linewidth,
                     alpha=0.9,
                     label = f"Cluster {cluster_label_map[j]} – mittlere Fläche")

        # Mittlerer Pol fett
        Xm, Ym = project_trend_plunge(trend_pole_mean, plunge_pole_mean)
        ax_full.plot(Xm, Ym,
                     marker='o',
                     markersize=mean_markersize,
                     linestyle='none',
                     markerfacecolor=base,
                     markeredgecolor='k',
                     alpha=0.9)

        # Kurzer Strich am Außenkreis an mittlerer Diprichtung
        az = dipdir_mean
        angle = np.deg2rad(az)
        r1 = 1.05
        r2 = 1.15
        x1 = r1 * np.sin(angle)
        y1 = r1 * np.cos(angle)
        x2 = r2 * np.sin(angle)
        y2 = r2 * np.cos(angle)
        ax_full.plot([x1, x2], [y1, y2],
                     color=base,
                     linewidth=mean_linewidth)

    # Lineare als rote Kreuze
    if lin_XY.size > 0:
        ax_full.plot(lin_XY[:, 0], lin_XY[:, 1],
                     marker='x',
                     linestyle='none',
                     color='red',
                     markersize=6,
                     label="Lineare")

    # Legende nur, wenn etwas gelabelt ist
    handles, labels = ax_full.get_legend_handles_labels()
    if handles:
        ax_full.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    if save_full is not None:
        fig_full.savefig(save_full, dpi=dpi, bbox_inches="tight")

    # --- 9. Plot 2: nur gemittelte Flächen + Großkreise + Lineare ---
    fig_means, ax_means = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    fig_means.suptitle(f"Gemittelte Flächen + Lineare – {os.path.basename(path)}",
                       y=0.98)

    for j in valid_ids:
        cols = cluster_colors[j]
        base = cols['base']

        mrec = df_means[df_means["cluster"] == j].iloc[0]
        dipdir_mean = mrec['dipdir']
        dip_mean = mrec['dip']
        trend_pole_mean = mrec['trend_pole']
        plunge_pole_mean = mrec['plunge_pole']

        # Großkreis
        Xgc_m, Ygc_m = great_circle_from_plane(dipdir_mean, dip_mean)
        label = f"Cluster {int(j)+1} – mittlere Fläche"
        ax_means.plot(Xgc_m, Ygc_m,
                      color=base,
                      linewidth=mean_linewidth,
                      alpha=0.9,
                      label = f"Cluster {cluster_label_map[j]} – mittlere Fläche")

        # Pol
        Xm, Ym = project_trend_plunge(trend_pole_mean, plunge_pole_mean)
        ax_means.plot(Xm, Ym,
                      marker='o',
                      markersize=mean_markersize,
                      linestyle='none',
                      markerfacecolor=base,
                      markeredgecolor='k',
                      alpha=0.9)

        # Strich am Außenkreis
        az = dipdir_mean
        angle = np.deg2rad(az)
        r1 = 1.05
        r2 = 1.15
        x1 = r1 * np.sin(angle)
        y1 = r1 * np.cos(angle)
        x2 = r2 * np.sin(angle)
        y2 = r2 * np.cos(angle)
        ax_means.plot([x1, x2], [y1, y2],
                      color=base,
                      linewidth=mean_linewidth)

    # Lineare wieder als rote Kreuze
    if lin_XY.size > 0:
        ax_means.plot(lin_XY[:, 0], lin_XY[:, 1],
                      marker='x',
                      linestyle='none',
                      color='red',
                      markersize=6,
                      label="Lineare")

    handles, labels = ax_means.get_legend_handles_labels()
    if handles:
        ax_means.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    if save_means is not None:
        fig_means.savefig(save_means, dpi=dpi, bbox_inches="tight")

    return df_means, (fig_full, ax_full), (fig_means, ax_means)

def intersections_from_means_csv(path_means,
                                 grid_step=10,
                                 figsize=(6, 6),
                                 save_csv="plane_intersections_means.csv",
                                 save_fig="plane_intersections_means.png",
                                 dpi=300,
                                 cross_size=0.03,
                                 cross_linewidth=2.5):
    """
    Liest eine CSV mit gemittelten Flächen (clusters_means_density.csv)
    und berechnet für alle Flächenpaare:

    - Schnittlineation (Trend/Plunge)
    - Winkel zwischen den Flächen (0..90°)
    - schreibt alles in eine neue CSV
    - erzeugt eine Abbildung mit:
        * Großkreisen der gemittelten Flächen (farbig je Cluster)
        * Polen der gemittelten Flächen
        * Schnittlineationen als zweifarbige Kreuze

    Rückgabe:
      df_int, (fig, ax)
    """
    df_means = pd.read_csv(path_means)

    # Erwartete Spalten: cluster, n_poles, trend_pole, plunge_pole, dipdir, dip
    if not {"cluster", "trend_pole", "plunge_pole", "dipdir", "dip"}.issubset(df_means.columns):
        raise ValueError("CSV scheint nicht im erwarteten Format zu sein "
                         "(benötigt: cluster, trend_pole, plunge_pole, dipdir, dip).")

    # --- 1. Farben pro Cluster zuweisen ---
    unique_clusters = sorted(df_means["cluster"].unique())
    color_map = {}
    for idx, cid in enumerate(unique_clusters):
        base = BASE_COLORS[idx % len(BASE_COLORS)]
        light = lighten_color(base, amount=0.5)
        color_map[cid] = dict(base=base, light=light)
    
    # Mapping von interner Cluster-ID -> „schöner“ Cluster-Nummer (1..N)
    # Falls df_means bereits eine Spalte 'cluster_label' hat, nutzen wir die.
    if "cluster_label" in df_means.columns:
        id_to_label = {
            int(row["cluster"]): int(row["cluster_label"])
            for _, row in df_means.iterrows()
        }
    else:
        # Fallback: interne ID direkt anzeigen
        id_to_label = {int(cid): int(cid) for cid in unique_clusters}


    # --- 2. Alle Flächenpaare durchgehen und Schnittlineation berechnen ---
    records = []
    n = len(df_means)

    for i in range(n):
        for j in range(i + 1, n):
            row_i = df_means.iloc[i]
            row_j = df_means.iloc[j]

            cid_i = int(row_i["cluster"])
            cid_j = int(row_j["cluster"])

            dipdir_i = float(row_i["dipdir"])
            dip_i = float(row_i["dip"])
            dipdir_j = float(row_j["dipdir"])
            dip_j = float(row_j["dip"])

            # Normalenvektoren aus den Polen
            tpi, ppi = float(row_i["trend_pole"]), float(row_i["plunge_pole"])
            tpj, ppj = float(row_j["trend_pole"]), float(row_j["plunge_pole"])

            n1x, n1y, n1z = trend_plunge_to_xyz(tpi, ppi)
            n2x, n2y, n2z = trend_plunge_to_xyz(tpj, ppj)

            n1 = np.array([n1x, n1y, n1z])
            n2 = np.array([n2x, n2y, n2z])

            # Schnittlinie = Kreuzprodukt der Normalen
            l = np.cross(n1, n2)
            norm_l = np.linalg.norm(l)
            if norm_l < 1e-6:
                # Fast parallel – keine definierte Schnittlineation
                trend_int = np.nan
                plunge_int = np.nan
                angle_planes = 0.0
            else:
                l = l / norm_l
                # auf untere Hemisphäre bringen
                if l[2] > 0:
                    l = -l

                lx, ly, lz = l
                trend_int, plunge_int = xyz_to_trend_plunge(lx, ly, lz)

                # Winkel zwischen den Flächen: Winkel zwischen Normalen
                dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
                theta = np.degrees(np.arccos(dot))  # 0..180
                angle_planes = min(theta, 180.0 - theta)  # 0..90

            records.append(dict(
                cluster_i=cid_i,
                cluster_j=cid_j,
                dipdir_i=dipdir_i,
                dip_i=dip_i,
                dipdir_j=dipdir_j,
                dip_j=dip_j,
                trend_int=trend_int,
                plunge_int=plunge_int,
                angle_planes_deg=angle_planes
            ))

    df_int = pd.DataFrame(records)

    # CSV speichern
    if save_csv is not None:
        df_int.to_csv(save_csv, index=False)

    # --- 3. Abbildung: Flächen + Pole + Schnittlineationen ---
    fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    fig.suptitle(f"Schnittlineationen der gemittelten Flächen – {os.path.basename(path_means)}",
                 y=0.98)

    # 3a) Gemittelte Flächen + Pole
    for _, row in df_means.iterrows():
        cid = int(row["cluster"])
        colors = color_map[cid]
        base = colors["base"]
        light = colors["light"]

        dipdir = float(row["dipdir"])
        dip = float(row["dip"])
        tpol = float(row["trend_pole"])
        ppol = float(row["plunge_pole"])

        # Großkreis
        Xgc, Ygc = great_circle_from_plane(dipdir, dip)
        ax.plot(Xgc, Ygc,
                color=light,
                linewidth=2.0,
                alpha=0.9)

        # Pol
        Xp, Yp = project_trend_plunge(tpol, ppol)
        ax.plot(Xp, Yp,
                marker='o',
                markersize=10,
                linestyle='none',
                markerfacecolor=base,
                markeredgecolor='k',
                alpha=0.9)

    # 3b) Schnittlineationen als zweifarbige Kreuze
    for _, row in df_int.iterrows():
        if np.isnan(row["trend_int"]):
            continue  # parallele Flächen

        cid_i = int(row["cluster_i"])
        cid_j = int(row["cluster_j"])

        c1 = color_map[cid_i]["base"]
        c2 = color_map[cid_j]["base"]

        trend_int = float(row["trend_int"])
        plunge_int = float(row["plunge_int"])

        Xint, Yint = project_trend_plunge(trend_int, plunge_int)

        plot_bicolor_cross(ax, Xint, Yint, c1, c2,
                           size=cross_size,
                           linewidth=cross_linewidth)

    # ---------------------------------------------------
    # Legende: Clusterfarben + Symbolerklärung
    # ---------------------------------------------------
    legend_handles = []
    
    # 1) Clusterfarben (Kreis-Symbol, gefüllt)
    for cid in unique_clusters:
        base = color_map[cid]["base"]
        label_num = id_to_label.get(int(cid), cid)  # 1,2,3,... statt 0,1,2,...
        h = plt.Line2D(
            [], [],
            marker='o', linestyle='none',
            markersize=8,
            markerfacecolor=base,
            markeredgecolor='k',
            label=f"Cluster {label_num}"
        )
        legend_handles.append(h)
    
    # 2) Symbolerklärung (schwarz, unabhängig von Cluster)
    line_handle = plt.Line2D(
        [0], [0],
        color='k',
        linewidth=2,
        label="Großkreis (mittlere Fläche)"
    )
    
    circle_handle = plt.Line2D(
        [0], [0],
        marker='o',
        linestyle='none',
        markersize=8,
        markerfacecolor='none',
        markeredgecolor='k',
        label="Pol (mittlere Fläche)"
    )
    
    cross_handle = plt.Line2D(
        [0], [0],
        marker='x',
        linestyle='none',
        markersize=8,
        color='k',
        label="Schnittlineation"
    )
    
    legend_handles.extend([line_handle, circle_handle, cross_handle])
    
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.0)
    )


    # Speichern
    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    return df_int, (fig, ax)

def add_intersection_poles_with_plot(path_means,
                                     path_intersections,
                                     grid_step=10,
                                     figsize=(6, 6),
                                     save_csv="plane_intersections_means_with_poles.csv",
                                     save_fig="plane_intersections_full.png",
                                     dpi=300,
                                     cross_size=0.03,
                                     cross_linewidth=2.5,
                                     square_size=0.03,
                                     square_linewidth=1.5):
    """
    Kombinierte Darstellung:

    - Liest:
        * path_means = clusters_means_density.csv
        * path_intersections = plane_intersections_means.csv
    - Berechnet für jede Schnittlineation eine "Pol-Lineation":
        trend_int_pole  = (trend_int + 180°) mod 360
        plunge_int_pole = 90° - plunge_int
    - Fügt hinzu:
        * trend_int_pole, plunge_int_pole
        * X_int_pole, Y_int_pole (projiziert im Schmidt-Netz)
    - Speichert alles in save_csv.
    - Zeichnet eine Abbildung mit:
        * gemittelten Flächen (Großkreise + fette Pole, farbig pro Cluster)
        * Schnittlineationen als zweifarbige Kreuze
        * "Pol-Lineationen" als zweifarbige Quadrate
    """
    # --- 1. Einlesen der Daten ---
    df_means = pd.read_csv(path_means)
    df_int = pd.read_csv(path_intersections)

    req_means = {"cluster", "trend_pole", "plunge_pole", "dipdir", "dip"}
    if not req_means.issubset(df_means.columns):
        raise ValueError(f"'{path_means}' hat nicht alle benötigten Spalten: {req_means}")

    req_int = {"cluster_i", "cluster_j", "trend_int", "plunge_int"}
    if not req_int.issubset(df_int.columns):
        raise ValueError(f"'{path_intersections}' hat nicht alle benötigten Spalten: {req_int}")

    # --- 2. Farben pro Cluster (wie in den anderen Plots) ---
    unique_clusters = sorted(df_means["cluster"].unique())
    color_map = {}
    for idx, cid in enumerate(unique_clusters):
        base = BASE_COLORS[idx % len(BASE_COLORS)]
        light = lighten_color(base, amount=0.5)
        color_map[int(cid)] = dict(base=base, light=light)

    # Mapping von interner Cluster-ID -> „schöner“ Cluster-Nummer (1..N)
    if "cluster_label" in df_means.columns:
        id_to_label = {
            int(row["cluster"]): int(row["cluster_label"])
            for _, row in df_means.iterrows()
        }
    else:
        # Fallback: interne ID direkt anzeigen
        id_to_label = {int(cid): int(cid) for cid in unique_clusters}


    # --- 3. "Pol-Lineationen" berechnen ---
    trend_pole_list = []
    plunge_pole_list = []
    X_pole_list = []
    Y_pole_list = []

    for _, row in df_int.iterrows():
        trend_int = row["trend_int"]
        plunge_int = row["plunge_int"]

        if np.isnan(trend_int) or np.isnan(plunge_int):
            trend_pole_list.append(np.nan)
            plunge_pole_list.append(np.nan)
            X_pole_list.append(np.nan)
            Y_pole_list.append(np.nan)
        else:
            # Deine Definition:
            trend_pole = (trend_int + 180.0) % 360.0
            plunge_pole = 90.0 - plunge_int

            trend_pole_list.append(trend_pole)
            plunge_pole_list.append(plunge_pole)

            Xp, Yp = project_trend_plunge(trend_pole, plunge_pole)
            X_pole_list.append(Xp)
            Y_pole_list.append(Yp)

    df_int["trend_int_pole"] = trend_pole_list
    df_int["plunge_int_pole"] = plunge_pole_list
    df_int["X_int_pole"] = X_pole_list
    df_int["Y_int_pole"] = Y_pole_list

    # CSV mit allen Infos speichern
    if save_csv is not None:
        df_int.to_csv(save_csv, index=False)

    # --- 4. Abbildung: Schmidt-Netz + gemittelte Flächen + Schnittlineationen + "Pole" ---
    fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    fig.suptitle(
        f"Schnittlineationen & 'Pole' – {os.path.basename(path_means)} / {os.path.basename(path_intersections)}",
        y=0.98
    )

    # 4a) Gemittelte Flächen (Großkreise + Pole)
    for _, row in df_means.iterrows():
        cid = int(row["cluster"])
        colors = color_map[cid]
        base = colors["base"]
        light = colors["light"]

        dipdir = float(row["dipdir"])
        dip = float(row["dip"])
        trend_pole = float(row["trend_pole"])
        plunge_pole = float(row["plunge_pole"])

        # Großkreis
        Xgc, Ygc = great_circle_from_plane(dipdir, dip)
        ax.plot(Xgc, Ygc,
                color=light,
                linewidth=2.0,
                alpha=0.9)

        # Pol
        Xp, Yp = project_trend_plunge(trend_pole, plunge_pole)
        ax.plot(Xp, Yp,
                marker='o',
                markersize=10,
                linestyle='none',
                markerfacecolor=base,
                markeredgecolor='k',
                alpha=0.9)

    # 4b) Schnittlineationen als zweifarbige Kreuze
    for _, row in df_int.iterrows():
        trend_int = row["trend_int"]
        plunge_int = row["plunge_int"]
        if np.isnan(trend_int) or np.isnan(plunge_int):
            continue  # parallele Flächen o.Ä.

        cid_i = int(row["cluster_i"])
        cid_j = int(row["cluster_j"])

        c1 = color_map[cid_i]["base"]
        c2 = color_map[cid_j]["base"]

        Xint, Yint = project_trend_plunge(trend_int, plunge_int)

        plot_bicolor_cross(ax, Xint, Yint,
                           color1=c1,
                           color2=c2,
                           size=cross_size,
                           linewidth=cross_linewidth)

    # 4c) "Pol-Lineationen" als zweifarbige Quadrate
    for _, row in df_int.iterrows():
        Xp = row["X_int_pole"]
        Yp = row["Y_int_pole"]
        if np.isnan(Xp) or np.isnan(Yp):
            continue

        cid_i = int(row["cluster_i"])
        cid_j = int(row["cluster_j"])

        c1 = color_map[cid_i]["base"]
        c2 = color_map[cid_j]["base"]

        plot_bicolor_square(ax, Xp, Yp,
                            color1=c1,
                            color2=c2,
                            size=square_size,
                            edgecolor='k',
                            linewidth=square_linewidth)

        # ---------------------------------------------------
        # Legende: Clusterfarben + Symbolerklärung
        # ---------------------------------------------------
        legend_handles = []
    
        # 1) Clusterfarben (Kreis-Symbol, gefüllt)
        for cid in unique_clusters:
            base = color_map[int(cid)]["base"]
            label_num = id_to_label.get(int(cid), cid)  # 1,2,3,... statt 0,1,2,...
            h = plt.Line2D(
                [], [],
                marker='o',
                linestyle='none',
                markersize=8,
                markerfacecolor=base,
                markeredgecolor='k',
                label=f"Cluster {label_num}"
            )
            legend_handles.append(h)
    
        # 2) Symbolerklärung (schwarz, unabhängig von Cluster)
        line_handle = plt.Line2D(
            [0], [0],
            color='k',
            linewidth=2,
            label="Großkreis (mittlere Fläche)"
        )
    
        circle_handle = plt.Line2D(
            [0], [0],
            marker='o',
            linestyle='none',
            markersize=8,
            markerfacecolor='none',
            markeredgecolor='k',
            label="Pol (mittlere Fläche)"
        )
    
        cross_handle = plt.Line2D(
            [0], [0],
            marker='x',
            linestyle='none',
            markersize=8,
            color='k',
            label="Schnittlineation"
        )
    
        square_handle = plt.Line2D(
            [0], [0],
            marker='s',
            linestyle='none',
            markersize=8,
            markerfacecolor='none',
            markeredgecolor='k',
            label="'Pol' der Schnittlineation"
        )
    
        legend_handles.extend([line_handle, circle_handle, cross_handle, square_handle])
    
        ax.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(1.05, 1.0)
        )


    # Speichern
    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    return df_int, (fig, ax)


################################################################################################
### Massenbewegungen Kinematische Analyse
################################################################################################

# -------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------

def circular_diff(a, b):
    """
    kleinster Winkelunterschied zwischen a und b (in Grad, [-180,180])
    """
    # Idee:
    # - Differenz a - b wird so „gewrappt“, dass sie immer im Bereich [-180, 180] liegt
    # - praktisch für Azimut-Vergleiche (z.B. „wie weit weicht dipdir von slope_dipdir ab?“)
    #   ohne Probleme bei 0°/360°-Übergängen
    return (a - b + 180.0) % 360.0 - 180.0


def az_from_xy(X, Y):
    """
    'Azimut' auf deinem Schmidt-Netz, wie am Rand beschriftet:
    0° = oben, 90° = rechts, 180° = unten, 270° = links
    """
    # arctan2(X, Y) -> Winkel im Bogenmaß, bezogen auf „Norden“ (Y-Achse),
    # dann nach Grad umrechnen und in [0, 360) bringen.
    # Diese Definition ist konsistent mit deiner Schmidt-Netz-Beschriftung.
    return (np.degrees(np.arctan2(X, Y)) + 360.0) % 360.0


def xy_to_xyz(X, Y):
    """
    Inverse der Projektion X = x / sqrt(1 - z), Y = y / sqrt(1 - z)
    (Lambert-Gleichflächenprojektion untere Hemisphäre, Radius=1).

    Gibt Einheitsvektor (x, y, z) zurück (ggf. auf untere Hemisphäre gespiegelt).
    """
    S = X**2 + Y**2          # = r^2 (Quadrat des stereographischen Radius im Netz)
    z = S - 1.0              # aus der Umkehrformel: z = r^2 - 1
    k = np.sqrt(1.0 - z)     # Faktor, der aus (X, Y) wieder (x, y) macht
    x = X * k                # zurückprojizierte x-Komponente auf der Sphäre
    y = Y * k                # zurückprojizierte y-Komponente auf der Sphäre

    # sicherheitshalber auf Länge 1 normieren
    # numerische Fehler können dazu führen, dass (x, y, z) nicht exakt Norm 1 hat
    norm = np.sqrt(x*x + y*y + z*z)
    x /= norm
    y /= norm
    z /= norm

    # auf untere Hemisphäre spiegeln
    # in der Schmidt-Projektion arbeitest du konsequent mit der unteren Halbkugel (z ≤ 0)
    # falls der Punkt aus numerischen Gründen auf der oberen Halbkugel landet (z > 0),
    # wird der Vektor einfach durch den Ursprung gespiegelt.
    if z > 0:
        x, y, z = -x, -y, -z

    return x, y, z


# -------------------------------------------------------
# Funktionen
# -------------------------------------------------------

def plot_plane_failure_zone(slope_dipdir,
                            slope_dip,
                            friction_angle,
                            dipdir_tolerance=20.0,
                            grid_step=10,
                            figsize=(6, 6),
                            save_fig=None,
                            dpi=400,
                            ax=None,
                            add_title=True,
                            add_legend=True,
                            plot_slope=True):
    """
    Zeichnet die kritische Zone für PLANE FAILURE im Schmidt-Netz
    (Pol-Darstellung der Trennflächen).

    Idee:
      - Jeder Punkt im Schmidt-Netz wird als Pol einer möglichen Trennfläche
        interpretiert.
      - Aus diesem Pol wird die Ebenenorientierung (dipdir, dip) berechnet.
      - Eine Fläche ist kritisch für Planarversagen, wenn:
          * Reibungswinkel < dip < Hangneigung
          * dipdir nicht weiter als dipdir_tolerance von der Hangrichtung
            (slope_dipdir) abweicht.
      - Alle Polpunkte, die diese Bedingungen erfüllen, bilden die „plane-failure
        zone“ – diese wird schraffiert und mit gestrichelter Linie umrandet.

    Parameter:
      slope_dipdir      : Fallrichtung der Hangfläche (Azimut, Grad, 0°=N, 90°=E)
      slope_dip         : Fallen der Hangfläche (Grad)
      friction_angle    : Reibungswinkel (φ) der Trennflächen (Grad)
      dipdir_tolerance  : maximale Abweichung der Fallrichtung der Trennfläche
                          von slope_dipdir (Faustregel: ±20°)
      grid_step         : Gitterabstand für das Schmidt-Netz (an plot_schmidt_net)
      figsize           : Figurgröße (Breite, Höhe) in Zoll
      save_fig          : Dateiname für optionales Speichern (z.B. "plane_failure.png")
      dpi               : Auflösung beim Speichern
      ax                : vorhandenes matplotlib-Axes-Objekt; falls None, wird
                          ein neues Schmidt-Netz erzeugt
      add_title         : wenn True, wird ein Titel zur Figur hinzugefügt
      add_legend        : wenn True, wird eine Legende erzeugt
      plot_slope        : wenn True, wird Hang-Großkreis + Hang-Pol gezeichnet
    """

    # Eingaben normieren / in float umwandeln
    slope_dipdir   = float(slope_dipdir) % 360.0   # Azimut in [0, 360)
    slope_dip      = float(slope_dip)
    friction_angle = float(friction_angle)
    tol            = float(dipdir_tolerance)

    # --- Figure / Axes Setup ---
    # Falls kein Axes übergeben wurde: neues Schmidt-Netz aufziehen
    if ax is None:
        fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    else:
        # Falls ax existiert: dazugehörige Figure herausholen
        fig = ax.figure

    # Optional Titel setzen
    if add_title:
        fig.suptitle("Plane Failure – kritische Zone", y=0.98)

    # Trivialfall: Reibungswinkel ≥ Hangneigung -> kein kritischer Bereich
    if friction_angle >= slope_dip:
        print("Kein kritischer Bereich: Reibungswinkel ≥ Hangneigung.")
        if plot_slope:
            # Hang-Großkreis
            Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
            ax.plot(Xgc, Ygc, color="k", linewidth=2.5)
            # Hang-Pol
            tp, pp = plane_pole_from_dipdir(slope_dipdir, slope_dip)
            Xp, Yp = project_trend_plunge(tp, pp)
            ax.plot(Xp, Yp, "ko", markersize=10)
        # optional Figur speichern und direkt zurückgeben
        if save_fig is not None:
            fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
        return fig, ax

    # --- Hangfläche (Großkreis + Pol) zeichnen ---
    if plot_slope:
        # Großkreis der Hangfläche
        Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
        ax.plot(Xgc, Ygc, color="k", linewidth=2.5)

        # Pol der Hangfläche
        trend_pole_slope, plunge_pole_slope = plane_pole_from_dipdir(
            slope_dipdir, slope_dip
        )
        Xp_slope, Yp_slope = project_trend_plunge(trend_pole_slope, plunge_pole_slope)
        ax.plot(Xp_slope, Yp_slope,
                marker="o", markersize=10,
                markerfacecolor="k", markeredgecolor="k",
                linestyle="none")

    # --- Punkt-in-Zone-Test (wie bisher) ---
    # Prüft für ein gegebenes (x, y) im Schmidt-Netz, ob der entsprechende Polpunkt
    # zu einer kritisch geneigten Fläche gehört (nach Definition oben).
    def point_in_zone(x, y):
        # außerhalb des Schmidt-Randes -> ignorieren
        if x**2 + y**2 > 1.0:
            return False

        # 2D-Punkt (Schmidt) zurück auf Kugel (Normalenvektor) projizieren
        vx, vy, vz = xy_to_xyz(x, y)

        # daraus Trend/Plunge des Pols bestimmen
        trend_pole, plunge_pole = xyz_to_trend_plunge(vx, vy, vz)

        # Pol -> Ebenenorientierung:
        # dip = 90° - plunge (Polplunge)
        dip = 90.0 - plunge_pole
        # dipdir 180° versetzt vom Pol-Trend (Pol zeigt Richtung des Aufrichtens)
        dipdir = (trend_pole - 180.0) % 360.0

        # 1) Reibungs-Bedingung: φ < dip < slope_dip
        if not (friction_angle < dip < slope_dip):
            return False

        # 2) Azimut-Bedingung: Pol/Ebene soll innerhalb des erlaubten Sektors liegen
        if abs(circular_diff(dipdir, slope_dipdir)) > tol:
            return False

        return True

    # --- horizontale Schraffur ---
    # Wir zeichnen horizontale Linien (konstantes y) und färben nur die Segmente,
    # für die point_in_zone == True ist -> ergibt eine horizontale Schraffur
    N_stripes = 40
    y_vals = np.linspace(-1.0, 1.0, N_stripes)
    for y in y_vals:
        X_line = np.linspace(-1.0, 1.0, 500)
        X_seg, Y_seg = [], []
        for x in X_line:
            if point_in_zone(x, y):
                # zusammenhängendes Segment innerhalb der Zone
                X_seg.append(x); Y_seg.append(y)
            else:
                # Segment-Ende: falls ein Segment existiert, zeichnen und leeren
                if X_seg:
                    ax.plot(X_seg, Y_seg,
                            color="red", linewidth=0.6, alpha=0.8, zorder=2)
                    X_seg, Y_seg = [], []
        # am Ende der Linie ggf. letztes Segment zeichnen
        if X_seg:
            ax.plot(X_seg, Y_seg,
                    color="red", linewidth=0.6, alpha=0.8, zorder=2)

    # --- Kontur ---
    # Hier wird ein feinmaschiges Gitter über das Netz gelegt; für jeden Gitterpunkt
    # wird point_in_zone geprüft und ein Maskenfeld aufgebaut.
    # Anschließend wird mit contour(...) die Randlinie (Kontur) der Zone gezeichnet.
    Ngrid = 400
    xs = np.linspace(-1.0, 1.0, Ngrid)
    ys = np.linspace(-1.0, 1.0, Ngrid)
    Xg, Yg = np.meshgrid(xs, ys)
    mask = np.zeros_like(Xg, dtype=float)
    for i in range(Ngrid):
        for j in range(Ngrid):
            if point_in_zone(Xg[i, j], Yg[i, j]):
                mask[i, j] = 1.0

    ax.contour(
        Xg, Yg, mask,
        levels=[0.5],           # 0.5 als Schwelle zwischen 0 und 1
        colors="red",
        linestyles="--",
        linewidths=1.5,
        zorder=3
    )

    # --- Legende ---
    if add_legend:
        # Hang-Großkreis
        slope_line = Line2D([0], [0],
                            color="black", linewidth=2.5,
                            label="Hangfläche (Großkreis)")
        # Hang-Pol
        slope_pole = Line2D([0], [0],
                            marker="o", linestyle="none",
                            markerfacecolor="black", markeredgecolor="black",
                            markersize=8, label="Pol der Hangfläche")
        # Symbol für die plane-failure-Zone (horizontale Schraffur, gestrichelter Rand)
        zone_patch_legend = Patch(facecolor='none',
                                  edgecolor='red',
                                  linestyle='--',
                                  hatch='-',
                                  label="kritische Zone (plane failure)")
        ax.legend(handles=[slope_line, slope_pole, zone_patch_legend],
                  loc="upper left", bbox_to_anchor=(1.05, 1.0))

    # optional Figur speichern
    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    return fig, ax




def plot_wedge_failure_marklandsche_area(slope_dipdir,
                                         slope_dip,
                                         friction_angle=None,
                                         grid_step=10,
                                         figsize=(6, 6),
                                         save_fig=None,
                                         dpi=400,
                                         ax=None,
                                         add_title=True,
                                         add_legend=True,
                                         plot_slope=True):
    """
    Zeichnet die „Marklandsche Fläche“ / Critical Pole Vector Zone, konstruiert aus
    den „Pol-Linearen“ des Großkreises der Hangfläche.

    Geometrische Idee (so wie du sie definiert hast):
      1. Man nimmt alle Punkte auf dem Großkreis der Hangfläche.
      2. Jeder dieser Punkte wird als LINEAR (trend_L, plunge_L) interpretiert.
      3. Für jedes Linear wird ein „Pol“ berechnet:
            trend_pole_L  = (trend_L + 180°) % 360
            plunge_pole_L = 90° - plunge_L
         (Das entspricht deiner Definition von „Pol“ zu einer Linie.)
      4. Diese Polpunkte werden ins Schmidt-Netz projiziert und zu einer Zone
         verbunden (Polygon, vom Zentrum aus geschlossen).
      5. Optional wird der Pole-Friction-Cone (Reibungswinkel) als inneres Loch
         aus der Zone „herausgeschnitten“, so dass nur der Bereich außerhalb
         dieses Kegels bleibt (kritische Bereiche).
    """

    # Eingaben normieren / in float umwandeln
    slope_dipdir = float(slope_dipdir) % 360.0
    slope_dip = float(slope_dip)
    if friction_angle is not None:
        friction_angle = float(friction_angle)

    # --- Figure / Axes Setup ---
    # Falls kein Axes übergeben wurde: neues Schmidt-Netz aufziehen
    if ax is None:
        fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    else:
        # Falls ax existiert: dazugehörige Figure herausholen
        fig = ax.figure

    # Optional Titel setzen
    if add_title:
        fig.suptitle("Critical Pole Vector Zone (aus 'Pol'-Lineinen der Hangfläche)",
                     y=0.98)

    # Hang-Großkreis + Pol zeichnen (falls gewünscht)
    if plot_slope:
        # Großkreis der Hangfläche
        Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
        ax.plot(Xgc, Ygc, color="k", linewidth=2.0, label="Slope plane (Großkreis)")

        # Pol der Hangfläche
        trend_pole_slope, plunge_pole_slope = plane_pole_from_dipdir(
            slope_dipdir, slope_dip
        )
        Xp_slope, Yp_slope = project_trend_plunge(trend_pole_slope, plunge_pole_slope)
        ax.plot(Xp_slope, Yp_slope,
                marker="o", markersize=10,
                markerfacecolor="k", markeredgecolor="k",
                linestyle="none", label="Pol der Hangfläche")

    # Großkreis der Hangfläche noch einmal (für die Zonenkonstruktion)
    # -> alle Punkte werden als LINEARE interpretiert
    Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
    X_zone, Y_zone = [], []
    for Xg, Yg in zip(Xgc, Ygc):
        # Punkte außerhalb des Schmidt-Randes ignorieren
        if Xg**2 + Yg**2 > 1.0:
            continue

        # 2D (Schmidt) -> 3D-Richtungsvektor (Linear)
        vx, vy, vz = xy_to_xyz(Xg, Yg)
        trend_L, plunge_L = xyz_to_trend_plunge(vx, vy, vz)

        # „Pol“ dieses Linears nach deiner Definition:
        # Trend um 180° versetzen, Plunge in „Polplunge“ umrechnen
        trend_pole_L = (trend_L + 180.0) % 360.0
        plunge_pole_L = 90.0 - plunge_L

        # Projektion dieses Pols ins Schmidt-Netz
        Xp, Yp = project_trend_plunge(trend_pole_L, plunge_pole_L)
        X_zone.append(Xp); Y_zone.append(Yp)

    # Falls aus irgendeinem Grund keine Punkte vorliegen: nur evtl. speichern und zurück
    if len(X_zone) == 0:
        if save_fig is not None:
            fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
        return fig, ax

    X_zone = np.asarray(X_zone)
    Y_zone = np.asarray(Y_zone)

    from matplotlib.path import Path
    # Polygon der Zone:
    #   - vom Zentrum (0,0) zu den Polpunkten entlang des Großkreises,
    #     dann wieder zurück zum Zentrum -> geschlossene „Linsen“-Fläche
    verts_x = np.concatenate(([0.0], X_zone, [0.0]))
    verts_y = np.concatenate(([0.0], Y_zone, [0.0]))
    poly_path = Path(np.column_stack((verts_x, verts_y)), closed=True)

    # Pole-Friction-Cone-Radius² (falls friction_angle angegeben)
    # Idee:
    #   - Pol einer Fläche mit dip = friction_angle: plunge_pole_f = 90° - φ
    #   - dessen Projektion hat einen Radius r_f -> r_f² merken
    if friction_angle is not None:
        trend_test = 0.0
        plunge_pole_f = 90.0 - friction_angle
        Xf0, Yf0 = project_trend_plunge(trend_test, plunge_pole_f)
        r_f2 = Xf0**2 + Yf0**2
    else:
        r_f2 = 0.0

    # Test, ob ein Punkt (x, y) in der definierten Zone liegt:
    #   1) im Schmidt-Kreis
    #   2) innerhalb des Polygonpfads (poly_path)
    #   3) außerhalb des (optional) inneren Pole-Friction-Kegels
    def point_in_zone(x, y):
        # außerhalb des Schmidt-Randes
        if x*x + y*y > 1.0:
            return False
        # außerhalb des Polygons
        if not poly_path.contains_point((x, y)):
            return False
        # innerhalb des Pole-Friction-Cone -> nicht kritisch
        if friction_angle is not None and (x*x + y*y <= r_f2):
            return False
        return True

    # vertikale Schraffur (Critical Pole Vector Zone)
    # Wir nehmen vertikale Linien x = const und zeichnen nur die Segmente,
    # in denen point_in_zone == True ist.
    N_stripes = 40
    x_vals = np.linspace(-1.0, 1.0, N_stripes)
    for x in x_vals:
        y_line = np.linspace(-1.0, 1.0, 500)
        X_seg, Y_seg = [], []
        for y in y_line:
            if point_in_zone(x, y):
                X_seg.append(x); Y_seg.append(y)
            else:
                if X_seg:
                    ax.plot(X_seg, Y_seg,
                            color="red", linewidth=0.6, alpha=0.8, zorder=2)
                    X_seg, Y_seg = [], []
        if X_seg:
            ax.plot(X_seg, Y_seg,
                    color="red", linewidth=0.6, alpha=0.8, zorder=2)

    # Kontur der Zone:
    #   - feines Gitter über das Netz legen
    #   - für jeden Punkt prüfen, ob er in der Zone liegt
    #   - mit contour(...) die Randlinie (inkl. innerem Loch) zeichnen
    Ngrid = 400
    xs = np.linspace(-1.0, 1.0, Ngrid)
    ys = np.linspace(-1.0, 1.0, Ngrid)
    Xg2, Yg2 = np.meshgrid(xs, ys)
    mask = np.zeros_like(Xg2, dtype=float)
    for i in range(Ngrid):
        for j in range(Ngrid):
            if point_in_zone(Xg2[i, j], Yg2[i, j]):
                mask[i, j] = 1.0

    ax.contour(
        Xg2, Yg2, mask,
        levels=[0.5],
        colors="red",
        linestyles="--",
        linewidths=1.5,
        zorder=3
    )

    # Legende für diese Zone (optional)
    if add_legend:
        zone_patch_legend = Patch(
            facecolor='none',
            edgecolor='red',
            linestyle='--',
            hatch='|',   # in der Legende als senkrechte Schraffur symbolisiert
            label="Critical Pole Vector Zone"
        )
        handles = []
        if plot_slope:
            handles.append(Line2D([0], [0], color="black", linewidth=2.0,
                                  label="Slope plane (Großkreis)"))
            handles.append(Line2D([0], [0],
                                  marker="o", linestyle="none",
                                  markerfacecolor="black", markeredgecolor="black",
                                  markersize=8, label="Pol der Hangfläche"))
        handles.append(zone_patch_legend)
        ax.legend(handles=handles,
                  loc="upper left", bbox_to_anchor=(1.05, 1.0))

    # optional Figur speichern
    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    return fig, ax




def plot_toppling_failure_zone(slope_dipdir,
                               slope_dip,
                               friction_angle,
                               dipdir_tolerance=10.0,
                               grid_step=10,
                               figsize=(6, 6),
                               save_fig=None,
                               dpi=400,
                               ax=None,
                               add_title=True,
                               add_legend=True,
                               plot_slope=True):
    """
    Toppling-Failure-Zone im Polnetz – jetzt mit optionalem ax.

    Geometrische/physikalische Idee:
      - Jeder Punkt im Schmidt-Netz wird als POL einer möglichen Trennfläche
        interpretiert.
      - Aus diesem Pol werden dipdir und dip der Trennfläche berechnet.
      - Für Toppling gelten zwei Bedingungen:

        1) Trennflächen müssen in Gegenrichtung des Hangs einfallen:
             dipdir_opposite = (slope_dipdir + 180°) % 360
             |dipdir_J - dipdir_opposite| <= dipdir_tolerance

        2) Trennflächen müssen „steil genug“ sein:
             (90° - ψ_f) + φ_j < ψ_P

           mit:
             ψ_f = slope_dip         (Fallen des Hangs)
             φ_j = friction_angle    (Reibungswinkel der Trennfläche)
             ψ_P = dip der Trennfläche

      - Alle Polpunkte, die diese Bedingungen erfüllen, bilden die toppling-
        kritische Zone. Diese wird mit diagonaler roter Schraffur dargestellt
        und mit einer rot gestrichelten Linie umrandet.

    Parameter:
      slope_dipdir       : Fallrichtung der Hangfläche (Azimut in Grad)
      slope_dip          : Fallen der Hangfläche (Grad)
      friction_angle     : Reibungswinkel φ_j (Grad)
      dipdir_tolerance   : Azimut-Toleranz (Faustregel ~ 10°)
      grid_step          : Gitterabstand fürs Schmidt-Netz
      figsize            : Figurgröße in Zoll
      save_fig           : Dateiname zum Speichern (oder None)
      dpi                : Auflösung beim Speichern
      ax                 : vorhandenes Axes-Objekt (oder None)
      add_title          : True -> Titel setzen
      add_legend         : True -> Legende für diese Zone einblenden
      plot_slope         : True -> Hang-Großkreis + -Pol zeichnen
    """

    # Winkel und Parameter in normierte float-Werte überführen
    slope_dipdir   = float(slope_dipdir) % 360.0
    slope_dip      = float(slope_dip)
    friction_angle = float(friction_angle)
    tol            = float(dipdir_tolerance)

    # Aus der Toppling-Bedingung (90° - ψ_f) + φ_j < ψ_P
    # ergibt sich die Mindest-Steilheit der Trennfläche:
    dip_min = (90.0 - slope_dip) + friction_angle

    # --- Figure / Axes Setup ---
    # Wenn kein Axes übergeben wurde: neues Schmidt-Netz erzeugen
    if ax is None:
        fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    else:
        # Wenn ax existiert: zugehörige Figure holen
        fig = ax.figure

    # Optional Titel setzen
    if add_title:
        fig.suptitle("Toppling Failure – kritische Pol-Zone", y=0.98)

    # Trivialfall: wenn die Mindest-Neigung ≥ 90°, kann keine reale Fläche
    # diese Bedingung erfüllen -> kein toppling-kritischer Bereich.
    if dip_min >= 90.0:
        print("Kein toppling-kritischer Bereich: (90° - ψ_f) + φ_j ≥ 90°.")
        if plot_slope:
            # Hang-Großkreis
            Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
            ax.plot(Xgc, Ygc, color="k", linewidth=2.5)
            # Hang-Pol
            tp, pp = plane_pole_from_dipdir(slope_dipdir, slope_dip)
            Xp, Yp = project_trend_plunge(tp, pp)
            ax.plot(Xp, Yp, "ko", markersize=10)
        # ggf. speichern und zurück
        if save_fig is not None:
            fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
        return fig, ax

    # Hangfläche (Großkreis + Pol) zeichnen
    if plot_slope:
        # Großkreis der Hangfläche
        Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
        ax.plot(Xgc, Ygc, color="k", linewidth=2.5)
        # Pol der Hangfläche
        trend_pole_slope, plunge_pole_slope = plane_pole_from_dipdir(
            slope_dipdir, slope_dip
        )
        Xp_slope, Yp_slope = project_trend_plunge(trend_pole_slope, plunge_pole_slope)
        ax.plot(Xp_slope, Yp_slope,
                marker="o", markersize=10,
                markerfacecolor="k", markeredgecolor="k",
                linestyle="none")

    # Ziel-Fallrichtung für Toppling: Gegenrichtung des Hangs
    dipdir_opposite = (slope_dipdir + 180.0) % 360.0

    # Punkt-in-Zone-Test:
    # prüft, ob ein gegebener Punkt (x, y) im Schmidt-Netz zu einer Trennfläche
    # gehört, die die Toppling-Bedingungen erfüllt.
    def point_in_zone(x, y):
        # außerhalb des Schmidt-Randes -> nein
        if x*x + y*y > 1.0:
            return False

        # 2D -> 3D-Polvektor
        nx, ny, nz = xy_to_xyz(x, y)

        # Poltrend / Polplunge
        trend_pole, plunge_pole = xyz_to_trend_plunge(nx, ny, nz)

        # Pol -> Ebenenorientierung:
        # - dip = 90° - plunge_pole
        # - dipdir = (trend_pole - 180°) % 360 (Pol zeigt nach „oben“ der Ebene)
        dip = 90.0 - plunge_pole
        dipdir = (trend_pole - 180.0) % 360.0

        # 1) Trennfläche muss innerhalb des Azimut-Sektors um die Gegenrichtung liegen
        if abs(circular_diff(dipdir, dipdir_opposite)) > tol:
            return False

        # 2) Trennfläche muss steiler als dip_min sein (Toppling-Bedingung)
        if dip <= dip_min:
            return False

        return True

    # diagonale Schraffur
    # Wir parametrisieren Linien mit x + y = const (u), also 45°-Linien im Diagramm,
    # und zeichnen nur die Segmente, für die point_in_zone == True ist.
    N_stripes = 40
    u_vals = np.linspace(-1.5, 1.5, N_stripes)
    for u in u_vals:
        xs = np.linspace(-1.2, 1.2, 600)
        ys = u - xs
        X_seg, Y_seg = [], []
        for x, y in zip(xs, ys):
            if point_in_zone(x, y):
                # inneres Segment der Zone
                X_seg.append(x); Y_seg.append(y)
            else:
                # Segmentende: bisher gesammelte Punkte plotten
                if X_seg:
                    ax.plot(X_seg, Y_seg,
                            color="red", linewidth=0.6, alpha=0.8, zorder=2)
                    X_seg, Y_seg = [], []
        # letztes Segment dieser Linie zeichnen, falls vorhanden
        if X_seg:
            ax.plot(X_seg, Y_seg,
                    color="red", linewidth=0.6, alpha=0.8, zorder=2)

    # Kontur
    # Gitter über das Netz legen, für jeden Punkt prüfen, ob er in der Zone liegt
    # und daraus eine Maske aufbauen. Anschließend mit contour(...) die Randlinie
    # der Zone (inkl. Form) als rot gestrichelte Linie zeichnen.
    Ngrid = 400
    xs = np.linspace(-1.0, 1.0, Ngrid)
    ys = np.linspace(-1.0, 1.0, Ngrid)
    Xg2, Yg2 = np.meshgrid(xs, ys)
    mask = np.zeros_like(Xg2, dtype=float)
    for i in range(Ngrid):
        for j in range(Ngrid):
            if point_in_zone(Xg2[i, j], Yg2[i, j]):
                mask[i, j] = 1.0

    ax.contour(
        Xg2, Yg2, mask,
        levels=[0.5],        # Schwelle zwischen 0 und 1
        colors="red",
        linestyles="--",
        linewidths=1.5,
        zorder=3
    )

    # Legende für Toppling-Zone
    if add_legend:
        # Hang-Großkreis
        slope_line = Line2D([0], [0],
                            color="black", linewidth=2.5,
                            label="Hangfläche (Großkreis)")
        # Hang-Pol
        slope_pole = Line2D([0], [0],
                            marker="o", linestyle="none",
                            markerfacecolor="black", markeredgecolor="black",
                            markersize=8, label="Pol der Hangfläche")
        # Symbol für die toppling-kritische Zone (diagonale Schraffur)
        zone_patch_legend = Patch(
            facecolor='none',
            edgecolor='red',
            linestyle='--',
            hatch='/',
            label="kritische Zone (toppling failure)"
        )
        ax.legend(handles=[slope_line, slope_pole, zone_patch_legend],
                  loc="upper left", bbox_to_anchor=(1.05, 1.0))

    # Figur bei Bedarf speichern
    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    return fig, ax





def plot_plane_failure_zone_orthotilt(slope_dipdir,
                                      slope_dip,
                                      friction_angle,
                                      max_orthotilt=20.0,
                                      grid_step=10,
                                      figsize=(6, 6),
                                      save_fig=None,
                                      dpi=400,
                                      ax=None,
                                      add_title=True,
                                      add_legend=True,
                                      plot_slope=True):
    """
    Kritische Zone für Plane Failure mit begrenzter „orthogonaler Verkippung“.

    Eine Trennfläche J gehört zur Zone, wenn:

      1) Reibung / Steilheit:
           friction_angle < dip_J < slope_dip

      2) Sie fällt grundsätzlich in Richtung des Hangs:
           |dipdir_J - slope_dipdir| <= 90°

      3) Es gibt mindestens eine Referenzfläche J* mit:
           - dipdir_J* = slope_dipdir
           - friction_angle <= dip_J* <= slope_dip
         sodass der Winkel zwischen n_J und n_J* (Polvektoren)
         höchstens max_orthotilt beträgt.
    """

    import math

    # Eingaben normieren
    slope_dipdir   = float(slope_dipdir) % 360.0
    slope_dip      = float(slope_dip)
    friction_angle = float(friction_angle)
    max_orthotilt  = float(max_orthotilt)

    # --- Figure / Axes Setup ---
    if ax is None:
        fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    else:
        fig = ax.figure

    if add_title:
        fig.suptitle("Plane Failure – Zone mit begrenzter orthogonaler Verkippung",
                     y=0.98)

    # Trivialfall: Reibungswinkel ≥ Hangneigung -> kein Versagen möglich
    if friction_angle >= slope_dip:
        print("Kein kritischer Bereich: Reibungswinkel ≥ Hangneigung.")
        if plot_slope:
            Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
            ax.plot(Xgc, Ygc, color="k", linewidth=2.5)
            tp, pp = plane_pole_from_dipdir(slope_dipdir, slope_dip)
            Xp, Yp = project_trend_plunge(tp, pp)
            ax.plot(Xp, Yp, "ko", markersize=10)
        if save_fig is not None:
            fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
        return fig, ax

    # --- Hangfläche (Großkreis + Pol) ---
    if plot_slope:
        Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
        ax.plot(Xgc, Ygc, color="k", linewidth=2.5)

        trend_pole_slope, plunge_pole_slope = plane_pole_from_dipdir(
            slope_dipdir, slope_dip
        )
        Xp_slope, Yp_slope = project_trend_plunge(trend_pole_slope, plunge_pole_slope)
        ax.plot(Xp_slope, Yp_slope,
                marker="o", markersize=10,
                markerfacecolor="k", markeredgecolor="k",
                linestyle="none")

    # --- Vorbereitungen für analytische Orthotilt-Berechnung ---
    # Pol-Azimut der Referenzfamilie (alle J* haben denselben Trend des Pols)
    trend_pole_slope, _ = plane_pole_from_dipdir(slope_dipdir, slope_dip)
    T = math.radians(trend_pole_slope)
    sinT, cosT = math.sin(T), math.cos(T)

    phi_rad = 0.0  # Referenzfamilie beginnt bei horizontal (0° Dip), unabhängig vom friction angle
    psi_rad = math.radians(slope_dip)
    max_orthotilt_rad = math.radians(max_orthotilt)

    def min_angle_to_family(nx_J, ny_J, nz_J):
        """
        Minimaler Winkel zwischen Pol n_J (nx_J,ny_J,nz_J) und allen
        Polvektoren n(dip) der Referenzfamilie:
           dipdir = slope_dipdir, dip ∈ [friction_angle, slope_dip]
        """

        # Koefizienten für dot(dip) = n_J · n(dip) = R * sin(dip - δ)
        c1 = nx_J * sinT + ny_J * cosT
        c2 = nz_J
        R = math.hypot(c1, c2)

        # Sonderfall: R ~ 0 -> dot(dip) ≈ 0 -> Winkel ≈ 90°
        if R == 0.0:
            return 90.0

        delta = math.atan2(c2, c1)

        def f(d):
            # Skalarprodukt für gegebenen Dip d (in Radiant)
            return R * math.sin(d - delta)

        # Kandidaten: Intervall-Endpunkte + evtl. inneres Maximum bei d_crit
        vals = [f(phi_rad), f(psi_rad)]

        d_crit = delta + math.pi / 2.0  # Stelle, an der sin(...) maximal ist
        if phi_rad <= d_crit <= psi_rad:
            vals.append(f(d_crit))

        best_dot = max(vals)
        best_dot = max(-1.0, min(1.0, best_dot))
        angle = math.degrees(math.acos(best_dot))
        return angle

    # --- Punkt-in-Zone-Test ---
    def point_in_zone(x, y):
        # außerhalb des Schmidt-Randes
        if x**2 + y**2 > 1.0:
            return False

        # 2D -> 3D-Polvektor n_J
        nx_J, ny_J, nz_J = xy_to_xyz(x, y)

        # Poltrend & Polplunge der realen Trennfläche
        trend_pole_J, plunge_pole_J = xyz_to_trend_plunge(nx_J, ny_J, nz_J)

        # Ebenendaten der realen Trennfläche
        dip_J = 90.0 - plunge_pole_J
        dipdir_J = (trend_pole_J - 180.0) % 360.0

        # 1) Steilheits-/Friction-Bedingung
        if not (friction_angle < dip_J < slope_dip):
            return False

        # 2) Trennfläche muss grob in Richtung des Hangs einfallen
        #    (kein Plane Failure, wenn sie „nach hinten“ kippt)
        if abs(circular_diff(dipdir_J, slope_dipdir)) > 90.0:
            return False

        # 3) Orthogonale Verkippung = minimaler Winkel zur Referenzfamilie
        angle_extra = min_angle_to_family(nx_J, ny_J, nz_J)
        if angle_extra > max_orthotilt:
            return False

        return True

    # --- horizontale Schraffur ---
    N_stripes = 40
    y_vals = np.linspace(-1.0, 1.0, N_stripes)
    for y in y_vals:
        X_line = np.linspace(-1.0, 1.0, 500)
        X_seg, Y_seg = [], []
        for x in X_line:
            if point_in_zone(x, y):
                X_seg.append(x); Y_seg.append(y)
            else:
                if X_seg:
                    ax.plot(X_seg, Y_seg,
                            color="red", linewidth=0.6, alpha=0.8, zorder=2)
                    X_seg, Y_seg = [], []
        if X_seg:
            ax.plot(X_seg, Y_seg,
                    color="red", linewidth=0.6, alpha=0.8, zorder=2)

    # --- Kontur ---
    Ngrid = 400
    xs = np.linspace(-1.0, 1.0, Ngrid)
    ys = np.linspace(-1.0, 1.0, Ngrid)
    Xg, Yg = np.meshgrid(xs, ys)
    mask = np.zeros_like(Xg, dtype=float)
    for i in range(Ngrid):
        for j in range(Ngrid):
            if point_in_zone(Xg[i, j], Yg[i, j]):
                mask[i, j] = 1.0

    ax.contour(
        Xg, Yg, mask,
        levels=[0.5],
        colors="red",
        linestyles="--",
        linewidths=1.5,
        zorder=3
    )

    # --- Legende ---
    if add_legend:
        slope_line = Line2D([0], [0],
                            color="black", linewidth=2.5,
                            label="Hangfläche (Großkreis)")
        slope_pole = Line2D([0], [0],
                            marker="o", linestyle="none",
                            markerfacecolor="black", markeredgecolor="black",
                            markersize=8, label="Pol der Hangfläche")
        zone_patch_legend = Patch(facecolor='none',
                                  edgecolor='red',
                                  linestyle='--',
                                  hatch='-',
                                  label="Plane failure (orth. Verkippung ≤ {:.0f}°)".format(max_orthotilt))
        ax.legend(handles=[slope_line, slope_pole, zone_patch_legend],
                  loc="upper left", bbox_to_anchor=(1.05, 1.0))

    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    return fig, ax





def plot_all_critical_zones(slope_dipdir,
                            slope_dip,
                            friction_angle,
                            show_plane=True,
                            show_plane_orthotilt=False,
                            show_wedge=True,
                            show_toppling=True,
                            dipdir_tolerance_plane=20.0,
                            dipdir_tolerance_toppling=10.0,
                            max_orthotilt=20.0,
                            grid_step=10,
                            figsize=(6, 6),
                            save_fig=None,
                            dpi=400):
    """
    Kombiniert:
      - plot_plane_failure_zone
      - plot_plane_failure_zone_orthotilt
      - plot_wedge_failure_marklandsche_area
      - plot_toppling_failure_zone
    """

    # Eingaben normieren / in float umwandeln
    slope_dipdir = float(slope_dipdir) % 360.0
    slope_dip = float(slope_dip)
    friction_angle = float(friction_angle)

    # 1) Basis-Schmidt-Netz
    fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)

    
    # --- Titel: Hauptzeile fett+unterstrichen, Rest kleiner, alles zentriert ---
    main_title = "Stabilitätsanalyse - Kritische Zonen"
    subtitle = (
        f"Fallrichtung: {slope_dipdir:.1f}°\n"
        f"Fallen: {slope_dip:.1f}°\n"
        f"Reibungswinkel: {friction_angle:.1f}°"
    )

    # Haupttitel (fett + unterstrichen)
    fig.text(
        0.5, 1.0,
        main_title,
        ha="center", va="top",
        fontsize=14,
        fontweight="bold"
    )

    # Untertitel (kleiner)
    fig.text(
        0.5, 0.96,
        subtitle,
        ha="center", va="top",
        fontsize=10
    )



    # Wir wollen die Hangfläche (Großkreis + Pol) genau EINMAL zeichnen.
    slope_already_drawn = False

    # 2) klassische Plane-failure-Zone
    if show_plane:
        plot_plane_failure_zone(
            slope_dipdir=slope_dipdir,
            slope_dip=slope_dip,
            friction_angle=friction_angle,
            dipdir_tolerance=dipdir_tolerance_plane,
            grid_step=grid_step,
            ax=ax,
            add_title=False,
            add_legend=False,
            plot_slope=not slope_already_drawn
        )
        slope_already_drawn = True

    # 3) Plane-failure-Zone mit begrenzter orthogonaler Verkippung
    if show_plane_orthotilt:
        plot_plane_failure_zone_orthotilt(
            slope_dipdir=slope_dipdir,
            slope_dip=slope_dip,
            friction_angle=friction_angle,
            max_orthotilt=max_orthotilt,
            grid_step=grid_step,
            ax=ax,
            add_title=False,
            add_legend=False,
            plot_slope=not slope_already_drawn
        )
        slope_already_drawn = True

    # 4) Critical Pole Vector Zone (Wedge failure / Marklandsche Fläche)
    if show_wedge:
        plot_wedge_failure_marklandsche_area(
            slope_dipdir=slope_dipdir,
            slope_dip=slope_dip,
            friction_angle=friction_angle,
            grid_step=grid_step,
            ax=ax,
            add_title=False,
            add_legend=False,
            plot_slope=not slope_already_drawn
        )
        slope_already_drawn = True

    # 5) Toppling-Zone
    if show_toppling:
        plot_toppling_failure_zone(
            slope_dipdir=slope_dipdir,
            slope_dip=slope_dip,
            friction_angle=friction_angle,
            dipdir_tolerance=dipdir_tolerance_toppling,
            grid_step=grid_step,
            ax=ax,
            add_title=False,
            add_legend=False,
            plot_slope=not slope_already_drawn
        )
        slope_already_drawn = True

    # Falls keine Zone die Hangfläche gezeichnet hat:
    if not slope_already_drawn:
        Xgc, Ygc = great_circle_from_plane(slope_dipdir, slope_dip)
        ax.plot(Xgc, Ygc, color="k", linewidth=2.0)
        tp, pp = plane_pole_from_dipdir(slope_dipdir, slope_dip)
        Xp, Yp = project_trend_plunge(tp, pp)
        ax.plot(Xp, Yp,
                marker="o", markersize=10,
                markerfacecolor="k", markeredgecolor="k",
                linestyle="none")

    # 6) Gemeinsame Legende mit großen Symbolen
    legend_handles = []

    legend_handles.append(
        Line2D([0], [0],
               color="black",
               linewidth=3.0,
               label="Hangfläche (Großkreis)")
    )

    legend_handles.append(
        Line2D([0], [0],
               marker="o",
               linestyle="none",
               markersize=12,
               markerfacecolor="black",
               markeredgecolor="black",
               label="Pol der Hangfläche")
    )

    # klassische Plane failure – horizontale Schraffur
    if show_plane:
        legend_handles.append(
            Patch(facecolor='none',
                  edgecolor='red',
                  linestyle='--',
                  hatch='-',
                  label="Plane failure – kritische Zone")
        )

    # orthotilt-Plane-failure – Kreuzschraffur
    if show_plane_orthotilt:
        legend_handles.append(
            Patch(facecolor='none',
                  edgecolor='red',
                  linestyle='--',
                  hatch='-',
                  label="Plane failure – kritische Zone")
        )

    # Critical Pole Vector Zone – vertikale Schraffur
    if show_wedge:
        legend_handles.append(
            Patch(facecolor='none',
                  edgecolor='red',
                  linestyle='--',
                  hatch='|',
                  label="Wedge failure - kritische Zone")
        )

    # Toppling – diagonale Schraffur
    if show_toppling:
        legend_handles.append(
            Patch(facecolor='none',
                  edgecolor='red',
                  linestyle='--',
                  hatch='/',
                  label="Toppling failure – kritische Zone")
        )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        handlelength=6.0,
        handleheight=6.0,
        borderpad=0.8,
        labelspacing=2.0,
        markerscale=1.5
    )

    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_joint_critical_zones_from_table(
    path,
    slope_dipdir,
    slope_dip,
    show_plane=True,
    show_plane_orthotilt=False,
    show_wedge=False,
    show_toppling=True,
    dipdir_tolerance_plane=20.0,
    dipdir_tolerance_toppling=10.0,
    max_orthotilt=20.0,
    grid_step=10,
    figsize=(6, 6),
    dpi=400,
    save_prefix=None
):
    """
    Liest Trennflächen aus einer CSV- oder Excel-Datei und erzeugt
    für jede Trennfläche eine eigene Stabilitätsanalyse-Abbildung
    im Schmidt-Netz.

    Erwartete Spalten in der Tabelle:

        name            : Name der Trennfläche
        dip             : Fallen der Trennfläche (Grad)
        dipdir          : Fallrichtung der Trennfläche (Grad)
        friction_angle  : Reibungswinkel der Trennfläche (Grad)

    Falls diese Spaltennamen NICHT vorhanden sind, werden die ersten
    vier Spalten in genau dieser Reihenfolge interpretiert.

    Zusätzlich:
    - Für jede Trennfläche wird geprüft, ob ihr Pol
      in einer der gewählten kritischen Zonen liegt, und ein
      fett geschriebener Hinweis „Warnung …“ oder „Unkritisch …“
      im Plot angezeigt.
    - Es wird eine Zusammenfassungs-Tabelle ausgegeben:
      * Spalte „kritisch“ (bool)
      * „Status“ (kritisch/unkritisch)
      * „Failure“ (Plane/Wedge/Toppling …)
      * Name der Trennfläche
      Die Tabelle ist von kritisch nach unkritisch sortiert und
      kritisch/unkritisch leicht rot/grün hinterlegt.
    - Die Reihenfolge der zurückgegebenen Figuren folgt derselben Logik:
      zuerst alle kritischen, dann unkritische.
    """

    import os
    import math
    import pandas as pd
    from matplotlib.path import Path

    # --- 1) Tabelle einlesen ---
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx", ".xlsm", ".xltx", ".xltm"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    required_cols = ["name", "dip", "dipdir", "friction_angle"]

    if not set(required_cols).issubset(df.columns):
        # Fallback: erste 4 Spalten als (name, dip, dipdir, friction_angle)
        if df.shape[1] < 4:
            raise ValueError(
                "Tabelle braucht mindestens 4 Spalten: "
                "(Name, Fallen, Fallrichtung, Reibungswinkel)."
            )
        df = df.iloc[:, :4].copy()
        df.columns = required_cols

    # Eingaben normieren
    slope_dipdir = float(slope_dipdir) % 360.0
    slope_dip = float(slope_dip)

    # --- 2) Farb-Cycle für die Trennflächen ---
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color',
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
         'tab:purple', 'tab:brown']
    )

    # ---------------------------------------------------
    # 3) Vorbereitungen für Zonen-Tests (ohne Zeichnen)
    # ---------------------------------------------------

    # 3a) Hilfsfunktion: klassische Plane-failure-Zone
    def in_plane_zone(dipdir_J, dip_J, phi_J):
        # Reibungs- / Steilheitsbedingung
        if not (phi_J < dip_J < slope_dip):
            return False
        # Dipdir in „Hangrichtung ± Toleranz“
        if abs(circular_diff(dipdir_J, slope_dipdir)) > dipdir_tolerance_plane:
            return False
        return True

    # 3b) Orthotilt-Zone: Vorberechnung (nur einmal)
    if show_plane_orthotilt:
        trend_pole_slope, _ = plane_pole_from_dipdir(slope_dipdir, slope_dip)
        T = math.radians(trend_pole_slope)
        sinT, cosT = math.sin(T), math.cos(T)

        phi_rad = 0.0  # Referenzfamilie: Dip von 0° bis slope_dip
        psi_rad = math.radians(slope_dip)

        def min_angle_to_family(nx_J, ny_J, nz_J):
            """
            minimaler Winkel zwischen Pol n_J und allen
            Polvektoren der Familie:
                dipdir = slope_dipdir, dip ∈ [0°, slope_dip]
            """
            c1 = nx_J * sinT + ny_J * cosT
            c2 = nz_J
            R = math.hypot(c1, c2)

            if R == 0.0:
                return 90.0

            delta = math.atan2(c2, c1)

            def f(d):
                return R * math.sin(d - delta)

            vals = [f(phi_rad), f(psi_rad)]

            d_crit = delta + math.pi / 2.0
            if phi_rad <= d_crit <= psi_rad:
                vals.append(f(d_crit))

            best_dot = max(vals)
            best_dot = max(-1.0, min(1.0, best_dot))
            angle = math.degrees(math.acos(best_dot))
            return angle

        def in_plane_orthotilt_zone(dipdir_J, dip_J, phi_J):
            # 1) Steilheits-/Friction-Bedingung
            if not (phi_J < dip_J < slope_dip):
                return False
            # 2) grob in Hangrichtung (kein „nach hinten“ Kippen)
            if abs(circular_diff(dipdir_J, slope_dipdir)) > 90.0:
                return False
            # 3) minimaler Winkel zur Referenzfamilie
            trend_pole_J, plunge_pole_J = plane_pole_from_dipdir(dipdir_J, dip_J)
            nx_J, ny_J, nz_J = trend_plunge_to_xyz(trend_pole_J, plunge_pole_J)
            angle_extra = min_angle_to_family(nx_J, ny_J, nz_J)
            return angle_extra <= max_orthotilt
    else:
        def in_plane_orthotilt_zone(dipdir_J, dip_J, phi_J):
            return False

    # 3c) Wedge / Markland: Critical Pole Vector Zone (Polraum-Polygon)
    if show_wedge:
        Xgc_slope, Ygc_slope = great_circle_from_plane(slope_dipdir, slope_dip)
        X_zone, Y_zone = [], []
        for Xg, Yg in zip(Xgc_slope, Ygc_slope):
            if Xg**2 + Yg**2 > 1.0:
                continue
            vx, vy, vz = xy_to_xyz(Xg, Yg)
            trend_L, plunge_L = xyz_to_trend_plunge(vx, vy, vz)
            trend_pole_L = (trend_L + 180.0) % 360.0
            plunge_pole_L = 90.0 - plunge_L
            Xp, Yp = project_trend_plunge(trend_pole_L, plunge_pole_L)
            X_zone.append(Xp)
            Y_zone.append(Yp)

        if len(X_zone) > 0:
            X_zone = np.asarray(X_zone)
            Y_zone = np.asarray(Y_zone)
            verts_x = np.concatenate(([0.0], X_zone, [0.0]))
            verts_y = np.concatenate(([0.0], Y_zone, [0.0]))
            poly_path = Path(
                np.column_stack((verts_x, verts_y)),
                closed=True
            )
        else:
            poly_path = None

        def friction_radius2(phi_J):
            if phi_J is None or phi_J <= 0.0:
                return 0.0
            trend_test = 0.0
            plunge_pole_f = 90.0 - phi_J
            Xf0, Yf0 = project_trend_plunge(trend_test, plunge_pole_f)
            return Xf0**2 + Yf0**2

        def in_wedge_zone(dipdir_J, dip_J, phi_J):
            if poly_path is None:
                return False
            trend_pole_J, plunge_pole_J = plane_pole_from_dipdir(dipdir_J, dip_J)
            Xp_J, Yp_J = project_trend_plunge(trend_pole_J, plunge_pole_J)
            r2 = Xp_J*Xp_J + Yp_J*Yp_J
            if r2 > 1.0:
                return False
            if not poly_path.contains_point((Xp_J, Yp_J)):
                return False
            r_f2 = friction_radius2(phi_J)
            if phi_J is not None and r2 <= r_f2:
                return False
            return True
    else:
        def in_wedge_zone(dipdir_J, dip_J, phi_J):
            return False

    # 3d) Toppling-Zone (Analytik in Dip/Dipdir)
    def in_toppling_zone(dipdir_J, dip_J, phi_J):
        if not show_toppling:
            return False
        dip_min = (90.0 - slope_dip) + phi_J
        if dip_J <= dip_min:
            return False
        dipdir_opp = (slope_dipdir + 180.0) % 360.0
        if abs(circular_diff(dipdir_J, dipdir_opp)) > dipdir_tolerance_toppling:
            return False
        return True

    # ---------------------------------------
    # Sammel-Container für figs & Summary
    # ---------------------------------------
    figs = []
    rows_summary = []        # für Tabelle
    figs_extended = []       # für sortierte Rückgabe

    # ---------------------------------------------------
    # 4) Über alle Trennflächen iterieren
    # ---------------------------------------------------
    for idx, row in df.iterrows():
        name = str(row["name"])
        dip = float(row["dip"])
        dipdir = float(row["dipdir"]) % 360.0
        phi = float(row["friction_angle"])

        color = color_cycle[idx % len(color_cycle)]

        # Basis-Schmidt-Netz
        fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)

        # -----------------------------------------------
        # Titel (Hang + Trennfläche)
        # -----------------------------------------------
        main_title = "Stabilitätsanalyse - Kritische Zonen"
        subtitle = (
            f"Hang: ({slope_dip:.1f}°/{slope_dipdir:05.1f}°)\n"
            f"Trennfläche {name}: ({dip:.1f}°/{dipdir:05.1f}°) "
            f"mit Reibungswinkel: {phi:.1f}°"
        )

        fig.text(
            0.5, 1.0,
            main_title,
            ha="center", va="top",
            fontsize=14,
            fontweight="bold"
        )

        fig.text(
            0.5, 0.96,
            subtitle,
            ha="center", va="top",
            fontsize=10
        )

        # -----------------------------------------------
        # 4a) Zone-Mitgliedschaft der Trennfläche prüfen
        # -----------------------------------------------
        zone_hits = []

        if show_plane and in_plane_zone(dipdir, dip, phi):
            zone_hits.append("Plane failure")
        if show_plane_orthotilt and in_plane_orthotilt_zone(dipdir, dip, phi):
            zone_hits.append("Plane failure")
        if show_wedge and in_wedge_zone(dipdir, dip, phi):
            zone_hits.append("Wedge failure")
        if show_toppling and in_toppling_zone(dipdir, dip, phi):
            zone_hits.append("Toppling failure")

        # Kritikalität & Failure-Zusammenfassung für Tabelle
        if zone_hits:
            is_critical = True
            failure_summary = ", ".join(sorted(set(zone_hits)))
        else:
            is_critical = False
            failure_summary = "—"

        if zone_hits:
            if len(zone_hits) == 1:
                warn_text = (
                    f"Warnung: Polpunkt der Trennfläche {name} liegt in der "
                    f"folgenden kritischen Zone:\n- {zone_hits[0]}"
                )
            else:
                warn_text = (
                    f"Warnung: Polpunkt der Trennfläche {name} liegt in folgenden "
                    f"kritischen Zonen:\n- " + "\n- ".join(zone_hits)
                )
            # Hinweis UNTER der Abbildung
            ax.text(
                0.5, -0.25,
                warn_text,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=15,
                fontweight="bold",
                color="red"
            )
        else:
            ok_text = (
                f"Unkritisch: Polpunkt der Trennfläche {name} liegt in keiner "
                f"kritischen Zone."
            )
            ax.text(
                0.5, -0.2,
                ok_text,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=15,
                fontweight="bold",
                color="darkgreen"
            )

        # -----------------------------------------------
        # 4b) Kritische Zonen (zeichnen)
        # -----------------------------------------------
        slope_already_drawn = False

        if show_plane:
            plot_plane_failure_zone(
                slope_dipdir=slope_dipdir,
                slope_dip=slope_dip,
                friction_angle=phi,
                dipdir_tolerance=dipdir_tolerance_plane,
                grid_step=grid_step,
                ax=ax,
                add_title=False,
                add_legend=False,
                plot_slope=not slope_already_drawn
            )
            slope_already_drawn = True

        if show_plane_orthotilt:
            plot_plane_failure_zone_orthotilt(
                slope_dipdir=slope_dipdir,
                slope_dip=slope_dip,
                friction_angle=phi,
                max_orthotilt=max_orthotilt,
                grid_step=grid_step,
                ax=ax,
                add_title=False,
                add_legend=False,
                plot_slope=not slope_already_drawn
            )
            slope_already_drawn = True

        if show_wedge:
            plot_wedge_failure_marklandsche_area(
                slope_dipdir=slope_dipdir,
                slope_dip=slope_dip,
                friction_angle=phi,
                grid_step=grid_step,
                ax=ax,
                add_title=False,
                add_legend=False,
                plot_slope=not slope_already_drawn
            )
            slope_already_drawn = True

        if show_toppling:
            plot_toppling_failure_zone(
                slope_dipdir=slope_dipdir,
                slope_dip=slope_dip,
                friction_angle=phi,
                dipdir_tolerance=dipdir_tolerance_toppling,
                grid_step=grid_step,
                ax=ax,
                add_title=False,
                add_legend=False,
                plot_slope=not slope_already_drawn
            )
            slope_already_drawn = True

        # -----------------------------------------------
        # 4c) Trennfläche selbst: Großkreis + Pol
        # -----------------------------------------------
        Xgc_j, Ygc_j = great_circle_from_plane(dipdir, dip)
        ax.plot(Xgc_j, Ygc_j,
                color=color,
                linewidth=2.0,
                linestyle='-')

        trend_pole_j, plunge_pole_j = plane_pole_from_dipdir(dipdir, dip)
        Xp_j, Yp_j = project_trend_plunge(trend_pole_j, plunge_pole_j)
        ax.plot(Xp_j, Yp_j,
                marker='o',
                markersize=10,
                linestyle='none',
                markerfacecolor=color,
                markeredgecolor='k')

        # -----------------------------------------------
        # 4d) Legende
        # -----------------------------------------------
        legend_handles = []

        legend_handles.append(
            Line2D([0], [0],
                   color="black",
                   linewidth=3.0,
                   label="Hangfläche (Großkreis)")
        )
        legend_handles.append(
            Line2D([0], [0],
                   marker="o",
                   linestyle="none",
                   markersize=12,
                   markerfacecolor="black",
                   markeredgecolor="black",
                   label="Pol der Hangfläche")
        )

        legend_handles.append(
            Line2D([0], [0],
                   color=color,
                   linewidth=2.0,
                   label=f"Trennfläche {name} (Großkreis)")
        )
        legend_handles.append(
            Line2D([0], [0],
                   marker="o",
                   linestyle="none",
                   markersize=10,
                   markerfacecolor=color,
                   markeredgecolor="k",
                   label=f"Pol Trennfläche {name}")
        )

        if show_plane:
            legend_handles.append(
                Patch(facecolor='none',
                      edgecolor='red',
                      linestyle='--',
                      hatch='-',
                      label="Plane failure – kritische Zone")
            )
        if show_plane_orthotilt:
            legend_handles.append(
                Patch(facecolor='none',
                      edgecolor='red',
                      linestyle='--',
                      hatch='-',
                      label="Plane failure – kritische Zone")
            )
        if show_wedge:
            legend_handles.append(
                Patch(facecolor='none',
                      edgecolor='red',
                      linestyle='--',
                      hatch='|',
                      label="Wedge failure - kritische Zone")
            )
        if show_toppling:
            legend_handles.append(
                Patch(facecolor='none',
                      edgecolor='red',
                      linestyle='--',
                      hatch='/',
                      label="Toppling failure – kritische Zone")
            )

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.5, 1.15),
            handlelength=6.0,
            handleheight=6.0,
            borderpad=0.8,
            labelspacing=2.0,
            markerscale=1.5
        )

        # -----------------------------------------------
        # 4e) Optional: speichern
        # -----------------------------------------------
        if save_prefix is not None:
            safe_name = "".join(
                c if c.isalnum() or c in "._-"
                else "_"
                for c in name
            )
            filename = f"{save_prefix}_{idx+1}_{safe_name}.png"
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")

        # -----------------------------------------------
        # 4f) Infos für Summary / Sortierung sammeln
        # -----------------------------------------------
        rows_summary.append({
            "kritisch": is_critical,
            "Status": "kritisch" if is_critical else "unkritisch",
            "Failure": failure_summary,
            "Trennfläche": name,
            "dip": dip,
            "dipdir": dipdir,
            "phi": phi,
        })

        figs_extended.append(
            dict(
                name=name,
                fig=fig,
                ax=ax,
                kritisch=is_critical
            )
        )

    # ---------------------------------------------------
    # 5) Zusammenfassungs-Tabelle & Sortierung der Figuren
    # ---------------------------------------------------
    if rows_summary:
        df_summary = pd.DataFrame(rows_summary)

        # Kritische zuerst, dann unkritische
        df_summary = df_summary.sort_values(
            by="kritisch", ascending=False
        ).reset_index(drop=True)

        # leichte Rot-/Grün-Hinterlegung
        def _color_row(row):
            if row["kritisch"]:
                color = "background-color: rgba(255, 0, 0, 0.15)"  # leicht rot
            else:
                color = "background-color: rgba(0, 255, 0, 0.15)"  # leicht grün
            return [color] * len(row)

        df_styled = df_summary.style.apply(_color_row, axis=1)

        # Tabelle anzeigen (Notebook) oder Fallback-Print
        try:
            from IPython.display import display
            display(df_styled)
        except Exception:
            print(df_summary)

        # Figuren in gleicher Logik sortieren:
        # erst kritisch (True), dann unkritisch (False)
        figs_extended.sort(
            key=lambda d: int(not d["kritisch"])
        )

        # zurück zur alten Struktur: (name, fig, ax)
        figs = [
            (d["name"], d["fig"], d["ax"])
            for d in figs_extended
        ]
    else:
        figs = []

    return figs




def plot_joint_pair_critical_zones_from_table(
    path,
    slope_dipdir,
    slope_dip,
    show_plane=False,
    show_plane_orthotilt=False,
    show_wedge=True,
    show_toppling=False,
    dipdir_tolerance_plane=20.0,
    dipdir_tolerance_toppling=10.0,
    max_orthotilt=20.0,
    grid_step=10,
    figsize=(6, 6),
    dpi=400,
    save_prefix=None
):
    """
    Paarweise Stabilitätsanalyse von Trennflächen:

    - Liest eine Tabelle mit:
        name, dip, dipdir, friction_angle
    - Betrachtet alle Paarkombinationen (i < j)
    - Für jedes Paar (i,j) werden Plots erzeugt:
        * normalerweise zwei:
              - einer mit Reibungswinkel φ_i
              - einer mit Reibungswinkel φ_j
          ABER: wenn φ_i == φ_j → nur EIN Plot
      -> jeweils:
         - Hang + kritische Zonen (plane / orthotilt / wedge / toppling)
         - beide Trennflächen (Großkreise + Pole)
         - Schnittlineation (zweifarbiges Kreuz)
         - 'Pol' der Schnittlineation (zweifarbiges Quadrat)

    - Die Warnung bezieht sich auf den 'Pol' der Schnittlineation:
        liegt dieser in einer der aktivierten kritischen Zonen?
        -> fett roter Warntext
        ansonsten: fett grün 'Unkritisch'

    Zusätzlich:
    - Es wird eine Zusammenfassungs-Tabelle ausgegeben, in der
      alle Paare (i,j) pro φ_used als kritisch/unkritisch
      aufgelistet werden (kritische oben, unkritische unten),
      mit leichter rot/grün-Hinterlegung.
    - Die Rückgabe-Liste der Figuren ist in der gleichen Logik
      sortiert (erst kritisch, dann unkritisch).
    """

    import os
    import math
    import pandas as pd
    from matplotlib.path import Path

    # --- 1) Tabelle einlesen ---
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx", ".xlsm", ".xltx", ".xltm"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    required_cols = ["name", "dip", "dipdir", "friction_angle"]
    if not set(required_cols).issubset(df.columns):
        if df.shape[1] < 4:
            raise ValueError(
                "Tabelle braucht mindestens 4 Spalten: "
                "(Name, Fallen, Fallrichtung, Reibungswinkel)."
            )
        df = df.iloc[:, :4].copy()
        df.columns = required_cols

    # Hang-Orientierung normieren
    slope_dipdir = float(slope_dipdir) % 360.0
    slope_dip = float(slope_dip)

    # Farbcyle für Trennflächen
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color',
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
         'tab:purple', 'tab:brown']
    )

    # ---------------------------------------------------
    # 2) Vorbereitung für Wedge-Zone (Marklandsche Fläche)
    #     -> äußere Polygonhülle im Polraum (unabhängig von φ)
    # ---------------------------------------------------
    if show_wedge:
        Xgc_slope, Ygc_slope = great_circle_from_plane(slope_dipdir, slope_dip)
        X_zone, Y_zone = [], []
        for Xg, Yg in zip(Xgc_slope, Ygc_slope):
            if Xg**2 + Yg**2 > 1.0:
                continue
            vx, vy, vz = xy_to_xyz(Xg, Yg)
            trend_L, plunge_L = xyz_to_trend_plunge(vx, vy, vz)
            trend_pole_L = (trend_L + 180.0) % 360.0
            plunge_pole_L = 90.0 - plunge_L
            Xp, Yp = project_trend_plunge(trend_pole_L, plunge_pole_L)
            X_zone.append(Xp)
            Y_zone.append(Yp)

        if len(X_zone) > 0:
            X_zone = np.asarray(X_zone)
            Y_zone = np.asarray(Y_zone)
            verts_x = np.concatenate(([0.0], X_zone, [0.0]))
            verts_y = np.concatenate(([0.0], Y_zone, [0.0]))
            wedge_poly_path = Path(
                np.column_stack((verts_x, verts_y)),
                closed=True
            )
        else:
            wedge_poly_path = None
    else:
        wedge_poly_path = None

    # Hilfsfunktion: Reibungskegel-Radius² im Polnetz
    def friction_radius2(phi):
        if phi is None or phi <= 0.0:
            return 0.0
        trend_test = 0.0
        plunge_pole_f = 90.0 - phi
        Xf0, Yf0 = project_trend_plunge(trend_test, plunge_pole_f)
        return Xf0**2 + Yf0**2

    # ---------------------------------------------------
    # 3) Vorbereitung für orthotilt-Berechnung (nur slope-abhängig)
    # ---------------------------------------------------
    # Pol der Hangfläche
    trend_pole_slope, plunge_pole_slope = plane_pole_from_dipdir(slope_dipdir, slope_dip)
    T_s = math.radians(trend_pole_slope)
    sinT_s, cosT_s = math.sin(T_s), math.cos(T_s)

    phi_rad_base = 0.0  # Referenz-Dip-Untergrenze (0°)
    psi_rad_base = math.radians(slope_dip)
    max_orthotilt_rad = math.radians(max_orthotilt)

    def min_angle_to_family(nx_J, ny_J, nz_J):
        """
        minimaler Winkel zwischen Pol n_J und allen
        Polen einer Familie von Ebenen mit:
           dipdir = slope_dipdir
           dip    ∈ [0°, slope_dip]
        """
        c1 = nx_J * sinT_s + ny_J * cosT_s
        c2 = nz_J
        R = math.hypot(c1, c2)
        if R == 0.0:
            return 90.0

        delta = math.atan2(c2, c1)

        def f(d):
            return R * math.sin(d - delta)

        vals = [f(phi_rad_base), f(psi_rad_base)]

        d_crit = delta + math.pi / 2.0
        if phi_rad_base <= d_crit <= psi_rad_base:
            vals.append(f(d_crit))

        best_dot = max(vals)
        best_dot = max(-1.0, min(1.0, best_dot))
        angle = math.degrees(math.acos(best_dot))
        return angle

    # ---------------------------------------------------
    # 4) Zonen-Check-Funktionen für einen gegebenen φ
    # ---------------------------------------------------
    def build_zone_checkers(phi_used):
        """
        Erzeugt Funktionen, die für einen gegebenen Polpunkt (Xp, Yp)
        testen, ob er in einer der Zonen liegt.
        """

        phi_used = float(phi_used)

        def pole_xy_to_plane_orientation(Xp, Yp):
            vx, vy, vz = xy_to_xyz(Xp, Yp)
            trend_pole, plunge_pole = xyz_to_trend_plunge(vx, vy, vz)
            dip_plane = 90.0 - plunge_pole
            dipdir_plane = (trend_pole - 180.0) % 360.0
            return dipdir_plane, dip_plane

        def in_plane_zone_pole(Xp, Yp):
            if not show_plane:
                return False
            if Xp*Xp + Yp*Yp > 1.0:
                return False
            dipdir_J, dip_J = pole_xy_to_plane_orientation(Xp, Yp)
            if not (phi_used < dip_J < slope_dip):
                return False
            if abs(circular_diff(dipdir_J, slope_dipdir)) > dipdir_tolerance_plane:
                return False
            return True

        def in_plane_orthotilt_zone_pole(Xp, Yp):
            if not show_plane_orthotilt:
                return False
            if Xp*Xp + Yp*Yp > 1.0:
                return False
            dipdir_J, dip_J = pole_xy_to_plane_orientation(Xp, Yp)
            # 1) Steilheit / Reibung
            if not (phi_used < dip_J < slope_dip):
                return False
            # 2) keine "Rückwärts"-Situation
            if abs(circular_diff(dipdir_J, slope_dipdir)) > 90.0:
                return False
            # 3) zusätzlicher Orthotilt-Winkel
            trend_pole_J, plunge_pole_J = plane_pole_from_dipdir(dipdir_J, dip_J)
            nx_J, ny_J, nz_J = trend_plunge_to_xyz(trend_pole_J, plunge_pole_J)
            angle_extra = min_angle_to_family(nx_J, ny_J, nz_J)
            return angle_extra <= max_orthotilt

        def in_wedge_zone_pole(Xp, Yp):
            if not show_wedge:
                return False
            if wedge_poly_path is None:
                return False
            r2 = Xp*Xp + Yp*Yp
            if r2 > 1.0:
                return False
            if not wedge_poly_path.contains_point((Xp, Yp)):
                return False
            r_f2 = friction_radius2(phi_used)
            if r2 <= r_f2:
                return False
            return True

        def in_toppling_zone_pole(Xp, Yp):
            if not show_toppling:
                return False
            if Xp*Xp + Yp*Yp > 1.0:
                return False
            dipdir_J, dip_J = pole_xy_to_plane_orientation(Xp, Yp)
            dip_min = (90.0 - slope_dip) + phi_used
            if dip_J <= dip_min:
                return False
            dipdir_opposite = (slope_dipdir + 180.0) % 360.0
            if abs(circular_diff(dipdir_J, dipdir_opposite)) > dipdir_tolerance_toppling:
                return False
            return True

        return (
            in_plane_zone_pole,
            in_plane_orthotilt_zone_pole,
            in_wedge_zone_pole,
            in_toppling_zone_pole
        )

    figs_pairs = []

    n = len(df)

    # Sammeln von Infos pro Plot für die Zusammenfassung
    rows_summary = []          # für die Tabelle
    figs_pairs_extended = []   # wie figs_pairs, aber mit "kritisch"-Flag

    def is_between_az(a, b, x):
        """
        Prüft, ob Azimut x (in Grad) auf dem kürzesten Bogen
        zwischen a und b liegt (exklusive Gleichheit).
        a, b, x : Grad (0..360), werden intern zirkulär behandelt.
        """
        a = a % 360.0
        b = b % 360.0
        x = x % 360.0

        # Richtung von a nach b (kürzester Weg), signed in [-180, 180]
        ab = circular_diff(b, a)
        ax = circular_diff(x, a)

        if ab > 0:
            # b liegt "vor" a im Uhrzeigersinn
            return 0.0 < ax < ab
        elif ab < 0:
            # b liegt "vor" a gegen den Uhrzeigersinn
            return 0.0 > ax > ab
        else:
            # a und b identisch -> "zwischen" ist nicht definiert
            return False

    # ---------------------------------------------------
    # 5) Über alle Paarkombinationen (i < j) iterieren
    # ---------------------------------------------------
    for i in range(n):
        name_i = str(df.loc[i, "name"])
        dip_i = float(df.loc[i, "dip"])
        dipdir_i = float(df.loc[i, "dipdir"]) % 360.0
        phi_i = float(df.loc[i, "friction_angle"])

        color_i = color_cycle[i % len(color_cycle)]

        for j in range(i + 1, n):
            name_j = str(df.loc[j, "name"])
            dip_j = float(df.loc[j, "dip"])
            dipdir_j = float(df.loc[j, "dipdir"]) % 360.0
            phi_j = float(df.loc[j, "friction_angle"])

            color_j = color_cycle[j % len(color_cycle)]

            # --- Schnittlineation berechnen ---
            # Normale = Polvektor der Ebene
            trend_pole_i, plunge_pole_i = plane_pole_from_dipdir(dipdir_i, dip_i)
            nx1, ny1, nz1 = trend_plunge_to_xyz(trend_pole_i, plunge_pole_i)

            trend_pole_j, plunge_pole_j = plane_pole_from_dipdir(dipdir_j, dip_j)
            nx2, ny2, nz2 = trend_plunge_to_xyz(trend_pole_j, plunge_pole_j)

            vx = ny1 * nz2 - nz1 * ny2
            vy = nz1 * nx2 - nx1 * nz2
            vz = nx1 * ny2 - ny1 * nx2

            norm_v = math.sqrt(vx * vx + vy * vy + vz * vz)
            if norm_v < 1e-6:
                # nahezu parallel -> keine sinnvolle Schnittlineation
                intersection_exists = False
            else:
                intersection_exists = True
                vx /= norm_v
                vy /= norm_v
                vz /= norm_v
                if vz > 0:
                    vx, vy, vz = -vx, -vy, -vz
                trend_int, plunge_int = xyz_to_trend_plunge(vx, vy, vz)

                # Schnittlineation im Schmidt-Netz
                X_int, Y_int = project_trend_plunge(trend_int, plunge_int)

                # "Pol" der Schnittlineation (deine Definition)
                trend_int_pole = (trend_int + 180.0) % 360.0
                plunge_int_pole = 90.0 - plunge_int
                Xp_int, Yp_int = project_trend_plunge(trend_int_pole, plunge_int_pole)

            # Für dieses Paar jetzt Plots pro *unterschiedlichem* Reibungswinkel:
            # - wenn phi_i == phi_j → nur EIN Plot
            # - sonst → zwei Plots (phi_i und phi_j)
            phi_variants = [(phi_i, "phi_i")]
            if abs(phi_j - phi_i) > 1e-6:
                phi_variants.append((phi_j, "phi_j"))

            for phi_used, phi_label in phi_variants:
                # Standard-Annahme: unkritisch, bis das Gegenteil bewiesen wird
                is_critical = False
                failure_summary = ""

                # Zonenprüfer für diesen Reibungswinkel
                (
                    in_plane_pole,
                    in_plane_orthotilt_pole,
                    in_wedge_pole,
                    in_toppling_pole
                ) = build_zone_checkers(phi_used)

                # Basis-Schmidt-Netz
                fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)

                # -----------------------------
                # Titel (Hang + beide Trennflächen)
                # -----------------------------
                main_title = "Stabilitätsanalyse - Kritische Zonen"
                subtitle = (
                    f"Hang: ({slope_dip:.1f}°/{slope_dipdir:05.1f}°)\n"
                    f"Trennflächen: {name_i} ({dip_i:.1f}°/{dipdir_i:05.1f}°, φ={phi_i:.1f}°), "
                    f"{name_j} ({dip_j:.1f}°/{dipdir_j:05.1f}°, φ={phi_j:.1f}°)\n"
                    f"Aktuell betrachteter Reibungswinkel: {phi_used:.1f}°"
                )

                fig.text(
                    0.5, 1.0,
                    main_title,
                    ha="center", va="top",
                    fontsize=14,
                    fontweight="bold"
                )

                fig.text(
                    0.5, 0.96,
                    subtitle,
                    ha="center", va="top",
                    fontsize=10
                )

                # -----------------------------
                # Zonen für diesen φ zeichnen
                # -----------------------------
                slope_drawn = False

                if show_plane:
                    plot_plane_failure_zone(
                        slope_dipdir=slope_dipdir,
                        slope_dip=slope_dip,
                        friction_angle=phi_used,
                        dipdir_tolerance=dipdir_tolerance_plane,
                        grid_step=grid_step,
                        ax=ax,
                        add_title=False,
                        add_legend=False,
                        plot_slope=not slope_drawn
                    )
                    slope_drawn = True

                if show_plane_orthotilt:
                    plot_plane_failure_zone_orthotilt(
                        slope_dipdir=slope_dipdir,
                        slope_dip=slope_dip,
                        friction_angle=phi_used,
                        max_orthotilt=max_orthotilt,
                        grid_step=grid_step,
                        ax=ax,
                        add_title=False,
                        add_legend=False,
                        plot_slope=not slope_drawn
                    )
                    slope_drawn = True

                if show_wedge:
                    plot_wedge_failure_marklandsche_area(
                        slope_dipdir=slope_dipdir,
                        slope_dip=slope_dip,
                        friction_angle=phi_used,
                        grid_step=grid_step,
                        ax=ax,
                        add_title=False,
                        add_legend=False,
                        plot_slope=not slope_drawn
                    )
                    slope_drawn = True

                if show_toppling:
                    plot_toppling_failure_zone(
                        slope_dipdir=slope_dipdir,
                        slope_dip=slope_dip,
                        friction_angle=phi_used,
                        dipdir_tolerance=dipdir_tolerance_toppling,
                        grid_step=grid_step,
                        ax=ax,
                        add_title=False,
                        add_legend=False,
                        plot_slope=not slope_drawn
                    )
                    slope_drawn = True

                # -----------------------------
                # Trennflächen i & j (Großkreise + Pole)
                # -----------------------------
                # i
                Xgc_i, Ygc_i = great_circle_from_plane(dipdir_i, dip_i)
                ax.plot(Xgc_i, Ygc_i,
                        color=color_i,
                        linewidth=2.0)
                Xi_pole, Yi_pole = project_trend_plunge(trend_pole_i, plunge_pole_i)
                ax.plot(Xi_pole, Yi_pole,
                        marker='o',
                        markersize=10,
                        linestyle='none',
                        markerfacecolor=color_i,
                        markeredgecolor='k')

                # j
                Xgc_j, Ygc_j = great_circle_from_plane(dipdir_j, dip_j)
                ax.plot(Xgc_j, Ygc_j,
                        color=color_j,
                        linewidth=2.0)
                Xj_pole, Yj_pole = project_trend_plunge(trend_pole_j, plunge_pole_j)
                ax.plot(Xj_pole, Yj_pole,
                        marker='o',
                        markersize=10,
                        linestyle='none',
                        markerfacecolor=color_j,
                        markeredgecolor='k')

                # -----------------------------
                # Schnittlineation + 'Pol'
                # -----------------------------
                zone_hits = []

                if intersection_exists:
                    # Schnittlineation (bifarbiges Kreuz)
                    plot_bicolor_cross(
                        ax, X_int, Y_int,
                        color1=color_i, color2=color_j,
                        size=0.03,
                        linewidth=2.0
                    )

                    # 'Pol' der Schnittlineation (bifarbiges Quadrat)
                    plot_bicolor_square(
                        ax, Xp_int, Yp_int,
                        color1=color_i, color2=color_j,
                        size=0.03,
                        edgecolor='k',
                        linewidth=1.5
                    )

                    # Zonenprüfung für den 'Pol' der Schnittlineation
                    if in_plane_pole(Xp_int, Yp_int):
                        zone_hits.append("Plane failure")
                    if in_plane_orthotilt_pole(Xp_int, Yp_int):
                        zone_hits.append("Plane failure")
                    if in_wedge_pole(Xp_int, Yp_int):
                        zone_hits.append("Wedge failure")
                    if in_toppling_pole(Xp_int, Yp_int):
                        zone_hits.append("Toppling failure")

                    if zone_hits:
                        # -------------------------------------------------
                        # 1) Grund-Warntext (unabhängig von Mechanismus)
                        # -------------------------------------------------
                        if len(zone_hits) == 1:
                            base_txt = (
                                f"Warnung: 'Pol' der Schnittlineation "
                                f"({name_i} & {name_j}) liegt in der "
                                f"folgenden kritischen Zone:\n- {zone_hits[0]}"
                            )
                        else:
                            base_txt = (
                                f"Warnung: 'Pol' der Schnittlineation "
                                f"({name_i} & {name_j}) liegt in folgenden "
                                f"kritischen Zonen:\n- " + "\n- ".join(zone_hits)
                            )

                        # ab hier zunächst als kritisch zählen
                        is_critical = True
                        failure_summary = ", ".join(sorted(set(zone_hits)))

                        # -------------------------------------------------
                        # 2) plane vs wedge anhand der Fallrichtungen
                        # -------------------------------------------------
                        between_names = []
                        if is_between_az(slope_dipdir, trend_int, dipdir_i):
                            between_names.append(name_i)
                        if is_between_az(slope_dipdir, trend_int, dipdir_j):
                            between_names.append(name_j)

                        has_between = bool(between_names)

                        if has_between:
                            if len(between_names) == 1:
                                flabel = f"der Trennfläche {between_names[0]}"
                            else:
                                flabel = (
                                    "der Trennflächen " +
                                    " und ".join(between_names)
                                )

                            # Standard-Formulierung für "normalen" plane failure
                            plane_txt_normal = (
                                "\n\nAchtung: Die Fallrichtung "
                                f"{flabel} liegt zwischen der "
                                "Fallrichtung des Hanges \n und der "
                                f"Schnittlineation der Flächen {name_i} "
                                f"und {name_j}. \n Deshalb kommt es hier "
                                "zum plane failure auf dieser Trennfläche."
                            )
                        else:
                            plane_txt_normal = ""

                        # Standard-Formulierung für reinen Wedge-Fall
                        if not has_between:
                            wedge_txt = (
                                "\n\nErgänzung: Keine der Fallrichtungen "
                                "der beiden Trennflächen liegt zwischen "
                                "der Fallrichtung des Hanges \n und der "
                                f"Schnittlineation der Flächen {name_i} "
                                f"und {name_j}. \n Deshalb kommt es hier "
                                "zum Wedge failure."
                            )
                        else:
                            wedge_txt = ""

                        # -------------------------------------------------
                        # 3) Geometrische Möglichkeit von Wedge-Failure
                        # -------------------------------------------------
                        dev_i = abs(circular_diff(dipdir_i, slope_dipdir)) > 90.0
                        dev_j = abs(circular_diff(dipdir_j, slope_dipdir)) > 90.0
                        wedge_possible = not (dev_i or dev_j)

                        wedge_in_hits = ("Wedge failure" in zone_hits)
                        only_wedge = wedge_in_hits and all(
                            z == "Wedge failure" for z in zone_hits
                        )

                        # Standard: rote Warnung
                        warn_color = "red"

                        # -------------------------------------------------
                        # 4) Spezialfall:
                        #    Nur Wedge-Zone getroffen, Wedge geometrisch
                        #    unmöglich, UND eine Fallrichtung liegt
                        #    zwischen Hang und Schnittlineation.
                        #    -> Haupttext grün, plane-Hinweis rot
                        # -------------------------------------------------
                        if only_wedge and not wedge_possible and has_between:
                            offenders = []
                            if dev_i:
                                offenders.append(name_i)
                            if dev_j:
                                offenders.append(name_j)

                            if len(offenders) == 1:
                                zusatz = (
                                    "\n\nAber Entwarnung: Die Fallrichtung der "
                                    f"Trennfläche {offenders[0]} weicht um mehr "
                                    "als 90° von der Fallrichtung des Hanges ab.\n"
                                    "Geometrisch ist hier kein Wedge failure möglich."
                                )
                            else:
                                zusatz = (
                                    "\n\nAber Entwarnung: Die Fallrichtungen der "
                                    f"Trennflächen {offenders[0]} und {offenders[1]} "
                                    "weichen um mehr als 90° von der "
                                    "Fallrichtung des Hanges ab.\n"
                                    "Geometrisch ist hier kein Wedge failure möglich."
                                )

                            txt_main = base_txt + zusatz

                            plane_txt_special = (
                                "\n\nAchtung: Die Fallrichtung "
                                f"{flabel} liegt zwischen der "
                                "Fallrichtung des Hanges \n und der "
                                f"Schnittlineation der Flächen {name_i} "
                                f"und {name_j}. \n Hier ist in der Einzelanalyse "
                                "zu untersuchen, ob es zum plane failure "
                                "auf dieser Trennfläche kommen kann."
                            )

                            # >>> NEU: korrekte Zusammenfassungsbeschreibung
                            if len(between_names) == 1:
                                failure_summary = (
                                    f"Plane failure auf Trennfläche {between_names[0]} ist zu überprüfen"
                                )
                            else:
                                failure_summary = (
                                    "Plane failure auf Trennflächen "
                                    + " und ".join(between_names)
                                    + " ist zu überprüfen"
                                )
                            # <<< ENDE NEU

                            # Haupttext (grün)
                            ax.text(
                                0.5, -0.25,
                                txt_main,
                                transform=ax.transAxes,
                                ha="center", va="top",
                                fontsize=11,
                                fontweight="bold",
                                color="darkgreen"
                            )

                            # Plane-Hinweis (rot) etwas darunter
                            ax.text(
                                0.5, -0.38,
                                plane_txt_special,
                                transform=ax.transAxes,
                                ha="center", va="top",
                                fontsize=11,
                                fontweight="bold",
                                color="red"
                            )


                        # -------------------------------------------------
                        # 5) Zweiter Spezialfall:
                        #    Nur Wedge-Zone, Wedge geometrisch unmöglich,
                        #    aber KEINE Fallrichtung zwischen Hang & Linie
                        #    → alles grün (reine Entwarnung)
                        # -------------------------------------------------
                        elif only_wedge and not wedge_possible and not has_between:
                            offenders = []
                            if dev_i:
                                offenders.append(name_i)
                            if dev_j:
                                offenders.append(name_j)

                            if len(offenders) == 1:
                                zusatz = (
                                    "\n\nAber Entwarnung: Die Fallrichtung der "
                                    f"Trennfläche {offenders[0]} weicht um mehr "
                                    "als 90° von der Fallrichtung \n des Hanges ab. "
                                    "Geometrisch ist hier kein Wedge failure möglich."
                                )
                            else:
                                zusatz = (
                                    "\n\nAber Entwarnung: Die Fallrichtungen der "
                                    f"Trennflächen {offenders[0]} und {offenders[1]} "
                                    "weichen um mehr als 90° von der "
                                    "Fallrichtung \n des Hanges ab. "
                                    "Geometrisch ist hier kein Wedge failure möglich."
                                )

                            txt = base_txt + wedge_txt + zusatz

                            ax.text(
                                0.5, -0.25,
                                txt,
                                transform=ax.transAxes,
                                ha="center", va="top",
                                fontsize=11,
                                fontweight="bold",
                                color="darkgreen"
                            )

                            # in diesem Fall wirklich unkritisch
                            is_critical = False
                            failure_summary = "—"

                        # -------------------------------------------------
                        # 6) Normalfall:
                        #    Wedge möglich ODER mehrere Zonen
                        #    -> alles wie gehabt, in rot
                        # -------------------------------------------------
                        else:
                            txt = base_txt + plane_txt_normal + wedge_txt
                            ax.text(
                                0.5, -0.25,
                                txt,
                                transform=ax.transAxes,
                                ha="center", va="top",
                                fontsize=11,
                                fontweight="bold",
                                color=warn_color
                            )

                    else:
                        # FALL (2): Schnittlineation existiert, aber Pol NICHT in kritischer Zone
                        txt = (
                            f"Unkritisch: 'Pol' der Schnittlineation "
                            f"({name_i} & {name_j}) liegt in keiner "
                            f"kritischen Zone."
                        )
                        ax.text(
                            0.5, -0.25,
                            txt,
                            transform=ax.transAxes,
                            ha="center", va="top",
                            fontsize=11,
                            fontweight="bold",
                            color="darkgreen"
                        )

                        is_critical = False
                        failure_summary = ""

                else:
                    # FALL (3): Flächen nahezu parallel → keine definierte Schnittlineation
                    txt = (
                        f"Unkritisch: Trennflächen {name_i} & {name_j} sind "
                        f"nahezu parallel – keine eindeutige Schnittlineation."
                    )
                    ax.text(
                        0.5, -0.25,
                        txt,
                        transform=ax.transAxes,
                        ha="center", va="top",
                        fontsize=11,
                        fontweight="bold",
                        color="darkgreen"
                    )
                    is_critical = False
                    failure_summary = ""

                # -----------------------------
                # Legende
                # -----------------------------
                legend_handles = []

                # Hangfläche & Pol (in den Zonen-Funktionen gezeichnet)
                legend_handles.append(
                    Line2D([0], [0],
                           color="black",
                           linewidth=3.0,
                           label="Hangfläche (Großkreis)")
                )
                legend_handles.append(
                    Line2D([0], [0],
                           marker="o",
                           linestyle="none",
                           markersize=12,
                           markerfacecolor="black",
                           markeredgecolor="black",
                           label="Pol der Hangfläche")
                )

                # Trennflächen i & j
                legend_handles.append(
                    Line2D([0], [0],
                           color=color_i,
                           linewidth=2.0,
                           label=f"Trennfläche {name_i} (Großkreis)")
                )
                legend_handles.append(
                    Line2D([0], [0],
                           marker="o",
                           linestyle="none",
                           markersize=10,
                           markerfacecolor=color_i,
                           markeredgecolor="k",
                           label=f"Pol Trennfläche {name_i}")
                )

                legend_handles.append(
                    Line2D([0], [0],
                           color=color_j,
                           linewidth=2.0,
                           label=f"Trennfläche {name_j} (Großkreis)")
                )
                legend_handles.append(
                    Line2D([0], [0],
                           marker="o",
                           linestyle="none",
                           markersize=10,
                           markerfacecolor=color_j,
                           markeredgecolor="k",
                           label=f"Pol Trennfläche {name_j}")
                )

                # Schnittlineation / 'Pol'
                legend_handles.append(
                    Line2D([0], [0],
                           marker="x",
                           linestyle="none",
                           markersize=10,
                           color="k",
                           label="Schnittlineation")
                )
                legend_handles.append(
                    Line2D([0], [0],
                           marker="s",
                           linestyle="none",
                           markersize=10,
                           markerfacecolor="none",
                           markeredgecolor="k",
                           label="'Pol' der Schnittlineation")
                )

                # Zonen
                if show_plane:
                    legend_handles.append(
                        Patch(facecolor='none',
                              edgecolor='red',
                              linestyle='--',
                              hatch='-',
                              label="Plane failure – kritische Zone")
                    )
                if show_plane_orthotilt:
                    legend_handles.append(
                        Patch(facecolor='none',
                              edgecolor='red',
                              linestyle='--',
                              hatch='-',
                              label="Plane failure – kritische Zone")
                    )
                if show_wedge:
                    legend_handles.append(
                        Patch(facecolor='none',
                              edgecolor='red',
                              linestyle='--',
                              hatch='|',
                              label="Wedge failure - kritische Zone")
                    )
                if show_toppling:
                    legend_handles.append(
                        Patch(facecolor='none',
                              edgecolor='red',
                              linestyle='--',
                              hatch='/',
                              label="Toppling failure – kritische Zone")
                    )

                ax.legend(
                    handles=legend_handles,
                    loc="upper left",
                    bbox_to_anchor=(1.5, 1.15),
                    handlelength=6.0,
                    handleheight=6.0,
                    borderpad=0.8,
                    labelspacing=2.0,
                    markerscale=1.5
                )

                # -----------------------------
                # Speichern (optional)
                # -----------------------------
                if save_prefix is not None:
                    safe_i = "".join(
                        c if c.isalnum() or c in "._-" else "_" for c in name_i
                    )
                    safe_j = "".join(
                        c if c.isalnum() or c in "._-" else "_" for c in name_j
                    )
                    filename = (
                        f"{save_prefix}_pair_{i+1}_{j+1}_"
                        f"{phi_label}_{safe_i}_{safe_j}.png"
                    )
                    fig.savefig(filename, dpi=dpi, bbox_inches="tight")

                # -----------------------------
                # Summary-Infos für Tabelle & Sortierung sammeln
                # -----------------------------
                rows_summary.append({
                    "kritisch": is_critical,
                    "Status": "kritisch" if is_critical else "unkritisch",
                    "Failure": failure_summary if failure_summary else "—",
                    "Trennfläche_i": name_i,
                    "Trennfläche_j": name_j,
                    "phi_used": phi_used,
                })

                figs_pairs_extended.append(
                    dict(
                        name_i=name_i,
                        name_j=name_j,
                        phi_used=phi_used,
                        fig=fig,
                        ax=ax,
                        kritisch=is_critical
                    )
                )

    # ---------------------------------------------------
    # 6) Zusammenfassungs-Tabelle & Sortierung der Figuren
    # ---------------------------------------------------
    if rows_summary:
        df_summary = pd.DataFrame(rows_summary)

        # Kritische zuerst, dann unkritische
        df_summary = df_summary.sort_values(
            by="kritisch", ascending=False
        ).reset_index(drop=True)

        # leichte Rot-/Grün-Hinterlegung
        def _color_row(row):
            if row["kritisch"]:
                color = "background-color: rgba(255, 0, 0, 0.15)"  # leicht rot
            else:
                color = "background-color: rgba(0, 255, 0, 0.15)"  # leicht grün
            return [color] * len(row)

        df_styled = df_summary.style.apply(_color_row, axis=1)

        # Tabelle vor der Rückgabe anzeigen (im Notebook hübsch)
        try:
            from IPython.display import display
            display(df_styled)
        except Exception:
            # Fallback: plain print ohne Farben
            print(df_summary)

        # Figuren in der gleichen Logik sortieren:
        # erst kritisch (True), dann unkritisch (False)
        figs_pairs_extended.sort(
            key=lambda d: int(not d["kritisch"])
        )

        # zurück zur alten Struktur: (name_i, name_j, phi_used, fig, ax)
        figs_pairs = [
            (d["name_i"], d["name_j"], d["phi_used"], d["fig"], d["ax"])
            for d in figs_pairs_extended
        ]
    else:
        # kein Paar → leere Liste
        figs_pairs = []

    return figs_pairs




# -------------------------------------------------------
# Inverse Lambert-Projektion: (X,Y) -> 3D-Vektor
# -------------------------------------------------------
def schmidt_xy_to_xyz(X, Y):
    """
    Inverse zur Lambert-Gleichflächenprojektion der unteren Hemisphäre.
    Nimmt Punkte (X,Y) im Einheitskreis und gibt den zugehörigen
    Einheitsvektor (x,y,z) zurück.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    r2 = X**2 + Y**2
    denom2 = 2.0 - r2          # immer > 0 für r <= 1
    denom2 = np.where(denom2 <= 0, np.nan, denom2)
    denom = np.sqrt(denom2)

    x = X * denom
    y = Y * denom
    z = r2 - 1.0

    norm = np.sqrt(x*x + y*y + z*z)
    norm = np.where(norm == 0, np.nan, norm)

    return x / norm, y / norm, z / norm




# -------------------------------------------------------
# Zweifarbiger Pfeil am Außenkreis (für Schnittlineare)
# -------------------------------------------------------
def plot_bicolor_arrow_on_rim(ax, az_deg, color1, color2,
                              r_inner=1.25, r_outer=1.5,
                              linewidth=3.0):
    """
    Zeichnet einen zweifarbigen Pfeil am Außenkreis für eine gegebene Azimutrichtung.

    az_deg : Azimut der Linie (0°=N, 90°=E)
    color1 : Farbe des inneren Schaftsegments
    color2 : Farbe des äußeren Schaftsegments + Pfeilspitze
    """
    angle = np.deg2rad(az_deg)

    # Start / Ende / Mitte entlang des Pfeils
    x1 = r_inner * np.sin(angle)
    y1 = r_inner * np.cos(angle)
    x2 = r_outer * np.sin(angle)
    y2 = r_outer * np.cos(angle)
    r_mid = 0.5 * (r_inner + r_outer)
    xmid = r_mid * np.sin(angle)
    ymid = r_mid * np.cos(angle)

    # Inneres Segment (nur Linie)
    ax.plot([x1, xmid], [y1, ymid],
            color=color1,
            linewidth=linewidth)

    # Äußeres Segment mit Pfeilspitze
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(xmid, ymid),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=linewidth,
            color=color2
        )
    )


# -------------------------------------------------------
# Tabelle mit Trennflächen: name, dip, dipdir, friction
# -------------------------------------------------------
def load_discontinuity_table(path):
    """
    Liest eine Tabelle mit:
        Spalte 1: Name der Trennfläche
        Spalte 2: Dip (Fallen) [Grad]
        Spalte 3: Dipdir (Fallrichtung) [Grad]
        Spalte 4: Reibungswinkel phi [Grad]

    CSV oder Excel werden je nach Endung automatisch erkannt.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.csv', '.txt']:
        df = pd.read_csv(path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unbekanntes Dateiformat: {ext}")

    if df.shape[1] < 4:
        raise ValueError("Erwarte mindestens 4 Spalten: name, dip, dipdir, friction.")

    out = pd.DataFrame({
        "name": df.iloc[:, 0].astype(str),
        "dip": df.iloc[:, 1].astype(float),
        "dipdir": df.iloc[:, 2].astype(float),
        "friction": df.iloc[:, 3].astype(float)
    })
    return out


# -------------------------------------------------------
# Hang + Trennflächen + Schnittlineare (+ Reibungsbereich)
# -------------------------------------------------------
def plot_slope_and_discontinuities_from_table(
        slope_dip_deg,
        slope_dipdir_deg,
        friction_angle_deg,   # globaler Reibungswinkel
        path,
        grid_step=10,
        figsize=(10, 10),
        cross_size=0.05,
        cross_linewidth=3.0,
        arrow_r_inner=1.25,
        arrow_r_outer=1.5,
        arrow_linewidth=3.0,
        save_fig=None,
        save_all_csv=None,    # ⬅ NEU
        dpi=300):

    """
    Zeichnet ein Schmidt-Netz mit:
      - Großkreis des Hanges (fett schwarz) + dicker Strich am Außenkreis
      - Großkreise aller Trennflächen (verschiedene Farben) + Strich am Außenkreis
      - Alle Schnittlineare zwischen allen Trennflächen als zweifarbige Kreuze
      - Für jede Schnittlineation einen zweifarbigen Pfeil am Außenkreis
        (Fallrichtung der Schnittlinie).
      - Hellgrün transparenter „unkritischer Bereich“ aufgrund des
        Reibungswinkels φ (friction_angle_deg).

    WICHTIG: Parameterreihenfolge für den Hang:
      slope_dip_deg    : Fallen des Hanges [Grad]
      slope_dipdir_deg : Fallrichtung des Hanges (Dip Direction) [Grad]

    path : Pfad zur Tabelle mit
           Spalte 1: name
           Spalte 2: dip
           Spalte 3: dipdir
           Spalte 4: friction (wird für df_int weitergeführt, aber hier
                             NICHT mehr für den grünen Bereich benutzt)
    """
    from matplotlib.patches import Patch

    # --- 1. Trennflächen-Tabelle einlesen ---
    df_joints = load_discontinuity_table(path)

    # --- 2. Schmidt-Netz aufziehen ---
    fig, ax = plot_schmidt_net(grid_step=grid_step, figsize=figsize)
    fig.suptitle(f"Hang + Trennflächen + Schnittlineare – {os.path.basename(path)}",
                 y=0.98)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)

    # ---------------------------------------------------
    # 2b. Reibungswinkel-Bereich (hellgrün, transparent)
    # ---------------------------------------------------
    friction_patch = None  # für die Legende später

    if friction_angle_deg is not None and 0.0 < friction_angle_deg < 90.0:
        phi = float(friction_angle_deg)

        # --- Radien für die Dip-Grenzen ---
        # Vorderseite (±90° um Hangfallrichtung):
        # unkritisch, wenn Dip > 90° - φ  -> Kreis bei Dip_front = 90° - φ
        dip_front = 90.0 - phi
        plunge_pole_front = 90.0 - dip_front  # = φ
        Xf, Yf = project_trend_plunge(0.0, plunge_pole_front)
        r_front = np.sqrt(Xf**2 + Yf**2)

        # Rückseite (andere Hälfte):
        # Grenz-Dip: φ + (90° - Fallen des Hangs)
        # Auf dieser Rückseite sollen "kritisch" und "unkritisch"
        # im Vergleich zur alten Version vertauscht werden.
        dip_back = 90-(phi + (90.0 - float(slope_dip_deg)))
        plunge_pole_back = 90.0 - dip_back
        Xb, Yb = project_trend_plunge(0.0, plunge_pole_back)
        r_back = np.sqrt(Xb**2 + Yb**2)

        # Raster im Einheitskreis
        grid_n = 300
        xg = np.linspace(-1.0, 1.0, grid_n)
        yg = np.linspace(-1.0, 1.0, grid_n)
        Xg, Yg = np.meshgrid(xg, yg)
        Rg = np.sqrt(Xg**2 + Yg**2)
        mask_disc = Rg <= 1.0

        # Azimut des Punktes im Schmidt-Netz (0°=N, 90°=E usw.)
        az = np.degrees(np.arctan2(Xg[mask_disc], Yg[mask_disc])) % 360.0

        # ±90°-Sektor um die Hangfallrichtung (Fallrichtung des Hanges)
        d_az = (az - float(slope_dipdir_deg) + 540.0) % 360.0 - 180.0
        in_sector = np.abs(d_az) <= 90.0   # "Vorderseite" des Hanges

        r_here = Rg[mask_disc]

        # Vorderseite: unkritisch, wenn Dip > 90° - φ -> r >= r_front
        uncrit_front = in_sector & (r_here >= r_front)

        # Rückseite: unkritisch, wenn Dip < φ -> r <= r_back
        uncrit_back = (~in_sector) & (r_here >= r_back)

        green_flat = uncrit_front | uncrit_back

        green_map = np.full_like(Rg, np.nan, dtype=float)
        green_map[mask_disc] = np.where(green_flat, 1.0, np.nan)

        cmap_green = plt.cm.Greens.copy()
        cmap_green.set_bad(alpha=0.0)

        ax.imshow(
            green_map,
            extent=[-1.0, 1.0, -1.0, 1.0],
            origin="lower",
            cmap=cmap_green,
            alpha=0.98
        )

        # Legendeneintrag für diesen Bereich
        friction_patch = Patch(
            facecolor=plt.cm.Greens(0.6),
            edgecolor='none',
            alpha=0.98,
            label=f"Unkritischer Bereich (φ = {friction_angle_deg:.1f}°)"
        )




    # --- 3. Alle Flächen (Hang + Trennflächen) aufbauen ---
    planes = []

    # 3a) Hang
    planes.append(dict(
        name="Hang",
        dip=float(slope_dip_deg),
        dipdir=float(slope_dipdir_deg),
        friction=np.nan,
        color="black",
        kind="slope"
    ))

    # 3b) Trennflächen
    n_colors = len(BASE_COLORS)
    for idx, row in df_joints.iterrows():
        color = BASE_COLORS[idx % n_colors]
        planes.append(dict(
            name=str(row["name"]),
            dip=float(row["dip"]),
            dipdir=float(row["dipdir"]),
            friction=float(row["friction"]),
            color=color,
            kind="joint"
        ))

    # --- 4. Großkreise + Striche am Außenkreis ---
    for plane in planes:
        dip = plane["dip"]
        dipdir = plane["dipdir"]
        color = plane["color"]
        kind = plane["kind"]
        name = plane["name"]

        # Großkreis der Fläche
        Xgc, Ygc = great_circle_from_plane(dipdir, dip)

        if kind == "slope":
            # Hang: fetter schwarzer Großkreis
            ax.plot(Xgc, Ygc,
                    color="black",
                    linewidth=2.5,
                    label=name)
            line_color = "black"
        else:
            # Trennflächen: farbige Großkreise
            ax.plot(Xgc, Ygc,
                    color=color,
                    linewidth=1.8,
                    label=name)
            line_color = color

        # Strich am Außenkreis in Fallrichtung
        angle = np.deg2rad(dipdir)
        r1 = 1.25
        r2 = 1.5
        x1 = r1 * np.sin(angle)
        y1 = r1 * np.cos(angle)
        x2 = r2 * np.sin(angle)
        y2 = r2 * np.cos(angle)
        ax.plot([x1, x2], [y1, y2],
                color=line_color,
                linewidth=3.0)

    # --- 5. Normalenvektoren vorberechnen ---
    for plane in planes:
        dipdir = plane["dipdir"]
        dip = plane["dip"]
        trend_pole, plunge_pole = plane_pole_from_dipdir(dipdir, dip)
        nx, ny, nz = trend_plunge_to_xyz(trend_pole, plunge_pole)
        plane["nvec"] = np.array([nx, ny, nz])

    # --- 6. Schnittlineare zwischen allen Trennflächen (ohne Hang) ---
    intersection_records = []
    n_planes = len(planes)

    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            p1 = planes[i]
            p2 = planes[j]

            # nur Trennflächen untereinander (kind == "joint")
            if not (p1["kind"] == "joint" and p2["kind"] == "joint"):
                continue

            n1 = p1["nvec"]
            n2 = p2["nvec"]

            # Schnittlinie = Kreuzprodukt der Normalen
            l = np.cross(n1, n2)
            norm_l = np.linalg.norm(l)
            if norm_l < 1e-6:
                # nahezu parallel -> keine definierte Schnittlinie
                continue
            l = l / norm_l

            # Auf untere Hemisphäre bringen
            if l[2] > 0:
                l = -l

            lx, ly, lz = l
            trend_int, plunge_int = xyz_to_trend_plunge(lx, ly, lz)

            # Winkel zwischen den Flächen (0..90°)
            dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
            theta = np.degrees(np.arccos(dot))
            angle_planes = min(theta, 180.0 - theta)

            # 2D-Projektion der Schnittlinie
            Xint, Yint = project_trend_plunge(trend_int, plunge_int)

            # Zweifarbiges Kreuz an der Schnittlineation
            plot_bicolor_cross(ax, Xint, Yint,
                               p1["color"], p2["color"],
                               size=cross_size,
                               linewidth=cross_linewidth)

            # Zweifarbiger Pfeil am Außenkreis (Fallrichtung der Linie)
            plot_bicolor_arrow_on_rim(
                ax,
                trend_int,
                p1["color"],
                p2["color"],
                r_inner=arrow_r_inner,
                r_outer=arrow_r_outer,
                linewidth=arrow_linewidth
            )

            intersection_records.append(dict(
                plane_i=p1["name"],
                plane_j=p2["name"],
                dip_i=p1["dip"],
                dipdir_i=p1["dipdir"],
                friction_i=p1["friction"],
                dip_j=p2["dip"],
                dipdir_j=p2["dipdir"],
                friction_j=p2["friction"],
                trend_int=trend_int,
                plunge_int=plunge_int,
                angle_planes_deg=angle_planes
            ))


    # --- 7. Tabellen erzeugen: Schnittlineare & Gesamttabelle ---
    # 7a) Schnittlineations-Tabelle (wie bisher zurückgegeben)
    df_int = pd.DataFrame(intersection_records)

    # 7b) Gesamttabelle mit Hang, Trennflächen und Schnittlinearen
    records_all = []

    # Hang + Trennflächen (Flächen-Daten)
    for plane in planes:
        records_all.append(dict(
            kind=plane["kind"],          # 'slope' oder 'joint'
            name=plane["name"],
            dip=plane["dip"],
            dipdir=plane["dipdir"],
            friction=plane["friction"],  # Hang: NaN
            plane_i=np.nan,
            dip_i=np.nan,
            dipdir_i=np.nan,
            friction_i=np.nan,
            plane_j=np.nan,
            dip_j=np.nan,
            dipdir_j=np.nan,
            friction_j=np.nan,
            trend_int=np.nan,
            plunge_int=np.nan,
            angle_planes_deg=np.nan
        ))

    # Schnittlineare (Linien-Daten)
    for rec in intersection_records:
        records_all.append(dict(
            kind="intersection",
            name=f"{rec['plane_i']} – {rec['plane_j']}",
            dip=np.nan,
            dipdir=np.nan,
            friction=np.nan,
            plane_i=rec["plane_i"],
            dip_i=rec["dip_i"],
            dipdir_i=rec["dipdir_i"],
            friction_i=rec["friction_i"],
            plane_j=rec["plane_j"],
            dip_j=rec["dip_j"],
            dipdir_j=rec["dipdir_j"],
            friction_j=rec["friction_j"],
            trend_int=rec["trend_int"],
            plunge_int=rec["plunge_int"],
            angle_planes_deg=rec["angle_planes_deg"]
        ))

    df_all = pd.DataFrame(records_all)

    # optional als CSV speichern
    if save_all_csv is not None:
        df_all.to_csv(save_all_csv, index=False)


    # --- 7. Legende ---
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()

    # grüner Bereich wegen φ
    if friction_patch is not None:
        handles.append(friction_patch)
        labels.append(friction_patch.get_label())

    if intersection_records:
        # Symbolerklärung Schnittlineation (Kreuz)
        cross_handle = Line2D(
            [0], [0],
            marker='x',
            linestyle='none',
            markersize=8,
            color='k',
            label="Schnittlineation"
        )

        # Symbolerklärung Pfeil am Außenkreis
        arrow_handle = Line2D(
            [0], [1],
            linestyle='-',
            marker='>',
            markersize=8,
            color='k',
            label="Fallrichtung (Pfeil am Außenkreis)"
        )

        handles.extend([cross_handle, arrow_handle])
        labels.extend([
            cross_handle.get_label(),
            arrow_handle.get_label()
        ])

    if handles:
        ax.legend(handles, labels,
                  loc='upper left',
                  bbox_to_anchor=(1.05, 1.0))

    # --- 8. Optional speichern ---
    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")

    # DataFrame mit Schnittinfos (falls du später damit weiterrechnen willst)
    df_int = pd.DataFrame(intersection_records)
    return df_int, (fig, ax)
