#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from pathlib import Path

import sys, os

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    os.system("pip install rosbags --break-system-packages -q")
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

# ── Configuración ────────────────────────────────────────────────────────────
# Carpeta donde está lidar.py
BASE_DIR = Path(__file__).resolve().parent

# Carpeta del rosbag
BAG_PATH = BASE_DIR / "dataset_ROS"
SCAN_TOPIC = '/scan'

# ROI general: zona trasera del tractor donde vive el trailer
ROI_X_MIN =  0.05  # m
ROI_X_MAX =  0.75   # m
ROI_Y_MIN = -0.35   # m
ROI_Y_MAX =  0.35  # m

# ROI circular del poste: centro y radio en metros
POLE_CX      =  0.22  # m  centro X del círculo de búsqueda
POLE_CY      =  0.0   # m  centro Y
POLE_RADIUS  =  0.23   # m  radio del círculo
POLE_ARC_MIN =  8     # puntos mínimos para aceptar detección

# Punto de enganche tractor-trailer en coordenadas del sensor
HITCH_X = 0.22  # m  (xh, yh)
HITCH_Y = 0.0   # m
TRAILER_DIST_NOM = 0.63  # m
TRAILER_DIST_TOL = 1.50  # m  holgura para movimiento y ángulo

# DBSCAN para separar clusters en la zona del trailer
DBSCAN_EPS      = 0.12  # m
DBSCAN_MIN_PTS  = 3

# ── Carga del bag ─────────────────────────────────────────────────────────────
def load_scans(bag_path, topic=SCAN_TOPIC):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    scans = []
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            print(f"[ERROR] Tópico '{topic}' no encontrado.")
            sys.exit(1)
        for conn, ts_ns, raw in reader.messages(connections=conns):
            msg       = typestore.deserialize_cdr(raw, conn.msgtype)
            ranges    = np.array(msg.ranges, dtype=np.float32)
            angle_min = float(msg.angle_min)
            angle_inc = float(msg.angle_increment)
            r_max     = float(msg.range_max)
            r_min     = float(msg.range_min)
            angles    = angle_min + np.arange(len(ranges)) * angle_inc
            valid     = (ranges >= r_min) & (ranges < r_max)
            r_val     = ranges[valid]
            a_val     = angles[valid]
            scans.append({
                'ts':     ts_ns / 1e9,
                'x':      r_val * np.cos(a_val),
                'y':      r_val * np.sin(a_val),
                'ranges': r_val,
            })
    print(f"[OK] {len(scans)} scans cargados")
    return scans

# ── Carga ground truth y odometría ───────────────────────────────────────────
def load_ground_truth(bag_path):
    """Carga /gt_theta1, /gt_theta2 y /odom desde el bag."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    gt1, gt2, odom = [], [], []

    with Reader(bag_path) as reader:
        conns_gt1  = [c for c in reader.connections if c.topic == '/gt_theta1']
        conns_gt2  = [c for c in reader.connections if c.topic == '/gt_theta2']
        conns_odom = [c for c in reader.connections if c.topic == '/odom']
        conns_all  = conns_gt1 + conns_gt2 + conns_odom

        for conn, ts_ns, raw in reader.messages(connections=conns_all):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            t   = ts_ns / 1e9
            if conn.topic == '/gt_theta1':
                gt1.append({'ts': t, 'val': np.degrees(float(msg.data))})
            elif conn.topic == '/gt_theta2':
                gt2.append({'ts': t, 'val': np.degrees(float(msg.data))})
            elif conn.topic == '/odom':
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                # yaw desde quaternion
                siny = 2.0*(o.w*o.z + o.x*o.y)
                cosy = 1.0 - 2.0*(o.y*o.y + o.z*o.z)
                yaw  = np.degrees(np.arctan2(siny, cosy))
                odom.append({'ts': t, 'x': float(p.x), 'y': float(p.y), 'yaw': yaw})

    print(f"[OK] GT θ₁:{len(gt1)}  θ₂:{len(gt2)}  odom:{len(odom)}")
    return gt1, gt2, odom

def interp_gt(gt_list, ts_query):
    """Interpola la señal ground truth en los instantes ts_query."""
    if len(gt_list) < 2:
        return np.full(len(ts_query), np.nan)
    t_gt  = np.array([g['ts'] for g in gt_list])
    v_gt  = np.array([g['val'] for g in gt_list])
    t_gt -= t_gt[0]
    return np.interp(ts_query, t_gt, v_gt,
                     left=np.nan, right=np.nan)

# ── Filtro ROI general ────────────────────────────────────────────────────────
def apply_roi(x, y):
    mask = (
        (x >= ROI_X_MIN) & (x <= ROI_X_MAX) &
        (y >= ROI_Y_MIN) & (y <= ROI_Y_MAX)
    )
    return x[mask], y[mask]

# ── Detección del poste ───────────────────────────────────────────────────────
def detect_pole(x, y):
    """
    Busca puntos dentro del círculo (POLE_CX, POLE_CY) ± POLE_RADIUS.
    Devuelve centroide (cx, cy) del arco, o (None, None) si no hay suficientes.
    """
    dist = np.sqrt((x - POLE_CX)**2 + (y - POLE_CY)**2)
    mask = dist <= POLE_RADIUS
    xp, yp = x[mask], y[mask]
    if len(xp) < POLE_ARC_MIN:
        return None, None, xp, yp
    return float(np.mean(xp)), float(np.mean(yp)), xp, yp

# ── Ajuste de recta por RANSAC ────────────────────────────────────────────────
def fit_line_ransac(x, y):
    if len(x) < 4:
        return None, None, None
    # Si la línea es más vertical que horizontal, ajustar x = m'*y + b'
    # y después convertir, para evitar singularidades
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    try:
        ransac = RANSACRegressor(residual_threshold=0.05, min_samples=4,
                                 max_trials=300, random_state=42)
        if y_range > x_range:
            # ajuste x = f(y)
            ransac.fit(y.reshape(-1, 1), x)
            mp = ransac.estimator_.coef_[0]   # dx/dy
            bp = ransac.estimator_.intercept_
            # convertir a y = m*x + b
            if abs(mp) < 1e-9:
                return None, None, None
            m = 1.0 / mp
            b = -bp / mp
        else:
            ransac.fit(x.reshape(-1, 1), y)
            m = ransac.estimator_.coef_[0]
            b = ransac.estimator_.intercept_
        return m, b, ransac.inlier_mask_
    except Exception:
        return None, None, None

# ── Ángulos según el modelo geométrico ───────────────────────────────────────
def compute_angles(cx_pole, cy_pole, m_line):
    """
    φ₁ = arctan((yp - yh) / (xp - xh))
         ángulo del vector enganche→centroide_poste respecto al eje X del tractor
         donde (xh, yh) = HITCH_X, HITCH_Y  (punto fijo de enganche)
               (xp, yp) = centroide del poste detectado

    φ₂ = φ₁ + arctan(m) - 90
         ángulo de articulación del trailer
         donde m = pendiente RANSAC de la recta del trailer
    """
    if cx_pole is None:
        return None, None
    phi1 = np.degrees(np.arctan2(cy_pole - HITCH_Y, cx_pole - HITCH_X))
    if m_line is None:
        return phi1, None
    m_deg = np.degrees(np.arctan(m_line))
    if (phi1>0):
            phi2 = 90 - abs(phi1) - abs(m_deg)
    else:
            phi2 =  -90 + abs(phi1) + abs(m_deg)
    return phi1, phi2

# ── Procesamiento de un scan ──────────────────────────────────────────────────
def process_scan(sc):
    x_all, y_all = sc['x'], sc['y']

    # 1. ROI general
    xr, yr = apply_roi(x_all, y_all)
    if len(xr) < 5:
        return None

    # 2. Detectar poste dentro de su caja espacial estricta
    cx_pole, cy_pole, xp, yp = detect_pole(xr, yr)

    # 3. Excluir puntos del poste para quedarse solo con el trailer
    pole_mask = np.sqrt((xr - POLE_CX)**2 + (yr - POLE_CY)**2) <= POLE_RADIUS
    xt, yt = xr[~pole_mask], yr[~pole_mask]

    # 4. Filtrar solo puntos cerca de la distancia nominal del trailer
    r_trailer = np.sqrt(xt**2 + yt**2)
    trailer_mask = (
        (r_trailer >= TRAILER_DIST_NOM - TRAILER_DIST_TOL) &
        (r_trailer <= TRAILER_DIST_NOM + TRAILER_DIST_TOL)
    )
    xt, yt = xt[trailer_mask], yt[trailer_mask]

    theta1, theta2 = None, None
    m_front, b_front = None, None
    m_rear,  b_rear  = None, None
    x_front = y_front = x_rear = y_rear = np.array([])

    if len(xt) < DBSCAN_MIN_PTS:
        theta1, theta2 = compute_angles(cx_pole, cy_pole, None)
    else:
        # 5. DBSCAN
        labels = DBSCAN(eps=DBSCAN_EPS,
                        min_samples=DBSCAN_MIN_PTS).fit_predict(
                            np.c_[xt, yt])
        unique = [l for l in set(labels) if l != -1]

        if cx_pole is not None and len(unique) >= 1:
            pole_r = np.sqrt(cx_pole**2 + cy_pole**2)
            fronts, rears = [], []
            for lbl in unique:
                mask_l  = labels == lbl
                xc, yc  = xt[mask_l], yt[mask_l]
                dist_c  = np.mean(np.sqrt(xc**2 + yc**2))
                if dist_c < pole_r:
                    fronts.append((xc, yc))
                else:
                    rears.append((xc, yc))

            if fronts:
                x_front = np.concatenate([f[0] for f in fronts])
                y_front = np.concatenate([f[1] for f in fronts])
            if rears:
                x_rear = np.concatenate([f[0] for f in rears])
                y_rear = np.concatenate([f[1] for f in rears])

        elif len(unique) >= 1:
            # Sin poste detectado: usar el cluster más grande como trailer
            sizes   = [(labels == l).sum() for l in unique]
            biggest = unique[np.argmax(sizes)]
            mask_b  = labels == biggest
            x_front, y_front = xt[mask_b], yt[mask_b]

        # 6. Ajustar rectas
        if len(x_front) >= 4:
            m_front, b_front, _ = fit_line_ransac(x_front, y_front)

        if len(x_rear) >= 4:
            m_rear, b_rear, _ = fit_line_ransac(x_rear, y_rear)

    # 7. Calcular ángulos con el modelo geométrico
    # φ₁ usa la recta del segmento front (más cercano al poste)
    # si no hay front, intenta con rear
    m_for_angle = m_front if m_front is not None else m_rear
    theta1, theta2 = compute_angles(cx_pole, cy_pole, m_for_angle)

    # Si hay ambos segmentos, θ₂ usa la recta rear
    if m_rear is not None and cx_pole is not None:
        phi1_unused, theta2 = compute_angles(cx_pole, cy_pole, m_rear)

    return {
        'x_all': x_all, 'y_all': y_all,
        'xr': xr, 'yr': yr,
        'xp': xp, 'yp': yp,
        'cx_pole': cx_pole, 'cy_pole': cy_pole,
        'x_front': x_front, 'y_front': y_front,
        'x_rear':  x_rear,  'y_rear':  y_rear,
        'm_front': m_front,  'b_front': b_front,
        'm_rear':  m_rear,   'b_rear':  b_rear,
        'theta1': theta1,    'theta2': theta2,
        'ts': sc['ts'],
    }

# Dibuja una recta ajustada en el rango visible 
def draw_line(ax, m, b, x_pts, color, lw=1.5, label=''):
    if m is None or len(x_pts) == 0:
        return
    x0, x1 = x_pts.min() - 0.1, x_pts.max() + 0.1
    ax.plot([x0, x1], [m*x0+b, m*x1+b], color=color, lw=lw,
            ls='--', label=label, zorder=4)

#  Visualización interactiva 
def plot_interactive(scans):
    n = len(scans)

    print("Procesando scans...")
    results = [process_scan(s) for s in scans]
    print("[OK] Procesamiento completo")

    ts_arr  = np.array([scans[i]['ts'] - scans[0]['ts'] for i in range(n)])
    th1_arr = np.array([r['theta1'] if r and r['theta1'] is not None else np.nan
                        for r in results])
    th2_arr = np.array([r['theta2'] if r and r['theta2'] is not None else np.nan
                        for r in results])

    fig = plt.figure(figsize=(17, 9))
    fig.suptitle('Estimación de ángulos de articulación  ·  G2T IPD-482',
                 fontsize=12, fontweight='bold')

    # Layout: 3 paneles arriba + serie temporal abajo
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[3, 1],
                           hspace=0.38, wspace=0.32)
    ax_full  = fig.add_subplot(gs[0, 0])   # nube completa
    ax_roi   = fig.add_subplot(gs[0, 1])   # zoom ROI con cajas
    ax_pole  = fig.add_subplot(gs[0, 2])   # zoom ultra-cercano al poste
    ax_theta = fig.add_subplot(gs[1, :])   # serie temporal

    # Estilo común 
    for ax in [ax_full, ax_roi, ax_pole]:
        ax.set_aspect('equal')
        ax.grid(True, lw=0.4, color='#cccccc', ls='--')
        ax.axhline(0, color='#aaaaaa', lw=0.8)
        ax.axvline(0, color='#aaaaaa', lw=0.8)
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)

    # Panel 1: vista completa con círculos de referencia
    ax_full.set_xlim(-4, 8); ax_full.set_ylim(-4, 4)
    ax_full.set_title('Vista completa', fontsize=9)
    for r_c in [2, 4, 6]:
        ax_full.add_patch(plt.Circle((0, 0), r_c, color='#dddddd',
                                     fill=False, lw=0.6, ls=':'))
        ax_full.text(r_c*0.707+0.05, r_c*0.707+0.05, f'{r_c}m',
                     color='#bbbbbb', fontsize=6)

    # Panel 2: zoom ROI con cajas dibujadas
    pad = 0.15
    ax_roi.set_xlim(ROI_X_MIN - pad, ROI_X_MAX + pad)
    ax_roi.set_ylim(ROI_Y_MIN - pad, ROI_Y_MAX + pad)
    ax_roi.set_title('Zoom ROI', fontsize=9)

    # Caja ROI general
    from matplotlib.patches import Rectangle
    roi_rect = Rectangle((ROI_X_MIN, ROI_Y_MIN),
                          ROI_X_MAX - ROI_X_MIN, ROI_Y_MAX - ROI_Y_MIN,
                          lw=1.2, edgecolor='#2196F3', facecolor='#E3F2FD',
                          alpha=0.25, zorder=1, label='ROI general')
    ax_roi.add_patch(roi_rect)
    # Caja ROI del poste — ahora círculo
    pole_circle = plt.Circle((POLE_CX, POLE_CY), POLE_RADIUS,
                              lw=1.2, edgecolor='#E53935', facecolor='#FFEBEE',
                              alpha=0.35, zorder=1, label='ROI poste')
    ax_roi.add_patch(pole_circle)
    ax_roi.legend(fontsize=7, loc='upper right')

    # Panel 3: zoom sobre el círculo del poste
    ax_pole.set_xlim(POLE_CX - POLE_RADIUS - 0.05, POLE_CX + POLE_RADIUS + 0.05)
    ax_pole.set_ylim(POLE_CY - POLE_RADIUS - 0.05, POLE_CY + POLE_RADIUS + 0.05)
    ax_pole.set_title('Zoom poste', fontsize=9)
    pole_circle2 = plt.Circle((POLE_CX, POLE_CY), POLE_RADIUS,
                               lw=1, edgecolor='#E53935', facecolor='#FFEBEE',
                               alpha=0.25, zorder=1)
    ax_pole.add_patch(pole_circle2)

    # ── Serie temporal ────────────────────────────────────────────────────────
    ax_theta.set_xlabel('tiempo (s)', fontsize=8)
    ax_theta.set_ylabel('ángulo (°)', fontsize=8)
    ax_theta.set_title('Ángulos de articulación estimados', fontsize=9)
    ax_theta.grid(True, lw=0.4, color='#cccccc', ls='--')
    ax_theta.plot(ts_arr, th1_arr, color='#2196F3', lw=1.2,
                  label='θ₁ tractor–trailer1', alpha=0.8)
    ax_theta.plot(ts_arr, th2_arr, color='#FF5722', lw=1.2,
                  label='θ₂ trailer1–trailer2', alpha=0.8)
    vline1 = ax_theta.axvline(0, color='#333333', lw=1, ls='--')
    ax_theta.legend(fontsize=8, loc='upper right')

    # ── Elementos dinámicos ───────────────────────────────────────────────────
    # Panel completo
    scat_all   = ax_full.scatter([], [], s=2,  c='#cccccc', zorder=2)
    scat_roi_f = ax_full.scatter([], [], s=5,  c='#1565C0', zorder=3, label='ROI')
    scat_pole_f= ax_full.scatter([], [], s=25, c='#E53935', zorder=5,
                                 marker='D', label='poste')
    line_f_full= ax_full.plot([], [], '-', color='#4CAF50', lw=2)[0]
    line_r_full= ax_full.plot([], [], '-', color='#FF9800', lw=2)[0]
    ax_full.plot(0, 0, 'k+', ms=10, mew=2, zorder=7, label='sensor')
    ax_full.legend(fontsize=7, loc='upper right')

    # Panel ROI
    scat_all_r  = ax_roi.scatter([], [], s=3,  c='#cccccc', zorder=2)
    scat_front  = ax_roi.scatter([], [], s=12, c='#4CAF50', zorder=4,
                                 label='')
    scat_rear   = ax_roi.scatter([], [], s=12, c='#FF9800', zorder=4,
                                 label='trailer rear')
    scat_pole_r = ax_roi.scatter([], [], s=40, c='#E53935', zorder=5,
                                 marker='D', label='poste')
    # Líneas RANSAC extendidas al ancho del panel
    line_f_roi  = ax_roi.plot([], [], '-', color='#4CAF50', lw=2,
                              label='')[0]
    line_r_roi  = ax_roi.plot([], [], '-', color='#FF9800', lw=2,
                              label='recta rear')[0]
    pole_c_r    = ax_roi.plot([], [], 'r+', ms=12, mew=2, zorder=6)[0]
    ax_roi.plot(0, 0, 'k+', ms=8, mew=2, zorder=7)
    ax_roi.plot(HITCH_X, HITCH_Y, 's', color='#9C27B0', ms=7,
                mew=1.5, zorder=7, label='enganche (xh,yh)')
    ax_roi.legend(fontsize=7, loc='upper right')

    # Panel poste
    scat_pole_z = ax_pole.scatter([], [], s=30, c='#E53935', zorder=4,
                                  marker='D', label='puntos poste')
    pole_c_z    = ax_pole.plot([], [], 'r+', ms=14, mew=2, zorder=6,
                               label='centroide')[0]
    ax_pole.legend(fontsize=7, loc='upper right')

    title_f = ax_full.set_title('', fontsize=8, color='#555')
    title_r = ax_roi.set_title('Zoom ROI', fontsize=9)
    title_p = ax_pole.set_title('Zoom poste', fontsize=9)

    vdot = ax_theta.plot([], [], 'o', color='#333333', ms=5, zorder=5)[0]

    def _set_line_extended(line_obj, m, b, ax):
        """Extiende la recta al ancho completo del eje."""
        if m is None:
            line_obj.set_data([], [])
            return
        xlim = ax.get_xlim()
        x0, x1 = xlim[0], xlim[1]
        line_obj.set_data([x0, x1], [m*x0+b, m*x1+b])

    def update(val):
        idx = int(slider.val)
        r   = results[idx]
        sc  = scans[idx]
        t   = sc['ts'] - scans[0]['ts']

        # Nube completa
        pts = np.c_[sc['x'], sc['y']] if len(sc['x']) else np.empty((0,2))
        scat_all.set_offsets(pts)

        if r is None:
            for obj in [scat_roi_f, scat_pole_f, scat_all_r,
                        scat_front, scat_rear, scat_pole_r,
                        scat_pole_z, pole_c_r, pole_c_z]:
                try:
                    obj.set_offsets(np.empty((0,2)))
                except Exception:
                    obj.set_data([], [])
            for obj in [line_f_full, line_r_full, line_f_roi, line_r_roi]:
                obj.set_data([], [])
            title_f.set_text(f'frame {idx}  t={t:.3f}s  — sin detección')
            fig.canvas.draw_idle()
            return

        # ROI en panel completo
        roi_pts = np.c_[r['xr'], r['yr']] if len(r['xr']) else np.empty((0,2))
        scat_roi_f.set_offsets(roi_pts)

        # Poste en panel completo
        pole_pts = np.c_[r['xp'], r['yp']] if len(r['xp']) else np.empty((0,2))
        scat_pole_f.set_offsets(pole_pts)

        # Panel ROI
        scat_all_r.set_offsets(pts)
        scat_front.set_offsets(
            np.c_[r['x_front'], r['y_front']] if len(r['x_front']) else np.empty((0,2)))
        scat_rear.set_offsets(
            np.c_[r['x_rear'], r['y_rear']] if len(r['x_rear']) else np.empty((0,2)))
        scat_pole_r.set_offsets(pole_pts)

        # Centroide poste
        if r['cx_pole'] is not None:
            pole_c_r.set_data([r['cx_pole']], [r['cy_pole']])
        else:
            pole_c_r.set_data([], [])

        # Panel poste
        scat_pole_z.set_offsets(pole_pts)
        if r['cx_pole'] is not None:
            pole_c_z.set_data([r['cx_pole']], [r['cy_pole']])
        else:
            pole_c_z.set_data([], [])

        # Líneas extendidas
        _set_line_extended(line_f_roi,  r['m_front'], r['b_front'], ax_roi)
        _set_line_extended(line_r_roi,  r['m_rear'],  r['b_rear'],  ax_roi)
        _set_line_extended(line_f_full, r['m_front'], r['b_front'], ax_full)
        _set_line_extended(line_r_full, r['m_rear'],  r['b_rear'],  ax_full)

        # Títulos
        th1_str = f"{r['theta1']:.1f}°"  if r['theta1']  is not None else "—"
        th2_str = f"{r['theta2']:.1f}°"  if r['theta2']  is not None else "—"
        m_val   = r['m_front'] if r['m_front'] is not None else r['m_rear']
        if m_val is not None:
            m_str    = f"{m_val:.4f}"
            mang_str = f"{np.degrees(np.arctan(m_val)):.1f}°"
        else:
            m_str = mang_str = "—"

        title_f.set_text(
            f'frame {idx}  t={t:.3f}s  |  '
            f'φ₁={th1_str}   a={m_str}   arctan(a)={mang_str}   φ₂={th2_str}')
        title_r.set_text(
            f'Zoom ROI  ·  φ₁={th1_str}   a={m_str}   arctan(a)={mang_str}   φ₂={th2_str}')
        pole_det = "detectado" if r['cx_pole'] is not None else "no detectado"
        title_p.set_text(f'Zoom poste  ·  {pole_det}')

        # Cursor timeline
        vline1.set_xdata([t, t])
        vdot.set_data([t], [r['theta1'] if r['theta1'] is not None else np.nan])

        fig.canvas.draw_idle()

    # Slider
    ax_sl = fig.add_axes([0.12, 0.01, 0.76, 0.02])
    slider = Slider(ax_sl, '', 0, n-1, valinit=0, valstep=1)
    slider.label.set_visible(False)
    slider.valtext.set_fontsize(8)
    slider.on_changed(update)

    def on_resize(event):
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        ax_sl.set_position([0.12, 0.01, 0.76, 0.02])
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('resize_event', on_resize)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    update(0)
    plt.show()
    return results

# ── Ventana 2: Trayectoria ────────────────────────────────────────────────────
def plot_trajectory(odom):
    if not odom:
        print("[WARN] No hay datos de odometría")
        return
    xs  = np.array([o['x']   for o in odom])
    ys  = np.array([o['y']   for o in odom])
    ts  = np.array([o['ts']  for o in odom])
    ts -= ts[0]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Trayectoria del robot  ·  G2T IPD-482',
                 fontsize=12, fontweight='bold')

    # Color por tiempo
    sc = ax.scatter(xs, ys, c=ts, cmap='viridis', s=4, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('tiempo (s)', fontsize=8)

    # Inicio / fin
    ax.plot(xs[0],  ys[0],  'go', ms=10, zorder=5, label='inicio')
    ax.plot(xs[-1], ys[-1], 'rs', ms=10, zorder=5, label='fin')

    # Flecha de orientación cada N muestras
    step = max(1, len(odom)//30)
    for i in range(0, len(odom), step):
        yaw_r = np.radians(odom[i]['yaw'])
        ax.annotate('', xy=(xs[i]+0.08*np.cos(yaw_r),
                             ys[i]+0.08*np.sin(yaw_r)),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8))

    ax.set_aspect('equal')
    ax.grid(True, lw=0.4, color='#cccccc', ls='--')
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.legend(fontsize=8)

    # Métricas de trayectoria
    dist_total = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
    fig.text(0.02, 0.01,
             f'Distancia recorrida: {dist_total:.2f} m   |   '
             f'Duración: {ts[-1]:.1f} s',
             fontsize=8, color='#555')
    plt.tight_layout()
    plt.show()


# ── Ventana 3: Error angular φ₁ ──────────────────────────────────────────────
def plot_error_phi1(results, scans, gt1, save_dir='.'):
    if not gt1:
        print("[WARN] No hay datos de ground truth para θ₁")
        return

    ts_est  = np.array([scans[i]['ts'] - scans[0]['ts'] for i in range(len(scans))])
    th1_est = np.array([r['theta1'] if r and r['theta1'] is not None else np.nan
                        for r in results])
    th1_gt  = interp_gt(gt1, ts_est)
    err1    = th1_est - th1_gt

    rmse1 = np.sqrt(np.nanmean(err1**2))
    mae1  = np.nanmean(np.abs(err1))
    std1  = np.nanstd(err1)
    max1  = np.nanmax(np.abs(err1))
    valid1 = err1[~np.isnan(err1)]

    # ── Fig 1: estimado vs GT ─────────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_est, th1_gt,  color='#333333', lw=1.5, label='GT φ₁',   ls='--')
    ax.plot(ts_est, th1_est, color='#2196F3', lw=1.2, label='Est. φ₁', alpha=0.85)
    ax.set_title(f'φ₁  —  RMSE={rmse1:.2f}°  MAE={mae1:.2f}°', fontsize=10)
    ax.set_xlabel('tiempo (s)', fontsize=9)
    ax.set_ylabel('ángulo (°)', fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, lw=0.4, color='#cccccc', ls='--')
    plt.tight_layout()
    path1 = f'{save_dir}/phi1_estimado_vs_gt.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"[OK] Guardado: {path1}")
    plt.show()

    # ── Fig 2: error en el tiempo ─────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_est, err1, color='#2196F3', lw=1, alpha=0.85)
    ax.axhline(0,      color='black',   lw=0.8, ls='--')
    ax.axhline( rmse1, color='#E53935', lw=0.8, ls=':', label=f'+RMSE={rmse1:.2f}°')
    ax.axhline(-rmse1, color='#E53935', lw=0.8, ls=':')
    ax.set_title('Error φ₁  (estimado − GT)', fontsize=10)
    ax.set_xlabel('tiempo (s)', fontsize=9)
    ax.set_ylabel('error (°)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, color='#cccccc', ls='--')
    plt.tight_layout()
    path2 = f'{save_dir}/phi1_error_temporal.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"[OK] Guardado: {path2}")
    plt.show()

    # ── Fig 3: histograma ─────────────────────────────────────────────────────
    fig3, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid1, bins=30, color='#2196F3', alpha=0.75, edgecolor='white')
    ax.axvline(0,               color='black',   lw=1,   ls='--')
    ax.axvline(np.mean(valid1), color='#E53935', lw=1.2, ls='-',
               label=f'media={np.mean(valid1):.2f}°')
    ax.set_title('Distribución error φ₁', fontsize=10)
    ax.set_xlabel('error (°)', fontsize=9)
    ax.set_ylabel('frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, color='#cccccc', ls='--', axis='y')
    fig3.text(0.5, 0.01,
              f'RMSE={rmse1:.3f}°  MAE={mae1:.3f}°  std={std1:.3f}°  max_abs={max1:.3f}°',
              ha='center', fontsize=8, color='#333',
              bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', boxstyle='round,pad=0.3'))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path3 = f'{save_dir}/phi1_histograma.png'
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"[OK] Guardado: {path3}")
    plt.show()


# ── Ventana 4: Error angular φ₂ ──────────────────────────────────────────────
def plot_error_phi2(results, scans, gt2, save_dir='.'):
    if not gt2:
        print("[WARN] No hay datos de ground truth para θ₂")
        return

    ts_est  = np.array([scans[i]['ts'] - scans[0]['ts'] for i in range(len(scans))])
    th2_est = np.array([r['theta2'] if r and r['theta2'] is not None else np.nan
                        for r in results])
    th2_gt  = interp_gt(gt2, ts_est)
    err2    = th2_est - th2_gt

    rmse2 = np.sqrt(np.nanmean(err2**2))
    mae2  = np.nanmean(np.abs(err2))
    std2  = np.nanstd(err2)
    max2  = np.nanmax(np.abs(err2))
    valid2 = err2[~np.isnan(err2)]

    # ── Fig 4: estimado vs GT ─────────────────────────────────────────────────
    fig4, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_est, th2_gt,  color='#333333', lw=1.5, label='GT φ₂',   ls='--')
    ax.plot(ts_est, th2_est, color='#FF5722', lw=1.2, label='Est. φ₂', alpha=0.85)
    ax.set_title(f'φ₂  —  RMSE={rmse2:.2f}°  MAE={mae2:.2f}°', fontsize=10)
    ax.set_xlabel('tiempo (s)', fontsize=9)
    ax.set_ylabel('ángulo (°)', fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, lw=0.4, color='#cccccc', ls='--')
    plt.tight_layout()
    path4 = f'{save_dir}/phi2_estimado_vs_gt.png'
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    print(f"[OK] Guardado: {path4}")
    plt.show()

    # ── Fig 5: error en el tiempo ─────────────────────────────────────────────
    fig5, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_est, err2, color='#FF5722', lw=1, alpha=0.85)
    ax.axhline(0,      color='black',   lw=0.8, ls='--')
    ax.axhline( rmse2, color='#E53935', lw=0.8, ls=':', label=f'+RMSE={rmse2:.2f}°')
    ax.axhline(-rmse2, color='#E53935', lw=0.8, ls=':')
    ax.set_title('Error φ₂  (estimado − GT)', fontsize=10)
    ax.set_xlabel('tiempo (s)', fontsize=9)
    ax.set_ylabel('error (°)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, color='#cccccc', ls='--')
    plt.tight_layout()
    path5 = f'{save_dir}/phi2_error_temporal.png'
    plt.savefig(path5, dpi=150, bbox_inches='tight')
    print(f"[OK] Guardado: {path5}")
    plt.show()

    # ── Fig 6: histograma ─────────────────────────────────────────────────────
    fig6, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid2, bins=30, color='#FF5722', alpha=0.75, edgecolor='white')
    ax.axvline(0,               color='black',   lw=1,   ls='--')
    ax.axvline(np.mean(valid2), color='#E53935', lw=1.2, ls='-',
               label=f'media={np.mean(valid2):.2f}°')
    ax.set_title('Distribución error φ₂', fontsize=10)
    ax.set_xlabel('error (°)', fontsize=9)
    ax.set_ylabel('frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, color='#cccccc', ls='--', axis='y')
    fig6.text(0.5, 0.01,
              f'RMSE={rmse2:.3f}°  MAE={mae2:.3f}°  std={std2:.3f}°  max_abs={max2:.3f}°',
              ha='center', fontsize=8, color='#333',
              bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', boxstyle='round,pad=0.3'))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path6 = f'{save_dir}/phi2_histograma.png'
    plt.savefig(path6, dpi=150, bbox_inches='tight')
    print(f"[OK] Guardado: {path6}")
    plt.show()

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimación ángulos articulación G2T – IPD482')
    parser.add_argument('--bag',      default=BAG_PATH)
    parser.add_argument('--topic',    default=SCAN_TOPIC)
    parser.add_argument('--no-traj',  action='store_true', help='Omitir ventana trayectoria')
    parser.add_argument('--no-error', action='store_true', help='Omitir ventana error')
    parser.add_argument('--save-dir', default='.', help='Directorio donde guardar las figuras')
    args = parser.parse_args()

    scans          = load_scans(args.bag, args.topic)
    gt1, gt2, odom = load_ground_truth(args.bag)

    results = plot_interactive(scans)

    if not args.no_traj:
        plot_trajectory(odom)

    if not args.no_error:
        plot_error_phi1(results, scans, gt1, save_dir=args.save_dir)
        plot_error_phi2(results, scans, gt2, save_dir=args.save_dir)