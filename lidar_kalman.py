#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import sys, os
from pathlib import Path

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    os.system("pip install rosbags --break-system-packages -q")
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

BASE_DIR = Path(__file__).resolve().parent
BAG_PATH = BASE_DIR / "dataset_ROS"
SCAN_TOPIC = '/scan'

ROI_X_MIN =  0.05; ROI_X_MAX =  0.75
ROI_Y_MIN = -0.35; ROI_Y_MAX =  0.35
POLE_CX = 0.2; POLE_CY = 0.0; POLE_RADIUS = 0.23; POLE_ARC_MIN = 8
HITCH_X = 0.22; HITCH_Y = 0.0
TRAILER_DIST_NOM = 0.63; TRAILER_DIST_TOL = 1.50
DBSCAN_EPS = 0.12; DBSCAN_MIN_PTS = 3

def load_scans(bag_path, topic=SCAN_TOPIC):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    scans = []
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            print(f"[ERROR] Tópico '{topic}' no encontrado."); sys.exit(1)
        for conn, ts_ns, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            ranges = np.array(msg.ranges, dtype=np.float32)
            angle_min = float(msg.angle_min); angle_inc = float(msg.angle_increment)
            r_max = float(msg.range_max); r_min = float(msg.range_min)
            angles = angle_min + np.arange(len(ranges)) * angle_inc
            valid = (ranges >= r_min) & (ranges < r_max)
            r_val = ranges[valid]; a_val = angles[valid]
            scans.append({'ts': ts_ns/1e9, 'x': r_val*np.cos(a_val),
                          'y': r_val*np.sin(a_val), 'ranges': r_val})
    print(f"[OK] {len(scans)} scans cargados")
    return scans

def load_ground_truth(bag_path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    gt1, gt2, odom = [], [], []
    current_v = 0.0; current_alpha = 0.0
    with Reader(bag_path) as reader:
        conns_all = ([c for c in reader.connections if c.topic == t]
                     for t in ['/gt_theta1','/gt_theta2','/lidar_pose',
                                '/cmd_velocity','/cmd_steering'])
        conns_all = [c for group in conns_all for c in group]
        for conn, ts_ns, raw in reader.messages(connections=conns_all):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            t = ts_ns / 1e9
            if conn.topic == '/cmd_velocity':
                current_v = float(msg.data)
            elif conn.topic == '/cmd_steering':
                current_alpha = float(msg.data)
            elif conn.topic == '/gt_theta1':
                gt1.append({'ts': t, 'val': np.degrees(float(msg.data))})
            elif conn.topic == '/gt_theta2':
                gt2.append({'ts': t, 'val': np.degrees(float(msg.data))})
            elif conn.topic == '/lidar_pose':
                p = msg.pose.pose.position; o = msg.pose.pose.orientation
                siny = 2.0*(o.w*o.z + o.x*o.y)
                cosy = 1.0 - 2.0*(o.y*o.y + o.z*o.z)
                yaw = np.degrees(np.arctan2(siny, cosy))
                odom.append({'ts': t, 'x': float(p.x), 'y': float(p.y),
                             'yaw': yaw, 'v_rueda': current_v, 'alpha': current_alpha})
    print(f"[OK] GT θ₁:{len(gt1)}  θ₂:{len(gt2)}  odom:{len(odom)}")
    return gt1, gt2, odom

def interp_gt(gt_list, ts_query):
    if len(gt_list) < 2:
        return np.full(len(ts_query), np.nan)
    t_gt = np.array([g['ts'] for g in gt_list])
    v_gt = np.array([g['val'] for g in gt_list])
    t_gt -= t_gt[0]
    return np.interp(ts_query, t_gt, v_gt, left=np.nan, right=np.nan)

def apply_roi(x, y):
    mask = ((x >= ROI_X_MIN) & (x <= ROI_X_MAX) &
            (y >= ROI_Y_MIN) & (y <= ROI_Y_MAX))
    return x[mask], y[mask]

def detect_pole(x, y):
    dist = np.sqrt((x - POLE_CX)**2 + (y - POLE_CY)**2)
    mask = dist <= POLE_RADIUS
    xp, yp = x[mask], y[mask]
    if len(xp) < POLE_ARC_MIN:
        return None, None, xp, yp
    return float(np.mean(xp)), float(np.mean(yp)), xp, yp

def fit_line_ransac(x, y):
    if len(x) < 4:
        return None, None, None
    x_range = x.max() - x.min(); y_range = y.max() - y.min()
    try:
        ransac = RANSACRegressor(residual_threshold=0.05, min_samples=4,
                                 max_trials=300, random_state=42)
        if y_range > x_range:
            ransac.fit(y.reshape(-1,1), x)
            mp = float(ransac.estimator_.coef_[0])
            bp = float(ransac.estimator_.intercept_)
            if abs(mp) < 1e-9: return None, None, None
            m = 1.0/mp; b = -bp/mp
        else:
            ransac.fit(x.reshape(-1,1), y)
            m = float(ransac.estimator_.coef_[0])
            b = float(ransac.estimator_.intercept_)
        return m, b, ransac.inlier_mask_
    except Exception:
        return None, None, None

def compute_angles(cx_pole, cy_pole, m_line):
    if cx_pole is None:
        return None, None
    phi1 = np.degrees(np.arctan2(cy_pole - HITCH_Y, cx_pole - HITCH_X))
    if m_line is None:
        return phi1, None
    m_deg = np.degrees(np.arctan(m_line))
    phi2 = 90 - abs(phi1) - abs(m_deg) if phi1 > 0 else -90 + abs(phi1) + abs(m_deg)
    return phi1, phi2

class EKF_G2T:
    def __init__(self):
        self.L0=0.38; self.La0=0.22; self.Lb1=0.42; self.Lb2=0.73
        self.x = np.zeros((5,1))
        self.P = np.eye(5) * 1.0
        self.Q = np.diag([0.05,0.05,np.radians(0.5),np.radians(1.5),np.radians(1.5)])**2
        self.R_gps=0.05**2; self.R_yaw=np.radians(1.0)**2; self.R_lidar=np.radians(2.5)**2

    def f_kinematics(self, state, v_r, alpha):
        x,y,th0,th1,th2 = state.flatten()
        v0 = v_r * np.cos(alpha)
        w0 = (v_r * np.sin(alpha)) / self.L0 if self.L0 != 0 else 0.0
        x_dot=v0*np.cos(th0); y_dot=v0*np.sin(th0); th0_dot=w0
        th1_dot=(v0*np.sin(th0-th1)+w0*self.La0*np.cos(th0-th1))/self.Lb1
        v1=v0*np.cos(th0-th1)-w0*self.La0*np.sin(th0-th1)
        th2_dot=(v1*np.sin(th1-th2))/self.Lb2
        return np.array([[x_dot],[y_dot],[th0_dot],[th1_dot],[th2_dot]])

    def predict(self, v_rueda, alpha, dt):
        self.dt = dt
        x_dot = self.f_kinematics(self.x, v_rueda, alpha)
        self.x = self.x + x_dot * self.dt
        self.x[2:5,0] = (self.x[2:5,0]+np.pi)%(2*np.pi)-np.pi
        F = np.eye(5); epsilon = 1e-5
        for i in range(5):
            x_eps = self.x.copy(); x_eps[i,0] += epsilon
            x_dot_eps = self.f_kinematics(x_eps, v_rueda, alpha)
            F[:,i] += ((x_dot_eps-x_dot)/epsilon).flatten()*self.dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, x_gps, y_gps, yaw_meas, phi1, phi2):
        z_list=[]; H_list=[]; R_list=[]
        if x_gps is not None and y_gps is not None:
            z_list.extend([[x_gps],[y_gps]])
            H_list.extend([[1,0,0,0,0],[0,1,0,0,0]])
            R_list.extend([self.R_gps, self.R_gps])
        if yaw_meas is not None and not np.isnan(yaw_meas):
            z_list.append([yaw_meas]); H_list.append([0,0,1,0,0]); R_list.append(self.R_yaw)
        if phi1 is not None and not np.isnan(phi1):
            z_list.append([np.radians(phi1)]); H_list.append([0,0,-1,1,0]); R_list.append(self.R_lidar)
        if phi2 is not None and not np.isnan(phi2):
            z_list.append([np.radians(phi2)]); H_list.append([0,0,0,-1,1]); R_list.append(self.R_lidar)
        if len(z_list) == 0:
            return
        Z=np.array(z_list); H=np.array(H_list); R=np.diag(R_list)
        Z_pred = H @ self.x
        y = Z - Z_pred
        for i in range(len(y)):
            if R_list[i] in [self.R_lidar, self.R_yaw]:
                y[i,0] = (y[i,0]+np.pi)%(2*np.pi)-np.pi
        S = H @ self.P @ H.T + R
        S += np.eye(S.shape[0]) * 1e-6  # regularización
        try:
            K = self.P @ H.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ y
        self.x[2:5,0] = (self.x[2:5,0]+np.pi)%(2*np.pi)-np.pi
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P

def process_scan(sc):
    x_all, y_all = sc['x'], sc['y']
    xr, yr = apply_roi(x_all, y_all)
    if len(xr) < 5: return None
    cx_pole, cy_pole, xp, yp = detect_pole(xr, yr)
    pole_mask = np.sqrt((xr-POLE_CX)**2+(yr-POLE_CY)**2) <= POLE_RADIUS
    xt, yt = xr[~pole_mask], yr[~pole_mask]
    r_trailer = np.sqrt(xt**2+yt**2)
    trailer_mask = ((r_trailer >= TRAILER_DIST_NOM-TRAILER_DIST_TOL) &
                    (r_trailer <= TRAILER_DIST_NOM+TRAILER_DIST_TOL))
    xt, yt = xt[trailer_mask], yt[trailer_mask]
    theta1=theta2=None; m_front=b_front=m_rear=b_rear=None
    x_front=y_front=x_rear=y_rear=np.array([])
    if len(xt) < DBSCAN_MIN_PTS:
        theta1, theta2 = compute_angles(cx_pole, cy_pole, None)
    else:
        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS).fit_predict(np.c_[xt,yt])
        unique = [l for l in set(labels) if l != -1]
        if cx_pole is not None and len(unique) >= 1:
            pole_r = np.sqrt(cx_pole**2+cy_pole**2)
            fronts, rears = [], []
            for lbl in unique:
                mask_l = labels==lbl; xc,yc = xt[mask_l],yt[mask_l]
                (fronts if np.mean(np.sqrt(xc**2+yc**2)) < pole_r else rears).append((xc,yc))
            if fronts: x_front=np.concatenate([f[0] for f in fronts]); y_front=np.concatenate([f[1] for f in fronts])
            if rears:  x_rear =np.concatenate([f[0] for f in rears]);  y_rear =np.concatenate([f[1] for f in rears])
        elif len(unique) >= 1:
            biggest = unique[np.argmax([(labels==l).sum() for l in unique])]
            x_front, y_front = xt[labels==biggest], yt[labels==biggest]
        if len(x_front) >= 4: m_front, b_front, _ = fit_line_ransac(x_front, y_front)
        if len(x_rear)  >= 4: m_rear,  b_rear,  _ = fit_line_ransac(x_rear,  y_rear)
    m_for = m_front if m_front is not None else m_rear
    theta1, theta2 = compute_angles(cx_pole, cy_pole, m_for)
    if m_rear is not None and cx_pole is not None:
        _, theta2 = compute_angles(cx_pole, cy_pole, m_rear)
    return {'x_all':x_all,'y_all':y_all,'xr':xr,'yr':yr,'xp':xp,'yp':yp,
            'cx_pole':cx_pole,'cy_pole':cy_pole,'x_front':x_front,'y_front':y_front,
            'x_rear':x_rear,'y_rear':y_rear,'m_front':m_front,'b_front':b_front,
            'm_rear':m_rear,'b_rear':b_rear,'theta1':theta1,'theta2':theta2,'ts':sc['ts']}

# ── Ventana 1: slider + EKF ───────────────────────────────────────────────────
def plot_interactive(scans, odom):
    n = len(scans)
    print("Procesando scans y aplicando EKF...")
    results=[]; kf_results=[]; ekf=EKF_G2T()
    initialized=False; last_time=scans[0]['ts']
    odom_ts = np.array([o['ts'] for o in odom])

    for sc in scans:
        r = process_scan(sc); results.append(r)
        idx_odom = np.argmin(np.abs(odom_ts-sc['ts']))
        cur = odom[idx_odom]
        v_r=cur['v_rueda']; alpha=cur['alpha']
        yaw_meas=np.radians(cur['yaw'])
        x_gps=cur['x']+np.random.normal(0,0.05)
        y_gps=cur['y']+np.random.normal(0,0.05)

        if not initialized and r is not None and r['theta1'] is not None:
            ekf.x[0,0]=x_gps; ekf.x[1,0]=y_gps; ekf.x[2,0]=yaw_meas
            ekf.x[3,0]=np.radians(float(r['theta1']))
            ekf.x[4,0]=np.radians(float(r['theta2'])) if r['theta2'] is not None else 0.0
            initialized=True; last_time=sc['ts']
            p1=float(np.degrees((ekf.x[3,0]-ekf.x[2,0]+np.pi)%(2*np.pi)-np.pi))
            p2=float(np.degrees((ekf.x[4,0]-ekf.x[3,0]+np.pi)%(2*np.pi)-np.pi))
            kf_results.append({'x_kf':float(ekf.x[0,0]),'y_kf':float(ekf.x[1,0]),
                                'theta0_kf':float(np.degrees(ekf.x[2,0])),
                                'theta1_kf':float(np.degrees(ekf.x[3,0])),
                                'theta2_kf':float(np.degrees(ekf.x[4,0])),
                                'theta1_kf_rel':p1,'theta2_kf_rel':p2})
            continue

        dt = sc['ts']-last_time
        if dt <= 0: dt=0.001
        last_time=sc['ts']
        ekf.predict(v_r, alpha, dt)
        ekf.update(x_gps, y_gps, yaw_meas,
                   r['theta1'] if r else None,
                   r['theta2'] if r else None)
        p1=float(np.degrees((ekf.x[3,0]-ekf.x[2,0]+np.pi)%(2*np.pi)-np.pi))
        p2=float(np.degrees((ekf.x[4,0]-ekf.x[3,0]+np.pi)%(2*np.pi)-np.pi))
        kf_results.append({'x_kf':float(ekf.x[0,0]),'y_kf':float(ekf.x[1,0]),
                            'theta0_kf':float(np.degrees(ekf.x[2,0])),
                            'theta1_kf':float(np.degrees(ekf.x[3,0])),
                            'theta2_kf':float(np.degrees(ekf.x[4,0])),
                            'theta1_kf_rel':p1,'theta2_kf_rel':p2})

    print("[OK] EKF completo")
    ts_arr = np.array([sc['ts']-scans[0]['ts'] for sc in scans])
    th1_arr = np.array([r['theta1'] if r and r['theta1'] is not None else np.nan for r in results])
    th2_arr = np.array([r['theta2'] if r and r['theta2'] is not None else np.nan for r in results])
    th1_kf  = np.array([k['theta1_kf_rel'] for k in kf_results])
    th2_kf  = np.array([k['theta2_kf_rel'] for k in kf_results])

    fig = plt.figure(figsize=(17,9))
    fig.suptitle('Estimación de ángulos · G2T IPD-482', fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(2,3,figure=fig,height_ratios=[1.0,0.45],hspace=0.38,wspace=0.32)
    ax_full=fig.add_subplot(gs[0,0]); ax_roi=fig.add_subplot(gs[0,1])
    ax_pole=fig.add_subplot(gs[0,2]); ax_theta=fig.add_subplot(gs[1,:])

    for ax in [ax_full,ax_roi,ax_pole]:
        ax.set_aspect('equal'); ax.grid(True,lw=0.4,color='#cccccc',ls='--')
        ax.axhline(0,color='#aaaaaa',lw=0.8); ax.axvline(0,color='#aaaaaa',lw=0.8)
        ax.set_xlabel('X (m)',fontsize=8); ax.set_ylabel('Y (m)',fontsize=8)

    ax_full.set_xlim(-4,8); ax_full.set_ylim(-4,4); ax_full.set_title('Vista completa',fontsize=9)
    for r_c in [1,2,3,4]:
        ax_full.add_patch(plt.Circle((0,0),r_c,color='#dddddd',fill=False,lw=0.6,ls=':'))
        ax_full.text(r_c*0.707+0.05,r_c*0.707+0.05,f'{r_c}m',color='#bbbbbb',fontsize=6)

    pad=0.15
    ax_roi.set_xlim(ROI_X_MIN-pad,ROI_X_MAX+pad); ax_roi.set_ylim(ROI_Y_MIN-pad,ROI_Y_MAX+pad)
    from matplotlib.patches import Rectangle
    ax_roi.add_patch(Rectangle((ROI_X_MIN,ROI_Y_MIN),ROI_X_MAX-ROI_X_MIN,ROI_Y_MAX-ROI_Y_MIN,
                     lw=1.2,edgecolor='#2196F3',facecolor='#E3F2FD',alpha=0.25,zorder=1,label='ROI'))
    ax_roi.add_patch(plt.Circle((POLE_CX,POLE_CY),POLE_RADIUS,lw=1.2,edgecolor='#E53935',
                     facecolor='#FFEBEE',alpha=0.35,zorder=1,label='ROI poste'))
    ax_roi.legend(fontsize=7,loc='upper right')

    ax_pole.set_xlim(POLE_CX-POLE_RADIUS-0.05,POLE_CX+POLE_RADIUS+0.05)
    ax_pole.set_ylim(POLE_CY-POLE_RADIUS-0.05,POLE_CY+POLE_RADIUS+0.05)
    ax_pole.add_patch(plt.Circle((POLE_CX,POLE_CY),POLE_RADIUS,lw=1,edgecolor='#E53935',
                     facecolor='#FFEBEE',alpha=0.25,zorder=1))

    ax_theta.set_xlabel('tiempo (s)',fontsize=8); ax_theta.set_ylabel('ángulo (°)',fontsize=8)
    ax_theta.set_title('Ángulos: LiDAR vs EKF',fontsize=9)
    ax_theta.grid(True,lw=0.4,color='#cccccc',ls='--')
    ax_theta.plot(ts_arr,th1_arr,color='#2196F3',lw=1.2,ls='--',alpha=0.3,label='φ₁ Crudo')
    ax_theta.plot(ts_arr,th2_arr,color='#FF5722',lw=1.2,ls='--',alpha=0.3,label='φ₂ Crudo')
    ax_theta.plot(ts_arr,th1_kf, color='#0D47A1',lw=2.0,label='φ₁ EKF')
    ax_theta.plot(ts_arr,th2_kf, color='#BF360C',lw=2.0,label='φ₂ EKF')
    vline1=ax_theta.axvline(0,color='#333333',lw=1,ls='--')
    ax_theta.legend(fontsize=8,loc='upper right')

    scat_all  =ax_full.scatter([],[],s=2,c='#cccccc',zorder=2)
    scat_roi_f=ax_full.scatter([],[],s=5,c='#1565C0',zorder=3,label='ROI')
    scat_pole_f=ax_full.scatter([],[],s=25,c='#E53935',zorder=5,marker='D',label='poste')
    line_f_full,=ax_full.plot([],[],'-',color='#4CAF50',lw=2)
    line_r_full,=ax_full.plot([],[],'-',color='#FF9800',lw=2)
    ax_full.plot(0,0,'k+',ms=10,mew=2,zorder=7,label='sensor'); ax_full.legend(fontsize=7,loc='upper right')

    scat_all_r =ax_roi.scatter([],[],s=3,c='#cccccc',zorder=2)
    scat_front =ax_roi.scatter([],[],s=12,c='#4CAF50',zorder=4,label='front')
    scat_rear  =ax_roi.scatter([],[],s=12,c='#FF9800',zorder=4,label='rear')
    scat_pole_r=ax_roi.scatter([],[],s=40,c='#E53935',zorder=5,marker='D',label='poste')
    line_f_roi,=ax_roi.plot([],[],'-',color='#4CAF50',lw=2)
    line_r_roi,=ax_roi.plot([],[],'-',color='#FF9800',lw=2)
    pole_c_r,  =ax_roi.plot([],[],'r+',ms=12,mew=2,zorder=6)
    ax_roi.plot(0,0,'k+',ms=8,mew=2,zorder=7)
    ax_roi.plot(HITCH_X,HITCH_Y,'s',color='#9C27B0',ms=7,mew=1.5,zorder=7,label='hitch')
    ax_roi.legend(fontsize=7,loc='upper right')

    scat_pole_z=ax_pole.scatter([],[],s=30,c='#E53935',zorder=4,marker='D',label='poste')
    pole_c_z,  =ax_pole.plot([],[],'r+',ms=14,mew=2,zorder=6,label='centroide')
    ax_pole.legend(fontsize=7,loc='upper right')

    title_f=ax_full.set_title('',fontsize=8,color='#555')
    title_r=ax_roi.set_title('Zoom ROI',fontsize=9)
    title_p=ax_pole.set_title('Zoom poste',fontsize=9)
    vdot1,=ax_theta.plot([],[],'o',color='#0D47A1',ms=5,zorder=5)
    vdot2,=ax_theta.plot([],[],'o',color='#BF360C',ms=5,zorder=5)

    def _ext(line_obj,m,b,ax):
        if m is None: line_obj.set_data([],[]); return
        x0,x1=ax.get_xlim(); line_obj.set_data([x0,x1],[m*x0+b,m*x1+b])

    def update(val):
        idx=int(slider.val); r=results[idx]; k=kf_results[idx]; sc=scans[idx]
        t=sc['ts']-scans[0]['ts']
        pts=np.c_[sc['x'],sc['y']] if len(sc['x']) else np.empty((0,2))
        scat_all.set_offsets(pts)
        if r is None:
            for obj in [scat_roi_f,scat_pole_f,scat_all_r,scat_front,scat_rear,
                        scat_pole_r,scat_pole_z,pole_c_r,pole_c_z]:
                try: obj.set_offsets(np.empty((0,2)))
                except: obj.set_data([],[])
            for obj in [line_f_full,line_r_full,line_f_roi,line_r_roi]: obj.set_data([],[])
            title_f.set_text(f'frame {idx}  t={t:.3f}s  — sin detección')
            fig.canvas.draw_idle(); return
        scat_roi_f.set_offsets(np.c_[r['xr'],r['yr']] if len(r['xr']) else np.empty((0,2)))
        pole_pts=np.c_[r['xp'],r['yp']] if len(r['xp']) else np.empty((0,2))
        scat_pole_f.set_offsets(pole_pts); scat_all_r.set_offsets(pts)
        scat_front.set_offsets(np.c_[r['x_front'],r['y_front']] if len(r['x_front']) else np.empty((0,2)))
        scat_rear.set_offsets(np.c_[r['x_rear'],r['y_rear']] if len(r['x_rear']) else np.empty((0,2)))
        scat_pole_r.set_offsets(pole_pts); scat_pole_z.set_offsets(pole_pts)
        if r['cx_pole'] is not None:
            pole_c_r.set_data([r['cx_pole']],[r['cy_pole']]); pole_c_z.set_data([r['cx_pole']],[r['cy_pole']])
        else:
            pole_c_r.set_data([],[]); pole_c_z.set_data([],[])
        _ext(line_f_roi,r['m_front'],r['b_front'],ax_roi)
        _ext(line_r_roi,r['m_rear'], r['b_rear'], ax_roi)
        _ext(line_f_full,r['m_front'],r['b_front'],ax_full)
        _ext(line_r_full,r['m_rear'], r['b_rear'], ax_full)
        th1_s=f"{r['theta1']:.1f}°" if r['theta1'] is not None else "—"
        th2_s=f"{r['theta2']:.1f}°" if r['theta2'] is not None else "—"
        title_f.set_text(f'frame {idx}  t={t:.3f}s  |  Crudo: φ₁={th1_s} φ₂={th2_s}  |  EKF: φ₁={k["theta1_kf_rel"]:.1f}° φ₂={k["theta2_kf_rel"]:.1f}°')
        title_r.set_text('Zoom ROI'); title_p.set_text(f'Zoom poste · {"detectado" if r["cx_pole"] is not None else "no detectado"}')
        vline1.set_xdata([t,t]); vdot1.set_data([t],[k['theta1_kf_rel']]); vdot2.set_data([t],[k['theta2_kf_rel']])
        fig.canvas.draw_idle()

    ax_sl=fig.add_axes([0.12,0.01,0.76,0.02])
    slider=Slider(ax_sl,'',0,n-1,valinit=0,valstep=1)
    slider.label.set_visible(False); slider.valtext.set_fontsize(8)
    slider.on_changed(update)
    fig.subplots_adjust(bottom=0.08,top=0.95,left=0.05,right=0.98)
    fig.canvas.mpl_connect('resize_event', lambda e: ax_sl.set_position([0.12,0.01,0.76,0.02]))
    update(0); plt.show()
    return results, kf_results

# ── Ventana 2: pose comparison EKF vs GT ─────────────────────────────────────
def plot_pose_comparison(scans, odom, kf_results, gt1, gt2):
    ts_est = np.array([sc['ts']-scans[0]['ts'] for sc in scans])
    odom_ts = np.array([o['ts'] for o in odom])
    synced = [odom[np.argmin(np.abs(odom_ts-sc['ts']))] for sc in scans]
    x_gt=np.array([o['x'] for o in synced]); y_gt=np.array([o['y'] for o in synced])
    theta0_gt=((np.array([o['yaw'] for o in synced])+180)%360)-180
    theta1_gt=interp_gt(gt1,ts_est); theta2_gt=interp_gt(gt2,ts_est)
    x_kf=np.array([k['x_kf'] for k in kf_results]); y_kf=np.array([k['y_kf'] for k in kf_results])
    theta0_kf=((np.array([k['theta0_kf'] for k in kf_results])+180)%360)-180
    theta1_kf=np.array([k['theta1_kf_rel'] for k in kf_results])
    theta2_kf=np.array([k['theta2_kf_rel'] for k in kf_results])

    fig, axes = plt.subplots(5,1,figsize=(12,16),sharex=True)
    fig.suptitle('Pose EKF vs Ground Truth · G2T IPD-482',fontsize=12,fontweight='bold')
    datos = [('x (m)',x_gt,x_kf,'#2196F3'),('y (m)',y_gt,y_kf,'#FB8C00'),
             ('θ₀ (°)',theta0_gt,theta0_kf,'#4CAF50'),
             ('θ₁ (°)',theta1_gt,theta1_kf,'#0D47A1'),
             ('θ₂ (°)',theta2_gt,theta2_kf,'#BF360C')]
    labels_gt = ['GT x','GT y','GT θ₀','GT θ₁','GT θ₂']
    labels_kf = ['KF x','KF y','KF θ₀','KF θ₁','KF θ₂']
    for ax,(lbl,gt,kf,col),lgt,lkf in zip(axes,datos,labels_gt,labels_kf):
        ax.plot(ts_est,gt,color='#333333',ls='--',lw=1.5,label=lgt)
        ax.plot(ts_est,kf,color=col,lw=1.5,label=lkf)
        ax.set_ylabel(lbl); ax.legend(fontsize=8); ax.grid(True,lw=0.4,color='#cccccc',ls='--')
    axes[-1].set_xlabel('tiempo (s)')
    fig.subplots_adjust(hspace=0.25,bottom=0.06,top=0.94)
    plt.show()

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='EKF G2T – IPD482')
    parser.add_argument('--bag',   default=BAG_PATH)
    parser.add_argument('--topic', default=SCAN_TOPIC)
    args = parser.parse_args()

    scans          = load_scans(args.bag, args.topic)
    gt1, gt2, odom = load_ground_truth(args.bag)
    results, kf_results = plot_interactive(scans, odom)
    plot_pose_comparison(scans, odom, kf_results, gt1, gt2)