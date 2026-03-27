import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import CubicSpline, make_interp_spline

# 페이지 기본 설정
st.set_page_config(page_title="최대파고 산정 프로그램", layout="wide", page_icon="🌊")

# 한글 폰트 및 그래프 설정
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.family'] = 'Malgun Gothic' # 윈도우 로컬용
except:
    pass

# 1. SPM Table C-1
spm_table = [
    (0.040, 1.066), (0.041, 1.061), (0.042, 1.056), (0.043, 1.051),
    (0.044, 1.047), (0.045, 1.042), (0.046, 1.038), (0.047, 1.034),
    (0.048, 1.030), (0.049, 1.026), (0.050, 1.023), (0.051, 1.019),
    (0.052, 1.016), (0.053, 1.012), (0.054, 1.009), (0.055, 1.005)
]

def get_interpolated_hh0(target_dl0):
    for i in range(len(spm_table) - 1):
        x1, y1 = spm_table[i]
        x2, y2 = spm_table[i+1]
        if x1 <= target_dl0 <= x2:
            slope = (y2 - y1) / (x2 - x1)
            return y1 + slope * (target_dl0 - x1)
    return 1.05

# 2. 슈토(Shuto) 천수계수 2D 보간 테이블
shuto_matrix = {
    0.010: [(0.040, 1.15), (0.046, 1.10), (0.050, 1.07), (0.060, 1.04)],
    0.019: [(0.040, 1.12), (0.046, 1.08), (0.050, 1.06), (0.060, 1.02)],
    0.030: [(0.040, 1.08), (0.046, 1.05), (0.050, 1.03), (0.060, 1.00)]
}

def get_shuto_ks(h_L0, H0p_L0):
    y_keys = sorted(list(shuto_matrix.keys()))
    y1, y2 = y_keys[0], y_keys[-1]
    for i in range(len(y_keys) - 1):
        if y_keys[i] <= H0p_L0 <= y_keys[i+1]:
            y1, y2 = y_keys[i], y_keys[i+1]
            break

    def interp_x(target_x, points):
        for i in range(len(points) - 1):
            x1, v1 = points[i]
            x2, v2 = points[i+1]
            if x1 <= target_x <= x2:
                return v1 + (v2 - v1) * (target_x - x1) / (x2 - x1)
        return points[0][1] if target_x < points[0][0] else points[-1][1]

    val_y1 = interp_x(h_L0, shuto_matrix[y1])
    val_y2 = interp_x(h_L0, shuto_matrix[y2])

    if y1 == y2: return val_y1
    return val_y1 + (val_y2 - val_y1) * (H0p_L0 - y1) / (y2 - y1)

# H1/3 약산식 계산 함수
def calc_h13_formula(H0p, h, L0, tanTheta, Ks):
    H0p_L0 = H0p / L0
    if H0p_L0 <= 0: return 0, 0, 0, 0, 0, 0, 0
    beta0 = 0.028 * (H0p_L0 ** -0.38) * math.exp(20 * (tanTheta ** 1.5))
    beta1 = 0.52 * math.exp(4.2 * tanTheta)
    betaMax = max(0.92, 0.32 * (H0p_L0 ** -0.29) * math.exp(2.4 * tanTheta))
    val1 = beta0 * H0p + beta1 * h
    val2 = betaMax * H0p
    val3 = Ks * H0p
    return min(val1, val2, val3), beta0, beta1, betaMax, val1, val2, val3

# Hmax 약산식 계산 함수
def calc_hmax_formula(H0p, h, L0, tanTheta, Ks):
    H0p_L0 = H0p / L0
    if H0p_L0 <= 0: return 0, 0, 0, 0, 0, 0, 0
    beta0_star = 0.052 * (H0p_L0 ** -0.38) * math.exp(20 * (tanTheta ** 1.5))
    beta1_star = 0.63 * math.exp(3.8 * tanTheta)
    betaMax_star = max(1.65, 0.53 * (H0p_L0 ** -0.29) * math.exp(2.4 * tanTheta))
    val1 = beta0_star * H0p + beta1_star * h
    val2 = betaMax_star * H0p
    val3 = 1.8 * Ks * H0p
    return min(val1, val2, val3), beta0_star, beta1_star, betaMax_star, val1, val2, val3


# -------------------------------------------------------------------------
# ★★★ 초정밀 교정 완료된 Goda 도표 데이터베이스 (사용자 실측값 100% 반영) ★★★
# -------------------------------------------------------------------------
goda_data_master = {
    0.01: {
        0.002: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.659, 0.869, 1.176, 1.503, 1.843, 2.187, 2.546, 2.906, 3.18]},
        0.005: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.439, 0.698, 1.003, 1.348, 1.712, 2.069, 2.365, 2.497, 2.401]},
        0.01: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.335, 0.596, 0.929, 1.283, 1.625, 1.922, 2.083, 2.032, 1.952]},
        0.02: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.25, 0.527, 0.873, 1.216, 1.522, 1.743, 1.783, 1.748, 1.717]},
        0.04: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.186, 0.487, 0.811, 1.115, 1.369, 1.55, 1.625, 1.649, 1.653]},
        0.08: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.149, 0.429, 0.698, 0.944, 1.161, 1.332, 1.448, 1.532, 1.59]},
    },
    0.02: {
        0.002: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.684, 0.913, 1.21, 1.546, 1.908, 2.285, 2.668, 3.029, 3.275]},
        0.005: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.458, 0.717, 1.041, 1.408, 1.788, 2.154, 2.452, 2.515, 2.392]},
        0.01: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.345, 0.615, 0.961, 1.339, 1.705, 1.998, 2.115, 2.082, 1.973]},
        0.02: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.253, 0.559, 0.922, 1.273, 1.578, 1.776, 1.788, 1.756, 1.702]},
        0.04: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.202, 0.511, 0.856, 1.176, 1.422, 1.585, 1.637, 1.647, 1.645]},
        0.08: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.149, 0.45, 0.741, 0.992, 1.195, 1.351, 1.469, 1.553, 1.602]},
    },
    0.033: {
        0.002: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.757, 1.002, 1.316, 1.669, 2.055, 2.454, 2.852, 3.215, 3.44]},
        0.005: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.504, 0.782, 1.142, 1.511, 1.896, 2.296, 2.581, 2.555, 2.407]},
        0.01: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.382, 0.68, 1.042, 1.432, 1.813, 2.114, 2.144, 2.053, 1.963]},
        0.02: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.293, 0.609, 0.968, 1.341, 1.686, 1.85, 1.817, 1.769, 1.74]},
        0.04: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.233, 0.543, 0.898, 1.229, 1.488, 1.626, 1.655, 1.659, 1.656]},
        0.08: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.177, 0.49, 0.794, 1.061, 1.269, 1.411, 1.512, 1.589, 1.654]},
    },
    0.05: {
        0.002: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.7], 'y': [0.85, 1.112, 1.445, 1.823, 2.224, 2.64, 3.048, 3.403, 3.493]},
        0.005: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.573, 0.88, 1.248, 1.649, 2.066, 2.471, 2.719, 2.576, 2.413]},
        0.01: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.444, 0.76, 1.146, 1.557, 1.955, 2.227, 2.155, 2.038, 1.95]},
        0.02: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.331, 0.679, 1.066, 1.452, 1.78, 1.872, 1.8, 1.748, 1.72]},
        0.04: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.246, 0.612, 0.985, 1.318, 1.571, 1.648, 1.653, 1.646, 1.644]},
        0.08: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.182, 0.542, 0.863, 1.121, 1.314, 1.452, 1.544, 1.606, 1.644]},
    },
    0.1: {
        0.002: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.61], 'y': [1.208, 1.518, 1.927, 2.383, 2.879, 3.388, 3.498]},
        0.005: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.822, 1.183, 1.621, 2.115, 2.637, 3.039, 2.884, 2.604, 2.423]},
        0.01: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.628, 1.014, 1.472, 1.977, 2.419, 2.393, 2.169, 2.04, 1.959]},
        0.02: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.474, 0.865, 1.353, 1.826, 2.034, 1.908, 1.814, 1.761, 1.737]},
        0.04: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.371, 0.779, 1.228, 1.594, 1.726, 1.681, 1.658, 1.655, 1.655]},
        0.08: {'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 'y': [0.282, 0.683, 1.044, 1.316, 1.49, 1.583, 1.627, 1.642, 1.648]},
    },
}

# 개별 해저경사 데이터셋에서 스플라인 보간을 수행하는 함수
def get_user_curve_spline(x_arr, target_H0p_L0, slope_data_dict):
    keys = sorted(list(slope_data_dict.keys()))
    
    if target_H0p_L0 <= keys[0]:
        target_data = slope_data_dict[keys[0]]
        return CubicSpline(target_data["x"], target_data["y"])(x_arr)
    if target_H0p_L0 >= keys[-1]:
        target_data = slope_data_dict[keys[-1]]
        return CubicSpline(target_data["x"], target_data["y"])(x_arr)

    for i in range(len(keys)-1):
        if keys[i] <= target_H0p_L0 <= keys[i+1]:
            k1, k2 = keys[i], keys[i+1]
            break

    y1 = CubicSpline(slope_data_dict[k1]["x"], slope_data_dict[k1]["y"])(x_arr)
    y2 = CubicSpline(slope_data_dict[k2]["x"], slope_data_dict[k2]["y"])(x_arr)

    log_k1, log_k2, log_t = math.log10(k1), math.log10(k2), math.log10(target_H0p_L0)
    weight = (log_t - log_k1) / (log_k2 - log_k1)

    return y1 + weight * (y2 - y1)

# ★ 해저경사(tanTheta) 간의 보간을 포함하여 최종 파고비를 산출하는 함수 ★
def get_final_graph_ratio(h_H0p_val, H0p_L0_val, tanTheta):
    slope_keys = sorted(list(goda_data_master.keys()))
    
    if tanTheta <= slope_keys[0]:
        return float(get_user_curve_spline(np.array([h_H0p_val]), H0p_L0_val, goda_data_master[slope_keys[0]])[0])
    if tanTheta >= slope_keys[-1]:
        return float(get_user_curve_spline(np.array([h_H0p_val]), H0p_L0_val, goda_data_master[slope_keys[-1]])[0])
        
    for i in range(len(slope_keys)-1):
        if slope_keys[i] <= tanTheta <= slope_keys[i+1]:
            s1, s2 = slope_keys[i], slope_keys[i+1]
            break
            
    val1 = float(get_user_curve_spline(np.array([h_H0p_val]), H0p_L0_val, goda_data_master[s1])[0])
    val2 = float(get_user_curve_spline(np.array([h_H0p_val]), H0p_L0_val, goda_data_master[s2])[0])
    
    log_s1, log_s2, log_t = math.log10(s1), math.log10(s2), math.log10(tanTheta)
    weight = (log_t - log_s1) / (log_s2 - log_s1)
    
    return val1 + weight * (val2 - val1)

# ★ 스플라인 기반 모사 그래프 생성 함수 ★
def plot_authentic_chart_spline(h_H0p_read, read_ratio, user_H0p_L0, tanTheta):
    fig, ax = plt.subplots(figsize=(5.5, 6.8)) 
    
    closest_slope = min(goda_data_master.keys(), key=lambda k: abs(k - tanTheta))
    base_data = goda_data_master[closest_slope]
    
    all_x = []
    all_y = []
    for s in base_data.keys():
        all_x.extend(base_data[s]["x"])
        all_y.extend(base_data[s]["y"])
    
    x_max = max(4.0, h_H0p_read + 0.5, max(all_x) if all_x else 0)
    y_max = max(3.5, read_ratio + 0.5, max(all_y) if all_y else 0)
    
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    ax.grid(which='major', color='black', linewidth=1.0)
    ax.grid(which='minor', color='black', linewidth=0.4)
    
    # -----------------------------------------------------
    # 2% 감쇄선 (Decay line) 데이터베이스 (사용자 요청 정밀 좌표 반영)
    # -----------------------------------------------------
    decay_lines = {
        0.01: { 
            'x': [4.0, 3.5, 3.0, 2.9, 2.825, 2.82, 2.83, 2.86, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.43],
            'y': [2.63, 2.32, 1.99, 1.88, 1.785, 1.75, 1.7, 1.65, 1.63, 1.60, 1.59, 1.58, 1.582, 1.592, 1.6]
        },
        0.02: { 
            'x': [4.0, 3.5, 3.0, 2.9, 2.85, 2.825, 2.83, 2.86, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.64],
            'y': [2.74, 2.43, 2.01, 1.90, 1.80, 1.75, 1.70, 1.64, 1.62, 1.59, 1.575, 1.57, 1.58, 1.589, 1.592, 1.61, 1.62] 
        },
        0.033: { 
            'x': [4.0, 3.5, 3.0, 2.6, 2.59, 2.58, 2.6, 2.65, 2.7, 2.8, 2.9, 3.0, 3.1],
            'y': [3.07, 2.67, 2.228, 1.80, 1.77, 1.75, 1.698, 1.65, 1.63, 1.605, 1.585, 1.575, 1.57]
        },
        0.05: { 
            'x': [4.0, 3.5, 3.0, 2.5, 2.4, 2.39, 2.4, 2.45, 2.5, 2.6, 2.7, 2.8, 2.82],
            'y': [3.425, 2.97, 2.48, 1.91, 1.75, 1.70, 1.66, 1.63, 1.605, 1.58, 1.57, 1.575, 1.58]
        },
        0.1: {
            'x': [3.18, 3.0, 2.5, 2.3, 2.0, 1.95, 1.94, 1.96, 1.98, 2.0, 2.1, 2.2, 2.3, 2.35, 2.4, 2.5],
            'y': [3.5, 3.33, 2.748, 2.48, 2.0, 1.85, 1.8, 1.73, 1.70, 1.68, 1.63, 1.61, 1.598, 1.596, 1.6, 1.61]
        }
    }

    decay_data = decay_lines.get(closest_slope, {'x': [], 'y': []})
    
    decay_path = None
    if decay_data['x']:
        x_arr = np.array(decay_data['x'])
        y_arr = np.array(decay_data['y'])
        
        # 유클리드 거리를 이용한 매개변수 t 산출 (U자형 등 곡선 꺾임 및 튀는 현상 완벽 방지)
        dists = np.sqrt(np.diff(x_arr)**2 + np.diff(y_arr)**2)
        t = np.concatenate(([0], np.cumsum(dists)))
        t = t / t[-1]
        
        # 3차 스플라인(Cubic Spline, k=3) 적용으로 극도로 부드러운 곡률 보장
        spl_x = make_interp_spline(t, x_arr, k=3)
        spl_y = make_interp_spline(t, y_arr, k=3)
        
        t_fine = np.linspace(0, 1, 300)
        fine_x = spl_x(t_fine)
        fine_y = spl_y(t_fine)
        
        # 1. 감쇄선 렌더링 (진한색, 선명한 굵은 1점 쇄선)
        ax.plot(fine_x, fine_y, color='#333333', linestyle='-.', linewidth=2.0, zorder=4)
        
        # 2. 감쇄선 라벨 삽입
        ax.text(decay_data['x'][1] + 0.05, decay_data['y'][1], "2% Decay line", 
                color='#333333', fontsize=10, fontweight='bold', rotation=45, ha='left')

        # 3. 우측 감쇄 영역 판별용 폴리곤 Path 생성 (x=10 넉넉하게 확장)
        poly_points = list(zip(fine_x, fine_y))
        poly_points.append((10.0, fine_y[-1])) 
        poly_points.append((10.0, fine_y[0]))  
        decay_path = mpath.Path(poly_points)

    # -----------------------------------------------------
    # 파형경사(기본) 곡선들 플롯팅 (감쇄선 영역 마스킹 포함)
    for s in base_data.keys():
        curve_data = base_data[s]
        x_curve_arr = np.linspace(curve_data["x"][0], curve_data["x"][-1], 200)
        y_curve = CubicSpline(curve_data["x"], curve_data["y"])(x_curve_arr)
        
        if decay_path:
            pts = np.column_stack((x_curve_arr, y_curve))
            inside = decay_path.contains_points(pts) # Path 내부(우측) 판별
            
            y_solid = np.ma.masked_where(inside, y_curve)
            y_dashed = np.ma.masked_where(~inside, y_curve)
            
            ax.plot(x_curve_arr, y_solid, color='black', linewidth=1.2, zorder=2)
            ax.plot(x_curve_arr, y_dashed, color='black', linewidth=1.2, linestyle='--', zorder=2)
        else:
            ax.plot(x_curve_arr, y_curve, color='black', linewidth=1.2, zorder=2)
            
        label_x = curve_data["x"][-1] * 0.9
        label_y = np.interp(label_x, x_curve_arr, y_curve)
        
        label_text = f"H'o/Lo={s}" if s in [0.002, 0.005] else f"{s}"
        ax.text(label_x, label_y + 0.06, label_text, fontsize=9, backgroundcolor='white', ha='center', va='bottom', zorder=5)
        
    # -----------------------------------------------------
    # 사용자 도출 파형경사(모사) 곡선 플롯팅
    x_user_max = max(base_data[s]["x"][-1] for s in base_data.keys())
    x_user_arr = np.linspace(0, x_user_max, 500)
    y_user = get_user_curve_spline(x_user_arr, user_H0p_L0, base_data)
    
    if decay_path:
        pts_user = np.column_stack((x_user_arr, y_user))
        inside_user = decay_path.contains_points(pts_user)
        yu_solid = np.ma.masked_where(inside_user, y_user)
        yu_dashed = np.ma.masked_where(~inside_user, y_user)
        ax.plot(x_user_arr, yu_solid, color='blue', linewidth=2.5, alpha=0.7, zorder=3)
        ax.plot(x_user_arr, yu_dashed, color='blue', linewidth=2.5, linestyle='--', alpha=0.7, zorder=3)
    else:
        ax.plot(x_user_arr, y_user, color='blue', linewidth=2.5, alpha=0.7, zorder=3) 
    
    # 결과 교차점
    ax.axvline(x=h_H0p_read, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    ax.axhline(y=read_ratio, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    ax.plot(h_H0p_read, read_ratio, 'ro', markersize=8, zorder=6)
    
    ax.text(h_H0p_read + 0.1, read_ratio + 0.1, f"독취결과\nh/H'o={h_H0p_read:.2f}\nHmax/H'o={read_ratio:.3f}", 
            color='red', fontsize=10, fontweight='bold', 
            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3', alpha=0.9), zorder=6)
    
    frac_slope = "1/" + str(int(1/tanTheta)) if tanTheta > 0 else f"{tanTheta}"
    ax.text(0.2, y_max - 0.2, f"  해저경사 {frac_slope} (모사)  ", fontsize=11, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', linewidth=1.2), zorder=5)
    ax.text(1.0, y_max - 0.6, r"$H_{max} \equiv H_{1/250}$", fontsize=10, fontweight='bold', backgroundcolor='white', zorder=5)
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("$h / H_0'$", fontsize=12)
    ax.set_ylabel(r"$\frac{H_{max}}{H_0'}$", fontsize=14, rotation=0, labelpad=15)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    return fig

# --- UI 레이아웃 구성 ---
st.title("🌊 최대파고 완전 자동 산정 프로그램")
st.markdown("항만 및 어항 설계기준 산출 로직 (Spline 이중 보간 적용 완벽 모사)")

col1, col2 = st.columns([1, 2.5])

with col1:
    st.header("📝 입력 제원")
    H13 = st.number_input("설계 유의파고 (H1/3, m)", value=4.90, step=0.1)
    T13 = st.number_input("설계 주기 (T1/3, sec)", value=12.61, step=0.1)
    h = st.number_input("적용 수심 (h, m)", value=12.355, step=0.01)
    tanTheta = st.number_input("해저 경사 (tanθ)", value=0.010, step=0.001, format="%.3f")
    
    st.markdown("---")
    st.markdown("🤖 **스마트 판독 설정**")
    auto_ks = st.checkbox("천수계수 (Ks) 자동 판독 (도해 4-3)", value=True)
    if not auto_ks:
        Ks_input = st.number_input("천수계수 수동 입력 (Ks)", value=1.06, step=0.01)
    else:
        Ks_input = 1.06 
        
    auto_graph = st.checkbox("해저경사별 도표 자동 판독", value=True)
    if not auto_graph:
        graph_ratio_input = st.number_input("산정도 적용비율 수동입력 (Hmax/H'o)", value=1.78, step=0.01)
    else:
        graph_ratio_input = 1.78 
    
    calc_button = st.button("최대파고 계산 및 결과서 생성", type="primary", use_container_width=True)

with col2:
    if calc_button:
        L0 = 1.56 * (T13 ** 2)
        d_L0 = h / L0
        spm_ratio = get_interpolated_hh0(d_L0)
        H0p_spm = H13 / spm_ratio

        low, high = 1.0, 15.0
        verified_H0p = H0p_spm
        final_Ks = Ks_input
        
        for _ in range(100):
            mid = (low + high) / 2
            mid_H0p_L0 = mid / L0
            
            if auto_ks:
                current_Ks = get_shuto_ks(d_L0, mid_H0p_L0)
            else:
                current_Ks = Ks_input
                
            curr_H13, b0, b1, bM, v1, v2, v3 = calc_h13_formula(mid, h, L0, tanTheta, current_Ks)
            
            if curr_H13 < H13:
                low = mid
            else:
                high = mid
            
            verified_H0p = mid
            final_Ks = current_Ks
            if abs(curr_H13 - H13) < 0.0001:
                break
                
        H0p_L0_val = verified_H0p / L0
        h_H0p_val = h / verified_H0p
        
        Hmax_form, b0_s, b1_s, bM_s, fv1, fv2, fv3 = calc_hmax_formula(verified_H0p, h, L0, tanTheta, final_Ks)

        if auto_graph:
            graph_ratio = round(get_final_graph_ratio(h_H0p_val, H0p_L0_val, tanTheta), 4)
        else:
            graph_ratio = graph_ratio_input
                
        Hmax_graph = graph_ratio * verified_H0p
        Hmax_non_breaking = 1.8 * H13
        
        # --- 쇄파 저감 판단 및 최종 파고 선정 로직 ---
        is_breaking = (h_H0p_val <= 3.0)
        
        applied_str_graph = ""
        applied_str_form = ""
        applied_str_non = ""
        
        if is_breaking:
            final_hmax = max(Hmax_graph, Hmax_form)
            if final_hmax == Hmax_graph:
                applied_str_graph = f"🟢 **최종 적용** ($H_{{1/3}}$의 {final_hmax/H13:.2f}배)"
            else:
                applied_str_form = f"🟢 **최종 적용** ($H_{{1/3}}$의 {final_hmax/H13:.2f}배)"
        else:
            final_hmax = Hmax_non_breaking
            applied_str_non = f"🟢 **최종 적용** ($H_{{1/3}}$의 {final_hmax/H13:.2f}배)"

        with st.container():
            st.markdown("### 📊 검토 결과 요약")
            table_md = f"""
| 산정 방법 | 계산 결과 ($H_{{\\max}}$) | 비고 |
| :--- | :--- | :--- |
| **쇄파대 내 최대파고 산정도** | **{Hmax_graph:.4f} m** | {applied_str_graph if applied_str_graph else '비교용'} |
| **쇄파대 내 최대파고 약산식** | **{Hmax_form:.4f} m** | {applied_str_form if applied_str_form else '비교용'} |
| **비쇄파시 최대파고** | **{Hmax_non_breaking:.4f} m** | {applied_str_non if applied_str_non else f"참고용 ($1.8 \\times H_{{1/3}}$)"} |
            """
            st.markdown(table_md)

            if is_breaking:
                st.info(f"💡 **선정 사유:** 전면수심이 환산심해파고의 3배 이하($h/H_0' \\le 3.0$)이므로 **쇄파에 의한 저감을 고려**하여 산정도와 약산식 중 **큰 값**을 최종 선정함.")
            else:
                st.info(f"💡 **선정 사유:** 전면수심이 환산심해파고의 3배 초과($h/H_0' > 3.0$)이므로 쇄파에 의한 저감이 없다고 보아 **비쇄파파고**를 최종 선정함.")

            st.markdown("---")

            st.markdown("### 📝 상세 산출 과정")

            st.markdown("#### 1) 기본 제원 및 심해파 환산")
            st.markdown(f"- **설계유의파주기 ($T_{{1/3}}$)** = {T13} $\\mathrm{{s}}$")
            st.markdown(f"- **심해파장 ($L_0$)** = $1.56 \\times T_{{1/3}}^2$ = $1.56 \\times {T13}^2$ = **{L0:.4f} m**")
            st.markdown(f"- **파형경사 ($h/L_0$)** = {h} / {L0:.4f} = **{d_L0:.6f}**")
            st.markdown(f"- **환산심해파고 ($H_0'$)** = 역산 결과 **{verified_H0p:.4f} m** 적용 (천수계수 $K_s$ = {final_Ks:.4f} 고려)")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### 2) 쇄파 발생 여부 (쇄파 저감) 판단")
            if is_breaking:
                st.success(f"▶ **상대수심 ($h/H_0'$)** = {h} / {verified_H0p:.4f} = **{h_H0p_val:.4f}** $\\le 3.0$\n\n결과: 전면수심이 환산심해파고의 3배 이하이므로 **쇄파 저감 조건**에 해당합니다. 쇄파대 내 산정도와 약산식 산출값 중 비교 적용합니다.")
            else:
                st.warning(f"▶ **상대수심 ($h/H_0'$)** = {h} / {verified_H0p:.4f} = **{h_H0p_val:.4f}** $> 3.0$\n\n결과: 전면수심이 환산심해파고의 3배를 초과하므로 **비쇄파 조건**에 해당합니다. $1.8 H_{{1/3}}$ 을 적용합니다.")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### 3) 가) 해저경사별 쇄파대 최대파고 산정도 판독 (도참 4-18a ~ 4-19e)")
            st.info(f"""
            **[산정도 판독용 변수]**
            * 해저경사 ($\\tan\\theta$) = {tanTheta}
            * 환산심해파형경사 ($H_0'/L_0$) = {verified_H0p:.4f} / {L0:.4f} = **{H0p_L0_val:.6f}**
            * 상대수심 ($h/H_0'$) = **{h_H0p_val:.4f}**
            """)
            st.markdown(f"▶ 조건에 해당하는 산정도 곡선 자동 판독 결과: 파고비 ($H_{{\\max}}/H_0'$) = **{graph_ratio:.3f}**")
            st.success(f"▶ **산정도 $H_{{\\max}}$** = {graph_ratio:.3f} $\\times$ {verified_H0p:.4f} = **{Hmax_graph:.4f} m**")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### 4) 나) 쇄파대 내 파고 약산식을 이용한 $H_{{\\max}}$ 산정 (비교 검증용)")
            st.markdown("**① 약산식 계수 산출:**")
            st.markdown(f"- $\\beta_0^*$ = $0.052 \\times (H_0'/L_0)^{{-0.38}} \\times \\exp(20 \\times \\tan\\theta^{{1.5}})$ = **{b0_s:.6f}**")
            st.markdown(f"- $\\beta_1^*$ = $0.63 \\times \\exp(3.8 \\times \\tan\\theta)$ = **{b1_s:.6f}**")
            st.markdown(f"- $\\beta_{{\\max}}^*$ = $\\max\\left[1.65,\\ 0.53 \\times (H_0'/L_0)^{{-0.29}} \\times \\exp(2.4 \\times \\tan\\theta)\\right]$ = **{bM_s:.6f}**")

            st.markdown("**② 최대파고 조건별 계산:**")
            st.markdown(f"""
            > - **Condition 1**: $\\beta_0^* H_0' + \\beta_1^* h$ = ({b0_s:.4f} $\\times$ {verified_H0p:.4f}) + ({b1_s:.4f} $\\times$ {h}) = **{fv1:.6f} m**
            > - **Condition 2**: $\\beta_{{\\max}}^* H_0'$ = {bM_s:.4f} $\\times$ {verified_H0p:.4f} = **{fv2:.6f} m**
            > - **Condition 3**: $1.8 \\times K_s \\times H_0'$ = $1.8 \\times {final_Ks:.4f} \\times {verified_H0p:.4f}$ = **{fv3:.6f} m**
            """)
            st.success(f"▶ **약산식 $H_{{\\max}}$** = $\\min$(Condition 1, Condition 2, Condition 3) = **{Hmax_form:.6f} m**")

            st.markdown("---")

            spacer1, col_fig, spacer2 = st.columns([1.5, 2.5, 1.5])
            with col_fig:
                with st.spinner("스플라인 기반 원본 도표 생성 중..."):
                    fig = plot_authentic_chart_spline(h_H0p_val, graph_ratio, H0p_L0_val, tanTheta)
                    st.pyplot(fig, use_container_width=True)

    else:
        st.info("좌측에 제원을 확인한 후 '최대파고 계산 및 결과서 생성' 버튼을 클릭하세요.")
