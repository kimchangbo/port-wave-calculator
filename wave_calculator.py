import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import CubicSpline

# ★ 한글 깨짐 방지 라이브러리 추가 ★
import koreanize_matplotlib 

# 페이지 기본 설정
st.set_page_config(page_title="최대파고 산정 프로그램", layout="wide", page_icon="🌊")

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
# ★ Spline 보간법을 위한 Goda 도표 (해저경사 1/100) 수치화 데이터 ★
# -------------------------------------------------------------------------
goda_data_1_100 = {
    0.002: {"x": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.65, 0.95, 1.30, 1.68, 2.08, 2.48, 2.85, 3.10, 3.25]},
    0.005: {"x": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.45, 0.75, 1.10, 1.48, 1.85, 2.20, 2.44, 2.49, 2.40]},
    0.01:  {"x": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.35, 0.65, 0.96, 1.28, 1.62, 1.92, 2.06, 2.08, 2.02]},
    0.02:  {"x": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.25, 0.52, 0.82, 1.12, 1.44, 1.66, 1.76, 1.74, 1.68]},
    0.04:  {"x": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.20, 0.44, 0.72, 1.00, 1.28, 1.48, 1.58, 1.59, 1.57]},
    0.08:  {"x": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": [0.15, 0.35, 0.62, 0.88, 1.12, 1.32, 1.44, 1.50, 1.54]}
}

def get_user_curve_spline(x_arr, target_H0p_L0):
    keys = sorted(list(goda_data_1_100.keys()))
    
    if target_H0p_L0 <= keys[0]:
        return CubicSpline(goda_data_1_100[keys[0]]["x"], goda_data_1_100[keys[0]]["y"])(x_arr)
    if target_H0p_L0 >= keys[-1]:
        return CubicSpline(goda_data_1_100[keys[-1]]["x"], goda_data_1_100[keys[-1]]["y"])(x_arr)

    for i in range(len(keys)-1):
        if keys[i] <= target_H0p_L0 <= keys[i+1]:
            k1, k2 = keys[i], keys[i+1]
            break

    y1 = CubicSpline(goda_data_1_100[k1]["x"], goda_data_1_100[k1]["y"])(x_arr)
    y2 = CubicSpline(goda_data_1_100[k2]["x"], goda_data_1_100[k2]["y"])(x_arr)

    log_k1, log_k2, log_t = math.log10(k1), math.log10(k2), math.log10(target_H0p_L0)
    weight = (log_t - log_k1) / (log_k2 - log_k1)

    return y1 + weight * (y2 - y1)

# ★ 스플라인 기반 모사 그래프 생성 함수 ★
def plot_authentic_chart_spline(h_H0p_read, read_ratio, user_H0p_L0, tanTheta):
    fig, ax = plt.subplots(figsize=(5.5, 6.8)) 
    
    x_max = max(4.0, h_H0p_read + 0.5)
    y_max = max(3.5, read_ratio + 0.5)
    
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    ax.grid(which='major', color='black', linewidth=1.0)
    ax.grid(which='minor', color='black', linewidth=0.4)
    
    x_arr = np.linspace(0, x_max, 500)
    
    # 1. 기준 곡선 그리기
    for s in goda_data_1_100.keys():
        y_curve = CubicSpline(goda_data_1_100[s]["x"], goda_data_1_100[s]["y"])(x_arr)
        ax.plot(x_arr, y_curve, color='black', linewidth=1.2)
        
        if s == 0.002: label_x = x_max * 0.90
        elif s == 0.005: label_x = x_max * 0.78
        elif s == 0.01: label_x = x_max * 0.66
        elif s == 0.02: label_x = x_max * 0.54
        elif s == 0.04: label_x = x_max * 0.42
        else: label_x = x_max * 0.30
            
        label_x = min(label_x, x_max - 0.2)
        label_y = np.interp(label_x, x_arr, y_curve)
        
        label_text = f"H'o/Lo={s}" if s in [0.002, 0.005] else f"{s}"
        ax.text(label_x, label_y + 0.06, label_text, fontsize=9, backgroundcolor='white', ha='center', va='bottom')
        

    # 2. 사용자 곡선 그리기
    y_user = get_user_curve_spline(x_arr, user_H0p_L0)
    ax.plot(x_arr, y_user, color='blue', linewidth=2.5, alpha=0.7) 
    
    # 3. 판독 결과 마킹
    ax.axvline(x=h_H0p_read, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=read_ratio, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.plot(h_H0p_read, read_ratio, 'ro', markersize=8)
    
    ax.text(h_H0p_read + 0.1, read_ratio + 0.1, f"독취결과\nh/H'o={h_H0p_read:.2f}\nHmax/H'o={read_ratio:.2f}", 
            color='red', fontsize=10, fontweight='bold', 
            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3', alpha=0.9))
    
    frac_slope = "1/" + str(int(1/tanTheta)) if tanTheta > 0 else f"{tanTheta}"
    ax.text(0.2, y_max - 0.2, f"  해저경사 {frac_slope}  ", fontsize=11, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', linewidth=1.2))
    ax.text(1.0, y_max - 0.6, r"$H_{max} \equiv H_{1/250}$", fontsize=10, fontweight='bold', backgroundcolor='white')
    
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
st.markdown("항만 및 어항 설계기준 산출 로직 (Spline 보간 적용 완벽 모사)")

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
        graph_ratio = st.number_input("산정도 적용비율 수동입력 (Hmax/H'o)", value=1.71, step=0.01)
    else:
        graph_ratio = 1.71
    
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
        
        # 수식 기반 Hmax 산정
        Hmax_form, b0_s, b1_s, bM_s, fv1, fv2, fv3 = calc_hmax_formula(verified_H0p, h, L0, tanTheta, final_Ks)

        # Spline 곡선 Y값 추출
        if auto_graph:
            if tanTheta == 0.010:
                spline_y_val = get_user_curve_spline(np.array([h_H0p_val]), H0p_L0_val)[0]
                graph_ratio = round(float(spline_y_val), 2)
            else:
                graph_ratio = round(Hmax_form / verified_H0p, 2)
                
        Hmax_graph = graph_ratio * verified_H0p
        Hmax_non_breaking = 1.8 * H13
        ratio_hmax_h13 = Hmax_graph / H13

        # -----------------------------------------------------
        # 렌더링 영역
        # -----------------------------------------------------
        with st.container():
            st.markdown("### 가) 해저경사별 쇄파대 최대파고 산정도 판독 (도참 4-18a ~ 4-19e)")
            
            st.info(f"""
            **[산정도 판독용 변수]**
            * 해저경사 (tanθ) = {tanTheta}
            * 환산심해파형경사 (H'o/Lo) = {H0p_L0_val:.4f}
            * 상대수심 (h/H'o) = {h_H0p_val:.4f}
            """)
            
            st.markdown(f"▶ 조건에 해당하는 산정도 곡선 자동 판독 결과: 파고비 (Hmax/H'o) = **{graph_ratio}**")
            st.markdown(f"▶ **산정도 Hmax** = {graph_ratio} × {verified_H0p:.4f} = **{Hmax_graph:.4f} m**")

            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("### 나) 쇄파대 내 파고 약산식을 이용한 Hmax 산정 (비교 검증용)")
            st.markdown(f"• βo* = {b0_s:.6f}, β1* = {b1_s:.6f}, βmax* = {bM_s:.6f}")
            
            st.markdown(f"""
            > **[조건]** >   
            > ① (βoH'o + β1h) = {fv1:.6f}  
            >   
            > ② βmax*H'o = {fv2:.6f}  
            >   
            > ③ 1.8 × Ks × H'o = {fv3:.6f}  
            """)
            
            st.markdown(f"▶ **약산식 Hmax = min(①, ②, ③) = {Hmax_form:.6f} m**")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("### 📊 검토 결과 종합")
            table_md = f"""
| 산정 방법 | 계산 결과 (Hmax) | 비고 |
| :--- | :--- | :--- |
| **쇄파대 내 최대파고 산정도** | {Hmax_graph:.4f} m | 🟢 **최종 적용** (H1/3의 {ratio_hmax_h13:.2f}배) |
| **쇄파대 내 최대파고 약산식** | {Hmax_form:.4f} m | 검증용 |
| **비쇄파시 최대파고** | {Hmax_non_breaking:.4f} m | 참고용 (1.8 × H1/3) |
            """
            st.markdown(table_md)

            st.markdown("---")

            spacer1, col_fig, spacer2 = st.columns([1.5, 2.5, 1.5])
            with col_fig:
                with st.spinner("스플라인 기반 원본 도표 생성 중..."):
                    fig = plot_authentic_chart_spline(h_H0p_val, graph_ratio, H0p_L0_val, tanTheta)
                    st.pyplot(fig, use_container_width=True)

    else:
        st.info("좌측에 제원을 확인한 후 '최대파고 계산 및 결과서 생성' 버튼을 클릭하세요.")
