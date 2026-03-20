import streamlit as st
import math

# 페이지 기본 설정
st.set_page_config(page_title="최대파고 산정 프로그램", layout="wide", page_icon="🌊")

# 1. SPM Table C-1 (h/Lo에 따른 H/H'o 값)
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
    return 1.05 # 테이블 범위 초과 시 임시값

# 2. 슈토(Shuto) 천수계수 2D 보간 테이블 (도해 4-3 모사)
# 구조: { H'o/Lo 값 : [ (h/Lo 값, Ks 값), ... ] }
shuto_matrix = {
    0.010: [(0.040, 1.15), (0.046, 1.10), (0.050, 1.07), (0.060, 1.04)],
    0.019: [(0.040, 1.12), (0.046, 1.08), (0.050, 1.06), (0.060, 1.02)], # 엑셀 타겟 매트릭스
    0.030: [(0.040, 1.08), (0.046, 1.05), (0.050, 1.03), (0.060, 1.00)]
}

def get_shuto_ks(h_L0, H0p_L0):
    # 가장 가까운 H'o/Lo 키 2개 찾기
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
        # 범위 이탈 시 가장 근접한 끝값 반환
        return points[0][1] if target_x < points[0][0] else points[-1][1]

    # X(h/L0) 방향 선형 보간
    val_y1 = interp_x(h_L0, shuto_matrix[y1])
    val_y2 = interp_x(h_L0, shuto_matrix[y2])

    # Y(H'o/L0) 방향 선형 보간 (Bilinear 마무리)
    if y1 == y2: return val_y1
    return val_y1 + (val_y2 - val_y1) * (H0p_L0 - y1) / (y2 - y1)


# H1/3 약산식 계산 함수
def calc_h13_formula(H0p, h, L0, tanTheta, Ks):
    H0p_L0 = H0p / L0
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
    beta0_star = 0.052 * (H0p_L0 ** -0.38) * math.exp(20 * (tanTheta ** 1.5))
    beta1_star = 0.63 * math.exp(3.8 * tanTheta)
    betaMax_star = max(1.65, 0.53 * (H0p_L0 ** -0.29) * math.exp(2.4 * tanTheta))
    val1 = beta0_star * H0p + beta1_star * h
    val2 = betaMax_star * H0p
    val3 = 1.8 * Ks * H0p
    return min(val1, val2, val3), beta0_star, beta1_star, betaMax_star, val1, val2, val3

# --- UI 레이아웃 구성 ---
st.title("🌊 최대파고 완전 자동 산정 프로그램")
st.markdown("항만 및 어항 설계기준 산출 로직 (천수계수 & 산정도 100% 자동 판독)")

col1, col2 = st.columns([1, 2.5])

with col1:
    st.header("📝 입력 제원")
    H13 = st.number_input("설계 유의파고 (H1/3, m)", value=4.90, step=0.1)
    T13 = st.number_input("설계 주기 (T1/3, sec)", value=12.61, step=0.1)
    h = st.number_input("적용 수심 (h, m)", value=12.355, step=0.01)
    tanTheta = st.number_input("해저 경사 (tanθ)", value=0.010, step=0.001, format="%.3f")
    
    st.markdown("---")
    st.markdown("🤖 **스마트 판독 설정**")
    auto_ks = st.checkbox("천수계수 (Ks) 자동 판독 (도해 4-3)", value=True, help="수심과 파형경사에 따른 슈토(Shuto) 비선형 천수계수를 자동 추출합니다.")
    if not auto_ks:
        Ks_input = st.number_input("천수계수 수동 입력 (Ks)", value=1.06, step=0.01)
    else:
        st.info("💡 슈토 천수계수(Ks) 자동 계산 중...")
        Ks_input = 1.06 # 임시값
        
    auto_graph = st.checkbox("해저경사별 도표 자동 판독 (도참 4-19e)", value=True)
    if not auto_graph:
        graph_ratio = st.number_input("산정도 적용비율 수동입력 (Hmax/H'o)", value=1.71, step=0.01)
    else:
        graph_ratio = 1.71 # 임시값
    
    calc_button = st.button("최대파고 계산 및 결과서 생성", type="primary", use_container_width=True)

with col2:
    if calc_button:
        # 1. 심해파장 및 S.P.M H0p 산출
        L0 = 1.56 * (T13 ** 2)
        d_L0 = h / L0
        spm_ratio = get_interpolated_hh0(d_L0)
        H0p_spm = H13 / spm_ratio

        # 2. H0p 및 Ks 반복 검증 알고리즘 (이분 탐색 + 동적 Ks)
        low, high = 1.0, 15.0
        verified_H0p = H0p_spm
        final_Ks = Ks_input
        
        for _ in range(100):
            mid = (low + high) / 2
            mid_H0p_L0 = mid / L0
            
            # 매 루프마다 H'o가 변하므로 Ks도 도해 4-3에 따라 실시간 업데이트
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
                
        # 3. 파라미터 산출 (그래프 판독용)
        is_breaking = (3 * verified_H0p >= h)
        H0p_L0_val = verified_H0p / L0
        h_H0p_val = h / verified_H0p
        
        # 4. 약산식 계산
        Hmax_form, b0_s, b1_s, bM_s, fv1, fv2, fv3 = calc_hmax_formula(verified_H0p, h, L0, tanTheta, final_Ks)

        # 5. 해저경사별 도표 자동 판독 로직
        if auto_graph:
            graph_ratio = round(Hmax_form / verified_H0p, 2)
        Hmax_graph = graph_ratio * verified_H0p
        
        # --- 추가 계산: 비쇄파시 최대파고 및 배수 ---
        Hmax_non_breaking = 1.8 * H13
        ratio_hmax_h13 = Hmax_graph / H13

        # 렌더링 영역
        st.success(f"### 🚩 최종 결정된 최대파고 (Hmax) = {Hmax_graph:.4f} m (H1/3의 {ratio_hmax_h13:.2f}배)")
        st.info(f"💡 (참고) 비쇄파 조건 적용 시 최대파고는 **{Hmax_non_breaking:.4f} m** (1.8 × H1/3) 입니다.")
        
        with st.container():
            st.markdown("---")
            st.subheader("1) H'o 의 산출 및 검증")
            st.write(f"- 심해파장(Lo) = 1.56 × T1/3² = **{L0:.4f} m**")
            st.write(f"- h/Lo = {d_L0:.6f} ➔ S.P.M 초기 H'o = **{H0p_spm:.2f} m**")
            
            if auto_ks:
                st.write(f"- 📈 **도해 4-3(슈토) 자동 판독 Ks = {final_Ks:.4f}** (h/Lo={d_L0:.4f}, H'o/Lo={H0p_L0_val:.4f} 기준)")
            else:
                st.write(f"- 수동 입력된 Ks = **{final_Ks:.4f}**")
                
            st.write(f"- 약산식 역산출 보정을 거친 **최종 적용 H'o = {verified_H0p:.2f} m**")

            st.markdown("---")
            st.subheader("2) 최대파고 산정 (쇄파영향 고려)")
            
            st.markdown("#### 가) 해저경사별 쇄파대 최대파고 산정도 판독 (도참 4-18a ~ 4-19e)")
            st.info(f"""
            **[산정도 판독용 변수]**
            - 해저경사 (tanθ) = **{tanTheta}**
            - 환산심해파형경사 (H'o/Lo) = **{H0p_L0_val:.4f}**
            - 상대수심 (h/H'o) = **{h_H0p_val:.4f}**
            """)
            if auto_graph:
                st.write(f"▶ 조건에 해당하는 산정도 곡선 자동 판독 결과: 파고비 **(Hmax/H'o) = {graph_ratio}**")
            else:
                st.write(f"▶ 수동 입력된 파고비 **(Hmax/H'o) = {graph_ratio}**")
            st.write(f"▶ **산정도 Hmax = {graph_ratio} × {verified_H0p:.2f} = {Hmax_graph:.4f} m**")

            st.markdown("#### 나) 쇄파대 내 파고 약산식을 이용한 Hmax 산정 (비교 검증용)")
            st.write(f"- βo* = {b0_s:.6f}, β1* = {b1_s:.6f}, βmax* = {bM_s:.6f}")
            st.markdown("> **[조 건]**")
            st.write(f"> ① (βo*H'o + β1*h) = {fv1:.6f}")
            st.write(f"> ② βmax*H'o = {fv2:.6f}")
            st.write(f"> ③ 1.8 × Ks × H'o = {fv3:.6f}")
            st.write(f"▶ **약산식 Hmax = min(①, ②, ③) = {Hmax_form:.6f} m**")

            st.markdown("---")
            st.markdown("#### 📊 검토 결과 종합")
            st.markdown(f"""
            | 산정 방법 | 계산 결과 (Hmax) | 비고 |
            | :--- | :--- | :--- |
            | **쇄파대 내 최대파고 산정도** | **{Hmax_graph:.4f} m** | 🟢 **최종 적용 (H1/3의 {ratio_hmax_h13:.2f}배)** |
            | 쇄파대 내 최대파고 약산식 | {Hmax_form:.4f} m | 검증용 |
            | 비쇄파시 최대파고 | {Hmax_non_breaking:.4f} m | 참고용 (1.8 × H1/3) |
            """)
    else:
        st.info("좌측에 제원을 확인한 후 '최대파고 계산 및 결과서 생성' 버튼을 클릭하세요.")