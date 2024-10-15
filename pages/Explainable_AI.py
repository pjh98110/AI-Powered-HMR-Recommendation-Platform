import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components
# import eli5
# from eli5.sklearn import PermutationImportance
# from omnixai.explainers.tabular import TabularExplainer

# 데이터 로드
DATA_PATH = "./"
try:
    data = pd.read_excel(f"{DATA_PATH}생성형 AI 활용 부문_소비자 평판 정보 데이터 샘플.xlsx")
except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 데이터 파일의 경로를 확인하세요.")
    st.stop()

# 한글 폰트 설정 함수
def set_korean_font():
    font_path = f"{DATA_PATH}NanumGothic.ttf"  # 폰트 파일 경로

    from matplotlib import font_manager, rc
    font_manager.fontManager.addfont(font_path)
    rc('font', family='NanumGothic')

# 한글 폰트 설정 적용
set_korean_font()

# 데이터 샘플링 (속도 개선을 위해 샘플링 비율 조정)
sample_fraction = 1
data_sampled = data.sample(frac=sample_fraction, random_state=42)

# 타겟 선택
target_options = ['P1_성별', 'P2_연령대', 'P3_거주지역', 'P4_가구원수', 'P5_간편식_구매빈도']
selected_target = st.sidebar.selectbox(
    "분류 모델의 타겟 변수를 선택하세요.",
    target_options
)

# 페이지 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "Home"

# 페이지 헤더
st.markdown(f"""
<span style='font-size: 24px;'>
<div style=" color: #000000;">
<strong>Explainer Dashboard</strong>
</div>
""", unsafe_allow_html=True)

# XAI 선택 박스
selected_xai = st.selectbox(
    label="원하는 Explainable AI(XAI)를 선택하세요.",
    options=["SHAP", "LIME"], # ELI5 OmniXAI
    placeholder="하나를 선택하세요.",
    help="XAI는 사용자가 머신러닝 알고리즘으로 생성된 결과를 쉽게 이해할 수 있도록 도와주는 프로세스와 방법입니다.",
    key="xai_key"
)

# 분석 실행 버튼
if st.button("분석 실행"):
    # 데이터 준비
    # Label Encoding을 사용하여 범주형 데이터를 수치형으로 변환
    label_encoders = {}
    for column in data_sampled.columns:
        if data_sampled[column].dtype == 'object':
            le = LabelEncoder()
            data_sampled[column] = le.fit_transform(data_sampled[column])
            label_encoders[column] = le

    # 분석에 사용할 피처와 타겟 변수
    X = data_sampled.drop(columns=[selected_target])
    y = data_sampled[selected_target]
    data_sampled[selected_target] = y  # 타겟을 다시 데이터프레임에 추가

    # 모델 훈련 (예시로 RandomForestClassifier 사용)
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X, y)

    # SHAP 분석 실행
    if selected_xai == "SHAP":
        st.write("SHAP 분석을 실행합니다...")

        # SHAP 분석
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # SHAP summary plot
        st.subheader("SHAP Summary Plot")
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="dot", show=False)
        st.pyplot(fig1)

        # SHAP 해석 설명
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)**: 
        - SHAP는 모델 예측에 대한 각 피처의 기여도를 계산합니다.
        - Summary Plot은 전체 피처의 중요도를 시각화합니다.
        """)

    # LIME 분석 실행
    elif selected_xai == "LIME":
        st.write("LIME 분석을 실행합니다...")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X),
            feature_names=X.columns,
            mode='classification'
        )
        
        # 샘플 데이터 예측 및 설명
        i = np.random.randint(0, X.shape[0])
        exp = explainer.explain_instance(X.iloc[i], model.predict_proba, num_features=10)
        st.write(f"Instance {i} 설명:")
        components.html(exp.as_html(), height=500)

        # LIME 해석 설명
        st.markdown("""
        **LIME (Local Interpretable Model-agnostic Explanations)**:
        - LIME은 모델 예측을 쉽게 해석할 수 있도록 하는 도구입니다.
        - 특정 데이터 포인트에 대해 모델이 내린 예측을 설명합니다.
        - 각 피처가 예측에 미친 영향을 보여줍니다.
        """)

    # # ELI5 분석 실행
    # elif selected_xai == "ELI5":
    #     st.write("ELI5 분석을 실행합니다...")

    #     perm = PermutationImportance(model, random_state=42).fit(X, y)
    #     html_obj = eli5.show_weights(perm, feature_names=X.columns.tolist())
        
    #     st.subheader("ELI5 Permutation Importance")
    #     components.html(html_obj.data, height=500)

    #     # ELI5 해석 설명
    #     st.markdown("""
    #     **ELI5 (Explain Like I'm 5)**:
    #     - ELI5는 모델 예측을 쉽게 이해할 수 있도록 설명합니다.
    #     - Permutation Importance는 각 피처의 중요도를 측정하여 예측 성능에 미치는 영향을 평가합니다.
    #     """)

    # # Omni XAI 분석 실행
    # elif selected_xai == "OmniXAI":
    #     st.write("Omni XAI 분석을 실행합니다...")

    #     # Omni XAI 사용
    #     omni_explainer = TabularExplainer(
    #         training_data=X, 
    #         model=model, 
    #         mode="classification"
    #     )
    #     omni_explanation = omni_explainer.explain(X.iloc[[i]])

    #     # 설명 결과 표시
    #     st.subheader("Omni XAI Explanation")
    #     st.write(omni_explanation)
        
    #     # Omni XAI 해석 설명
    #     st.markdown("""
    #     **Omni XAI**:
    #     - Omni XAI는 다양한 XAI 기법을 통합한 프레임워크입니다.
    #     - 모델 예측에 대해 더 넓은 시각을 제공하며, 여러 해석 방법을 결합하여 설명할 수 있습니다.
    #     """)

