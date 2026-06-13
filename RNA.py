import streamlit as st
import pandas as pd
import joblib

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="RNA-Seq 암 예측 시스템",
    page_icon="🧬",
    layout="wide"
)

# =========================
# 스타일
# =========================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

div[data-testid="stSlider"] {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #e8f5e9;
    border-left: 8px solid #4CAF50;
    font-size: 24px;
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 모델 및 컬럼 불러오기
# =========================
rf_model = joblib.load("RNA_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =========================
# 암 이름 한글 변환
# =========================
cancer_mapping = {
    "BRCA": "유방암",
    "COAD": "대장암",
    "KIRC": "신장암",
    "LUAD": "폐선암",
    "PRAD": "전립선암"
}

# =========================
# 유전자 이름 매핑
# =========================
virtual_mapping = {
    "gene_17801": "BRCA1",
    "gene_2747": "BRCA2",
    "gene_5578": "TP53",
    "gene_7116": "ERBB2 (HER2)",
    "gene_7073": "ESR1",
    "gene_220": "PGR",
    "gene_7896": "MKI67",
    "gene_13210": "PTEN",
    "gene_16342": "PIK3CA",
    "gene_18746": "CDH1"
}

# =========================
# 중요 유전자
# =========================
top_10_features = [
    "gene_17801",
    "gene_2747",
    "gene_5578",
    "gene_7116",
    "gene_7073",
    "gene_220",
    "gene_7896",
    "gene_13210",
    "gene_16342",
    "gene_18746"
]

# =========================
# 제목
# =========================
st.title("🧬 AI 기반 RNA-Seq 암 종류 예측 시스템")

st.markdown("""
상위 중요 유전자의 발현량을 입력하면  
인공지능(Random Forest)이 암 종류를 예측합니다.
""")

st.divider()

# =========================
# 유전자 입력
# =========================
st.subheader("🔬 유전자 발현량 입력")

user_values = {}

col1, col2 = st.columns(2)

for i, gene in enumerate(top_10_features):

    display_name = virtual_mapping.get(gene, gene)

    current_col = col1 if i % 2 == 0 else col2

    with current_col:

        st.markdown(f"**🧬 {display_name}**")

        user_values[gene] = st.slider(
            label="",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            key=gene
        )

st.divider()

# =========================
# 예측 버튼
# =========================
if st.button("🩺 암 종류 예측", use_container_width=True):

    # 전체 유전자 구조 생성
    input_data = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    # 입력값 삽입
    for gene, value in user_values.items():
        input_data[gene] = value

    # 예측
    prediction = rf_model.predict(input_data)[0]
    probabilities = rf_model.predict_proba(input_data)[0]

    # 한글 변환
    prediction_ko = cancer_mapping.get(
        prediction,
        prediction
    )

    # =========================
    # 예측 결과
    # =========================
    st.markdown(
        f"""
        <div class="result-box">
        🧬 예측 결과 : {prediction_ko}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # 확률 데이터
    # =========================
    prob_df = pd.DataFrame({
        "암 종류": [
            cancer_mapping.get(c, c)
            for c in rf_model.classes_
        ],
        "확률(%)": probabilities * 100
    })

    prob_df = prob_df.sort_values(
        by="확률(%)",
        ascending=False
    )

    left, right = st.columns([1, 2])

    # 표
    with left:
        st.subheader("📋 예측 확률")

        st.dataframe(
            prob_df.style.format({
                "확률(%)": "{:.2f}"
            }),
            use_container_width=True
        )

    # 그래프
    with right:
        st.subheader("📊 암 종류별 예측 확률")

        st.bar_chart(
            prob_df.set_index("암 종류"),
            height=450
        )
