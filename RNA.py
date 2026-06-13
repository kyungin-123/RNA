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

.stApp {
    background: linear-gradient(
        135deg,
        #e3f2fd 0%,
        #f8fbff 50%,
        #ffffff 100%
    );
}

h1 {
    color: #1565c0;
}

div[data-testid="stSlider"] {
    background: white;
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}

.result-box {
    background: white;
    border-radius: 15px;
    padding: 25px;
    border-left: 8px solid #2196f3;
    font-size: 30px;
    font-weight: bold;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.info-card {
    background: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.stButton > button {
    width: 100%;
    height: 65px;
    font-size: 24px;
    font-weight: bold;
    border-radius: 15px;
    background-color: #1976d2;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #1565c0;
    color: white;
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
# 암 설명
# =========================
cancer_info = {
    "유방암": "유방 조직에서 발생하는 암으로 여성에게 가장 흔한 암 중 하나입니다.",
    "대장암": "대장 또는 직장에서 발생하는 암입니다.",
    "신장암": "신장에서 발생하는 암으로 혈뇨가 주요 증상입니다.",
    "폐선암": "폐암의 한 종류로 비흡연자에게도 발생할 수 있습니다.",
    "전립선암": "전립선에 발생하는 암으로 남성에게 주로 나타납니다."
}

# =========================
# 유전자 이름
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
# 헤더
# =========================
col1, col2 = st.columns([1,4])

with col1:
    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQID783B0W4sRPjqL413ROELoaUFndd3xnRsw0wIQ4JvIbUtuc4pYcJQK8&s=10",
        width=180
    )

with col2:
    st.title("🧬 AI 기반 RNA-Seq 암 종류 예측 시스템")
    st.markdown("""
    ### 의료 인공지능을 활용한 암 진단 보조 서비스
    RNA 발현 데이터를 기반으로 암 종류를 예측합니다.
    """)

st.divider()

# =========================
# 유전자 입력
# =========================
st.subheader("🔬 유전자 발현량 입력")

user_values = {}

left_col, right_col = st.columns(2)

for i, gene in enumerate(top_10_features):

    display_name = virtual_mapping.get(gene, gene)

    current_col = left_col if i % 2 == 0 else right_col

    with current_col:

        st.markdown(f"**🧬 {display_name}**")

        user_values[gene] = st.slider(
            "",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            key=gene
        )

st.write("")

# =========================
# 예측 버튼
# =========================
if st.button("🔍 암 종류 예측"):

    input_data = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    for gene, value in user_values.items():
        input_data[gene] = value

    prediction = rf_model.predict(input_data)[0]
    probabilities = rf_model.predict_proba(input_data)[0]

    prediction_ko = cancer_mapping.get(
        prediction,
        prediction
    )

    max_prob = max(probabilities) * 100

    # =========================
    # 결과 카드
    # =========================
    st.markdown(
        f"""
        <div class="result-box">
            🏆 예측 결과 : {prediction_ko}<br>
            발병률 : {max_prob:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    if prediction_ko in cancer_info:
        st.markdown(
            f"""
            <div class="info-card">
            <h3>📖 암 정보</h3>
            {cancer_info[prediction_ko]}
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

    col1, col2 = st.columns([1, 2])

    with col1:

        st.subheader("📋 예측 확률")

        st.dataframe(
            prob_df.style.format({
                "확률(%)": "{:.2f}"
            }),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 암 종류별 예측 확률")

        st.bar_chart(
            prob_df.set_index("암 종류"),
            height=450
        )

    st.success("예측이 완료되었습니다.")
