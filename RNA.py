import streamlit as st
import pandas as pd
import joblib

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
# 가상 유전자 매핑
# =========================
virtual_mapping = {
    "gene_17801": "BRCA1",
    "gene_2747": "BRCA2",
    "gene_5578": "TP53",
    "gene_7116": "ERBB2(HER2)",
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
# 웹앱 화면
# =========================
st.title("🧬AI 기반 RNA-Seq 암 종류 예측 시스템")

st.write(
    "상위 중요 유전자의 발현량을 입력하면 "
    "AI가 암 종류를 예측합니다."
)

# =========================
# 입력창 생성
# =========================
user_values = {}

for gene in top_10_features:
    display_name = virtual_mapping.get(gene, gene)

    user_values[gene] = st.number_input(
        f"{display_name} 발현량",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

# =========================
# 예측 버튼
# =========================
if st.button("암 종류 예측"):

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

    # 예측 결과 한글 변환
    prediction_ko = cancer_mapping.get(
        prediction,
        prediction
    )

    st.success(
        f"예측 암 종류: {prediction_ko}"
    )

    st.subheader("암 종류별 예측 확률")

    # 확률표 생성
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

    # 표 출력
    st.dataframe(
        prob_df,
        use_container_width=True
    )

    # 그래프 출력
    st.bar_chart(
        prob_df.set_index("암 종류")
    )