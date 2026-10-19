import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "K-Means 군집화를 활용하여 영화 데이터를 유형별로 분류하고 분석합니다."
)


# 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")

    total_count = len(df)

    # 필요한 데이터 수치형 변환 및 결측치/0값 처리
    cols_to_num = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    for col in cols_to_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 값이 없거나 first_week_audi가 0인 데이터 제거
    df_clean = df.dropna(subset=cols_to_num).copy()
    df_clean = df_clean[df_clean["first_week_audi"] > 0].copy()

    # 속성 생성 및 변환
    # 1. 스크린 수 (상용로그)
    df_clean["log_first_scrn"] = np.log10(
        df_clean["first_scrn"].replace(0, np.nan)
    )
    # 2. 누적 관객 (상용로그)
    df_clean["log_total_audi"] = np.log10(
        df_clean["total_audi"].replace(0, np.nan)
    )
    # 3. 롱런 지수 (누적 관객 / 첫 주 관객, 최대 20으로 클리핑)
    df_clean["long_run_index"] = (
        df_clean["total_audi"] / df_clean["first_week_audi"]
    )
    df_clean["long_run_index"] = df_clean["long_run_index"].clip(upper=20)

    # 4. 10위권 일수 (days_in_top10는 그대로 사용)

    # 생성된 로그 속성 결측치 추가 제거 (안전 조치)
    df_clean = df_clean.dropna(
        subset=[
            "log_first_scrn",
            "log_total_audi",
            "long_run_index",
            "days_in_top10",
        ]
    )

    filtered_count = len(df_clean)

    return df_clean, total_count, filtered_count


# 데이터 로드
df, total_count, filtered_count = load_and_preprocess_data()

# 편수 표시
st.info(
    f"📊 **전체 영화 수:** {total_count:,}편 | **분석에 사용된 묶은 영화 수:** {filtered_count:,}편"
)

# 속성 매핑 설정
feature_dict = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

# 1. 군집화 속성 선택
st.subheader("1. 군집화에 사용할 속성 선택")
selected_feature_labels = st.multiselect(
    "군집화에 사용할 속성을 2개 이상 선택하세요 (기본값: 전체 선택):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys()),
)

if len(selected_feature_labels) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 합니다.")
    st.stop()

selected_features = [feature_dict[label] for label in selected_feature_labels]

# 2. 군집화 수행
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means (k=3, random_state 고정)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df["raw_cluster"] = cluster_labels

# 누적 관객(total_audi) 평균에 따른 묶음 라벨링 (㉮, ㉯, ㉰)
cluster_audi_mean = (
    df.groupby("raw_cluster")["total_audi"].mean().sort_values(ascending=False)
)
rank_to_symbol = {0: "㉮", 1: "㉯", 2: "㉰"}
cluster_mapping = {
    old_cluster: rank_to_symbol[rank]
    for rank, old_cluster in enumerate(cluster_audi_mean.index)
}

df["cluster"] = df["raw_cluster"].map(cluster_mapping)

# 시각화용 색상 지도 설정
color_discrete_map = {"㉮": "#EF553B", "㉯": "#00CC96", "㉰": "#636EFA"}

st.markdown("---")

# 3. 2차원 산점도
st.subheader("2. 2차원 산점도")
col1, col2 = st.subplots(2)
with col1:
    x_2d_label = st.selectbox(
        "2D 가로축(X축) 속성",
        options=selected_feature_labels,
        index=0,
        key="2d_x",
    )
with col2:
    y_2d_label = st.selectbox(
        "2D 세로축(Y축) 속성",
        options=selected_feature_labels,
        index=min(1, len(selected_feature_labels) - 1),
        key="2d_y",
    )

fig_2d = px.scatter(
    df,
    x=feature_dict[x_2d_label],
    y=feature_dict[y_2d_label],
    color="cluster",
    hover_name="movieNm",
    labels={
        feature_dict[x_2d_label]: x_2d_label,
        feature_dict[y_2d_label]: y_2d_label,
        "cluster": "묶음",
    },
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
    color_discrete_map=color_discrete_map,
    title=f"2D 산점도 ({x_2d_label} vs {y_2d_label})",
)
fig_2d.update_traces(marker=dict(size=6, opacity=0.8))
st.plotly_chart(fig_2d, use_container_width=True)

st.markdown("---")

# 4. 3차원 산점도
st.subheader("3. 3차원 산점도")
if len(selected_feature_labels) < 3:
    st.info(
        "💡 3차원 산점도를 그리려면 위에서 속성을 3개 이상 선택해 주세요."
    )
else:
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        x_3d_label = st.selectbox(
            "3D X축 속성",
            options=selected_feature_labels,
            index=0,
            key="3d_x",
        )
    with col_y:
        y_3d_label = st.selectbox(
            "3D Y축 속성",
            options=selected_feature_labels,
            index=1,
            key="3d_y",
        )
    with col_z:
        z_3d_label = st.selectbox(
            "3D Z축 속성",
            options=selected_feature_labels,
            index=2,
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=feature_dict[x_3d_label],
        y=feature_dict[y_3d_label],
        z=feature_dict[z_3d_label],
        color="cluster",
        hover_name="movieNm",
        labels={
            feature_dict[x_3d_label]: x_3d_label,
            feature_dict[y_3d_label]: y_3d_label,
            feature_dict[z_3d_label]: z_3d_label,
            "cluster": "묶음",
        },
        category_orders={"cluster": ["㉮", "㉯", "㉰"]},
        color_discrete_map=color_discrete_map,
        title=f"3D 산점도 ({x_3d_label} x {y_3d_label} x {z_3d_label})",
    )
    fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 5. 묶음별 요약 통계량 (원래 단위 기준)
st.subheader("4. 묶음별 편수 및 속성 평균 (원래 단위)")

summary_df = (
    df.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        _10위권일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_index", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
)

# 표 포맷팅
formatted_summary = summary_df.copy()
formatted_summary["편수"] = formatted_summary["편수"].apply(
    lambda x: f"{x:,}편"
)
formatted_summary["스크린수_평균"] = formatted_summary[
    "스크린수_평균"
].apply(lambda x: f"{x:,.1f}개")
formatted_summary["누적관객_평균"] = formatted_summary[
    "누적관객_평균"
].apply(lambda x: f"{x:,.0f}명")
formatted_summary["_10위권일수_평균"] = formatted_summary[
    "_10위권일수_평균"
].apply(lambda x: f"{x:,.1f}일")
formatted_summary["롱런지수_평균"] = formatted_summary[
    "롱런지수_평균"
].apply(lambda x: f"{x:,.2f}")

formatted_summary.columns = [
    "편수",
    "스크린 수 평균",
    "누적 관객 평균",
    "10위권 일수 평균",
    "롱런 지수 평균",
]

st.dataframe(formatted_summary, use_container_width=True)

st.markdown("---")

# 6. 묶음별 누적 관객 수 TOP 5 영화
st.subheader("5. 묶음별 누적 관객 수 TOP 5 영화")

cols = st.columns(3)
for idx, cluster_symbol in enumerate(["㉮", "㉯", "㉰"]):
    with cols[idx]:
        st.markdown(f"### 묶음 {cluster_symbol}")
        top5_movies = df[df["cluster"] == cluster_symbol].nlargest(
            5, "total_audi"
        )
        for i, (_, row) in enumerate(top5_movies.iterrows(), 1):
            st.write(f"**{i}. {row['movieNm']}** ({row['total_audi']:,}명)")
