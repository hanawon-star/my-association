import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. App Title & Browser Tab Config
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "K-Means 군집화를 활용하여 영화 데이터를 유형별로 분류하고 분석합니다."
)


# 2. Data Loading & Preprocessing
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")

    total_count = len(df)

    # Convert numeric columns safely
    cols_to_num = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    for col in cols_to_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing values and invalid first_week_audi / positive values for log
    df_clean = df.dropna(subset=cols_to_num).copy()
    df_clean = df_clean[
        (df_clean["first_week_audi"] > 0)
        & (df_clean["first_scrn"] > 0)
        & (df_clean["total_audi"] > 0)
    ].copy()

    # Derived features & feature transformations
    df_clean["log_first_scrn"] = np.log10(df_clean["first_scrn"])
    df_clean["log_total_audi"] = np.log10(df_clean["total_audi"])
    df_clean["long_run_idx"] = np.minimum(
        df_clean["total_audi"] / df_clean["first_week_audi"], 20
    )
    # days_in_top10 is used as-is

    filtered_count = len(df_clean)

    return df_clean, total_count, filtered_count


df, total_count, filtered_count = load_and_preprocess_data()

# Display total and filtered movie count in one line
st.info(
    f"📊 **전체 영화:** {total_count:,}편 | **분석 대상 영화:** {filtered_count:,}편"
)

# Feature mapping dictionary
feature_dict = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_idx",
}

# 3. Feature Selection & Clustering
st.subheader("1. 군집화 속성 선택")
selected_feature_labels = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (최소 2개 이상 선택, 기본값: 전체 선택):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys()),
)

if len(selected_feature_labels) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 군집화를 수행할 수 있습니다.")
    st.stop()

selected_features = [feature_dict[label] for label in selected_feature_labels]

# Feature Standardization & KMeans
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df["raw_cluster"] = cluster_labels

# Cluster label renaming rule based on mean total_audi (descending: ㉮, ㉯, ㉰)
cluster_audi_mean = (
    df.groupby("raw_cluster")["total_audi"].mean().sort_values(ascending=False)
)
symbol_mapping = {
    old_cluster: symbol
    for old_cluster, symbol in zip(cluster_audi_mean.index, ["㉮", "㉯", "㉰"])
}
df["cluster"] = df["raw_cluster"].map(symbol_mapping)

color_discrete_map = {"㉮": "#EF553B", "㉯": "#00CC96", "㉰": "#636EFA"}

st.markdown("---")

# 4. Visualizations
st.subheader("2. 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_2d_label = st.selectbox(
        "2D X축 속성",
        options=selected_feature_labels,
        index=0,
        key="2d_x",
    )
with col2:
    y_2d_label = st.selectbox(
        "2D Y축 속성",
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

st.subheader("3. 3차원 산점도")
if len(selected_feature_labels) < 3:
    st.info(
        "💡 3차원 산점도를 그리려면 위 군집화 속성 선택에서 속성을 3개 이상 선택해 주세요."
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

# 5. Cluster Analysis & Summary Tables
st.subheader("4. 묶음별 편수 및 속성 평균 (원래 단위)")

summary_df = (
    df.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        _10위권일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_idx", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
)

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
