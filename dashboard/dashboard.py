import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from insight_utils import (
    insight_box,
    insight_categorical_distribution,
    insight_group_comparison,
    insight_distribution,
    insight_boxplot_by_group,
    insight_correlation_driver,
    insight_trend,
    insight_forecast,
    insight_cluster,
    insight_radar,
    insight_scatter_highlight,
    insight_kpi_alert,
    insight_cohort,
)

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="UMKM Investment Dashboard",
    layout="wide"
)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "toImageButtonOptions": {"format":"png","filename":"dashboard_chart","height":1600,"width":2400,"scale":4}
}
# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data(show_spinner=False)
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "dataset_bersih.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    bins = [0,2,4,6,8,10,12,16]
    labels = ["0-2","2-4","4-6","6-8","8-10","10-12","12+"]
    df["tenure_bin"] = pd.cut(
        df["business_tenure_years"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    return df

# ==========================================================
# CACHE MACHINE LEARNING
# ==========================================================

@st.cache_resource(show_spinner=False)
def train_regression(X, y):

    model = LinearRegression()
    model.fit(X, y)

    return model

@st.cache_resource(show_spinner=False)
def train_kmeans(data, k):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)
    return labels

@st.cache_data(show_spinner=False)
def correlation_matrix(df):
    numeric_cols = [
        "digital_adoption_score",
        "repeat_order_rate",
        "kepuasan_pelanggan",
        "review_volatility",
        "monthly_revenue",
        "business_tenure_years",
        "net_profit_margin"
    ]
    return df[numeric_cols].corr()

# ==========================================================
# FUNGSI SAMPLING
# ==========================================================

@st.cache_data(show_spinner=False)
def get_sample(df, n=5000):
    if len(df) <= n:
        return df.copy()
    return df.sample(
        n=n,
        random_state=42
    )

# ==========================================================
# LOAD DATA
# ==========================================================

df_full = load_data()
if df_full is None:
    st.error("dataset_bersih.csv tidak ditemukan.")
    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("Navigasi Dashboard")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview Keseluruhan",
        "Pertanyaan Bisnis",
        "Analisis Naratif Kelas",
        "Tren & Adaptive Intelligence"
    ]
)
st.sidebar.divider()
st.sidebar.subheader("Filter Data")

# ----------------------------------------------------------
# Filter Class
# ----------------------------------------------------------

class_options = sorted(df_full["class"].unique())
selected_classes = st.sidebar.multiselect(
    "Kelas Bisnis",
    class_options,
    default=class_options
)

# ----------------------------------------------------------
# Filter Tenure
# ----------------------------------------------------------

tenure_options = sorted(
    df_full["business_tenure_months"].unique()
)
selected_tenure = st.sidebar.multiselect(
    "Kategori Masa Operasional",
    tenure_options,
    default=tenure_options
)

# ----------------------------------------------------------
# Peak Hour Latency
# ----------------------------------------------------------

latency_options = sorted(
    df_full["peak_hour_latency"].unique()
)
selected_latency = st.sidebar.multiselect(
    "Peak Hour Latency",
    latency_options,
    default=latency_options
)

# ----------------------------------------------------------
# Revenue Slider
# ----------------------------------------------------------

rev_min = float(df_full["monthly_revenue"].min())
rev_max = float(df_full["monthly_revenue"].max())
selected_rev = st.sidebar.slider(
    "Rentang Monthly Revenue (Rp)",
    min_value=rev_min,
    max_value=rev_max,
    value=(rev_min, rev_max)
)

# ==========================================================
# FILTER DATA
# ==========================================================

df = df_full[
    (df_full["class"].isin(selected_classes))
    &
    (df_full["business_tenure_months"].isin(selected_tenure))
    &
    (df_full["peak_hour_latency"].isin(selected_latency))
    &
    (
        df_full["monthly_revenue"].between(
            selected_rev[0],
            selected_rev[1]
        )
    )
]

if df.empty:

    st.warning(
        "Tidak ada data yang sesuai dengan filter."
    )

    st.stop()

st.sidebar.caption(
    f"Menampilkan {len(df):,} dari {len(df_full):,} UMKM "
    f"({len(df)/len(df_full)*100:.1f}%)"
)
# =========================================================
# HALAMAN 1 : OVERVIEW KESELURUHAN
# =========================================================

if page == "Overview Keseluruhan":
    st.title("📊 Representasi Keseluruhan Data UMKM")
    st.write(
        "Gambaran umum performa UMKM berdasarkan filter yang dipilih."
    )

    # =====================================================
    # KPI
    # =====================================================

    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric(
        "Total UMKM",
        f"{len(df):,}",
        delta=f"{len(df)-len(df_full):,}"
    )
    col2.metric(
        "Rata-rata Revenue",
        f"Rp{df['monthly_revenue'].mean():,.0f}",
        delta=f"{df['monthly_revenue'].mean()-df_full['monthly_revenue'].mean():,.0f}"
    )
    col3.metric(
        "Profit Margin",
        f"{df['net_profit_margin'].mean():.2f}%",
        delta=f"{df['net_profit_margin'].mean()-df_full['net_profit_margin'].mean():.2f}%"
    )
    col4.metric(
        "Digital Adoption",
        f"{df['digital_adoption_score'].mean():.2f}/10",
        delta=f"{df['digital_adoption_score'].mean()-df_full['digital_adoption_score'].mean():.2f}"
    )
    col5.metric(
        "Repeat Order",
        f"{df['repeat_order_rate'].mean():.2f}%",
        delta=f"{df['repeat_order_rate'].mean()-df_full['repeat_order_rate'].mean():.2f}%"
    )
    st.divider()

    # =====================================================
    # PIE DAN BAR
    # =====================================================

    col_a,col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Kelas Bisnis")
        pie = px.pie(
            df,
            names="class",
            hole=0.45
        )
        st.plotly_chart(
            pie,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )
        insight_box(
            insight_categorical_distribution(df["class"], label="UMKM")
        )
    with col_b:
        st.subheader("Repeat Order Rate berdasarkan Peak Hour Latency")
        latency = (
            df.groupby("peak_hour_latency")["repeat_order_rate"]
            .mean()
            .reindex(["Low","Med","High"])
            .reset_index()
        )
        bar = px.bar(
            latency,
            x="peak_hour_latency",
            y="repeat_order_rate",
            labels={
                "peak_hour_latency":"Latency",
                "repeat_order_rate":"Repeat Order Rate (%)"
            }
        )
        st.plotly_chart(
            bar,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )
        insight_box(
            insight_group_comparison(
                df, "peak_hour_latency", "repeat_order_rate",
                unit="%", label="repeat order rate"
            )
        )
    st.divider()

    # =====================================================
    # HISTOGRAM DAN BOXPLOT
    # =====================================================

    col1,col2 = st.columns(2)
    with col1:
        hist = px.histogram(
            df,
            x="monthly_revenue",
            nbins=30,
            title="Distribusi Monthly Revenue"
        )
        st.plotly_chart(
            hist,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )
        insight_box(
            insight_distribution(df["monthly_revenue"], "Monthly Revenue", unit="")
        )
    with col2:
        box = px.box(
            df,
            x="class",
            y="net_profit_margin",
            color="class",
            title="Sebaran Net Profit Margin"
        )
        st.plotly_chart(
            box,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )
        insight_box(
            insight_boxplot_by_group(df, "class", "net_profit_margin", unit="%")
        )
    st.divider()

    # =====================================================
    # RINGKASAN
    # =====================================================

    top_class = (
        df.groupby("class")["net_profit_margin"]
        .mean()
        .idxmax()
    )
    top_margin = (
        df.groupby("class")["net_profit_margin"]
        .mean()
        .max()
    )
    dominant = df["class"].value_counts().idxmax()
    dominant_share = (
        df["class"]
        .value_counts(normalize=True)
        .max()*100
    )
    negative = (
        (df["net_profit_margin"]<0)
        .mean()*100
    )
    st.subheader("Ringkasan Performa")
    st.markdown(f"""

- {top_class} memiliki rata-rata margin tertinggi ({top_margin:.2f}%).
- Kelas yang paling dominan adalah {dominant} ({dominant_share:.1f}%).
- {negative:.1f}% UMKM masih mengalami margin negatif.
- Digital Adoption rata-rata {df['digital_adoption_score'].mean():.2f}/10
- Customer Satisfaction rata-rata {df['kepuasan_pelanggan'].mean():.2f}/5
""")
    st.divider()
    st.subheader("Sampel Dataset")
    st.dataframe(
        df.head(10),
        use_container_width=True
    )
# =========================================================
# HALAMAN 2 : PERTANYAAN BISNIS
# =========================================================

elif page == "Pertanyaan Bisnis":
    st.title("💡 Jawaban Pertanyaan Bisnis")

    # =====================================================
    # TOP GROWTH
    # =====================================================

    st.subheader("1. UMKM Top Growth untuk Investasi")
    avg_rev = df["monthly_revenue"].mean()
    min_margin = 15
    top_growth = df[
        (df["class"]=="Growth")
        &
        (df["net_profit_margin"]>min_margin)
        &
        (df["monthly_revenue"]>avg_rev)
    ]
    st.info(
        f"Rata-rata Monthly Revenue : Rp{avg_rev:,.0f}"
    )
    st.success(
        f"Jumlah UMKM Top Growth : {len(top_growth):,}"
    )

    # ===========================================
    # Mengambil sample agar scatter tidak berat
    # ===========================================

    sample_df = get_sample(df)
    col1,col2 = st.columns([2,1])
    with col1:
        fig_scatter = px.scatter(
            sample_df,
            x="monthly_revenue",
            y="net_profit_margin",
            color="class",
            opacity=0.55,
            title="Distribusi Revenue vs Margin"
        )
        # tetap tampilkan seluruh Top Growth
        fig_scatter.add_trace(
            go.Scatter(
                x=top_growth["monthly_revenue"],
                y=top_growth["net_profit_margin"],
                mode="markers",
                marker=dict(
                    color="red",
                    size=8,
                    symbol="star"
                ),
                name="Top Growth"
            )
        )
        fig_scatter.add_vline(
            x=avg_rev,
            line_dash="dash",
            line_color="gray"
        )
        fig_scatter.add_hline(
            y=min_margin,
            line_dash="dash",
            line_color="orange"
        )
        st.plotly_chart(
            fig_scatter,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )
        insight_box(
            insight_scatter_highlight(
                df, top_growth, "monthly_revenue", "net_profit_margin",
                highlight_label="UMKM Top Growth"
            )
        )
    with col2:
        pie_data = pd.DataFrame({
            "Kategori":[
                "Top Growth",
                "Lainnya"
            ],
            "Jumlah":[
                len(top_growth),
                len(df)-len(top_growth)
            ]
        })
        fig_pie = px.pie(
            pie_data,
            values="Jumlah",
            names="Kategori",
            hole=.45,
            color_discrete_sequence=[
                "#EF4444",
                "#3B82F6"
            ]
        )
        fig_pie.update_traces(
            textinfo="percent+label"
        )
        st.plotly_chart(
            fig_pie,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )
        _total_pie = pie_data["Jumlah"].sum()
        _top_growth_pct = (len(top_growth) / _total_pie * 100) if _total_pie else 0
        insight_box(
            f"**{_top_growth_pct:.1f}%** dari total UMKM terfilter masuk kategori **Top Growth** "
            f"(margin > {min_margin}% dan revenue di atas rata-rata), sisanya **{100 - _top_growth_pct:.1f}%** "
            f"belum memenuhi kriteria investasi ini."
        )

    # =====================================================
    # DATA TOP GROWTH
    # =====================================================

    with st.expander("Lihat Daftar UMKM Top Growth"):
        st.dataframe(
            top_growth.sort_values(
                "net_profit_margin",
                ascending=False
            ),
            use_container_width=True
        )
    st.divider()

    # =====================================================
    # TENURE VS LOYALTY
    # =====================================================

    st.subheader(
        "2. Dampak Masa Operasional terhadap Loyalitas"
    )
    tenure_compare = (
        df.groupby(
            "tenure_bin",
            observed=True
        )["repeat_order_rate"]
        .mean()
        .reset_index()
    )
    fig,ax = plt.subplots(
        figsize=(10,5)
    )
    sns.barplot(
        data=tenure_compare,
        x="tenure_bin",
        y="repeat_order_rate",
        hue="tenure_bin",
        palette="viridis",
        legend=False,
        ax=ax
    )
    ax.set_xlabel(
        "Masa Operasional (Tahun)"
    )
    ax.set_ylabel(
        "Repeat Order Rate (%)"
    )
    ax.set_title(
        "Loyalitas Berdasarkan Masa Operasional"
    )
    st.pyplot(fig)
    insight_box(
        insight_group_comparison(
            df.assign(tenure_bin=df["tenure_bin"].astype(str)),
            "tenure_bin", "repeat_order_rate",
            unit="%", label="repeat order rate"
        )
    )
# =========================================================
# HALAMAN 3 : ANALISIS NARATIF KELAS
# =========================================================

elif page == "Analisis Naratif Kelas":
    st.title("📝 Analisis Naratif Antar Kelas")
    metrics = [
        "net_profit_margin",
        "digital_adoption_score",
        "repeat_order_rate",
        "kepuasan_pelanggan"
    ]
    class_stats = (
        df
        .groupby("class")[metrics]
        .mean()
    )
    missing = [
        c
        for c in class_options
        if c not in class_stats.index
    ]
    if missing:
        st.info(
            "Kelas yang tidak muncul pada data terfilter : "
            + ", ".join(missing)
        )

    # ==========================================
    # NORMALISASI
    # ==========================================

    df_norm = (
        class_stats -
        class_stats.min()
    ) / (
        class_stats.max()
        -
        class_stats.min()
    )
    df_norm = df_norm.fillna(0)

    # ==========================================
    # RADAR
    # ==========================================

    def radar_chart(target_class,color):
        category = [
            "Margin",
            "Digital",
            "Repeat Order",
            "Kepuasan"
        ]
        value = df_norm.loc[target_class].tolist()
        value += value[:1]
        category += category[:1]
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=value,
                theta=category,
                fill="toself",
                line_color=color,
                name=target_class
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0,1]
                )
            ),
            showlegend=False,
            margin=dict(
                l=30,
                r=30,
                t=40,
                b=30
            )
        )
        return fig

    # ==========================================
    # TABS
    # ==========================================

    info = [
        ("Elite","#FFD700","🏆"),
        ("Growth","#00C853","📈"),
        ("Struggling","#FB8C00","⚠"),
        ("Critical","#E53935","🚨")
    ]
    info = [
        x
        for x in info
        if x[0] in class_stats.index
    ]
    tabs = st.tabs(
        [i[0] for i in info]
    )
    for (
        name,
        color,
        icon
    ),tab in zip(
        info,
        tabs
    ):
        with tab:
            c1,c2 = st.columns(
                [1,1]
            )
            with c1:
                st.markdown(
                    f"## {icon} {name}"
                )
                st.metric(
                    "Margin",
                    f"{class_stats.loc[name,'net_profit_margin']:.2f}%"
                )
                st.metric(
                    "Digital",
                    f"{class_stats.loc[name,'digital_adoption_score']:.2f}/10"
                )
                st.metric(
                    "Repeat Order",
                    f"{class_stats.loc[name,'repeat_order_rate']:.2f}%"
                )
                st.metric(
                    "Kepuasan",
                    f"{class_stats.loc[name,'kepuasan_pelanggan']:.2f}/5"
                )
            with c2:
                st.plotly_chart(
                    radar_chart(
                        name,
                        color
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG
                )
                insight_box(
                    insight_radar(df_norm, name)
                )
            st.markdown(
                "### 🔍 Drill Down"
            )
            sort_by = st.selectbox(
                f"Urutkan Data {name}",
                [
                    "net_profit_margin",
                    "monthly_revenue",
                    "repeat_order_rate",
                    "digital_adoption_score"
                ],
                key=f"sort_{name}"
            )
            rows = st.slider(
                f"Jumlah Data {name}",
                5,
                50,
                10,
                key=f"rows_{name}"
            )
            result = (
                df[
                    df["class"]==name
                ]
                .sort_values(
                    sort_by,
                    ascending=False
                )
                .head(rows)
            )
            st.dataframe(
                result,
                use_container_width=True
            )
# =========================================================
# HALAMAN 4 : TREN & ADAPTIVE INTELLIGENCE
# =========================================================
elif page == "Tren & Adaptive Intelligence":
    st.title("🧠 Tren & Adaptive Intelligence")
    st.caption(
        "Analisis tren menggunakan business tenure sebagai proksi waktu."
    )
    # =====================================================
    # 1. TREN BERDASARKAN TENURE
    # =====================================================
    st.subheader("1. Tren Berdasarkan Masa Operasional")
    trend_data = (
        df.groupby(
            "tenure_bin",
            observed=True
        )
        .agg(
            avg_revenue=("monthly_revenue","mean"),
            avg_margin=("net_profit_margin","mean"),
            avg_digital=("digital_adoption_score","mean")
        )
        .reset_index()
    )
    order = [
        "0-2",
        "2-4",
        "4-6",
        "6-8",
        "8-10",
        "10-12",
        "12+"
    ]
    trend_data["tenure_bin"] = trend_data["tenure_bin"].astype(str)
    trend_data["tenure_bin"] = pd.Categorical(
        trend_data["tenure_bin"],
        categories=order,
        ordered=True
    )
    trend_data = trend_data.sort_values("tenure_bin")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_data["tenure_bin"],
            y=trend_data["avg_revenue"],
            mode="lines+markers",
            name="Average Revenue"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_data["tenure_bin"],
            y=trend_data["avg_margin"],
            mode="lines+markers",
            yaxis="y2",
            name="Average Margin"
        )
    )
    fig.update_layout(
        height=650,
        xaxis=dict(type="category"),
        yaxis=dict(
            title="Revenue"
        ),
        yaxis2=dict(
            title="Margin (%)",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h"
        )
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )
    insight_box(
        insight_trend(trend_data["avg_revenue"].values, "Average Revenue", unit="")
        + " " +
        insight_trend(trend_data["avg_margin"].values, "Average Margin", unit="%")
    )
    st.divider()
    # =====================================================
    # 2. TREND DETECTION
    # =====================================================
    st.subheader("2. Deteksi Perubahan Tren")
    valid = trend_data.dropna()
    if len(valid)>=2:
        x = np.arange(len(valid))
        slope_margin = np.polyfit(
            x,
            valid["avg_margin"],
            1
        )[0]
        slope_rev = np.polyfit(
            x,
            valid["avg_revenue"],
            1
        )[0]
        c1,c2 = st.columns(2)
        with c1:
            arah = (
                "📈 Naik"
                if slope_margin>0.05
                else "📉 Turun"
                if slope_margin<-0.05
                else "➡ Stabil"
            )
            st.metric(
                "Trend Margin",
                arah,
                f"{slope_margin:.3f}"
            )
        with c2:
            arah2 = (
                "📈 Naik"
                if slope_rev>0
                else "📉 Turun"
                if slope_rev<0
                else "➡ Stabil"
            )
            st.metric(
                "Trend Revenue",
                arah2,
                f"Rp{slope_rev:,.0f}"
            )
        insight_box(
            f"Margin bergerak **{arah.split(' ')[-1].lower()}** (slope **{slope_margin:.3f}**) dan revenue "
            f"bergerak **{arah2.split(' ')[-1].lower()}** (slope **Rp{slope_rev:,.0f}** per kategori tenure). "
            f"Jika kedua tren searah, ini mengindikasikan hubungan yang konsisten antara masa operasional, "
            f"revenue, dan profitabilitas."
        )
    st.divider()

    # =====================================================
    # 3 KPI ALERT
    # =====================================================

    st.subheader("3. KPI Alert")
    c1,c2 = st.columns(2)
    margin_limit = c1.slider(
        "Minimum Margin",
        -20,
        30,
        0
    )
    digital_limit = c2.slider(
        "Minimum Digital Score",
        0.0,
        10.0,
        3.0
    )
    margin_pct = (
        df["net_profit_margin"]<margin_limit
    ).mean()*100
    digital_pct = (
        df["digital_adoption_score"]<digital_limit
    ).mean()*100
    st.metric(
        "Margin di bawah batas",
        f"{margin_pct:.1f}%"
    )
    st.metric(
        "Digital di bawah batas",
        f"{digital_pct:.1f}%"
    )
    insight_box(
        insight_kpi_alert(margin_pct, digital_pct, margin_limit, digital_limit)
    )
    st.divider()

    # =====================================================
    # 4 KOHORT
    # =====================================================

    st.subheader("4. Analisis Kohort")
    cohort = (
        df.groupby(
            "business_tenure_months"
        )
        .agg(
            Revenue=("monthly_revenue","mean"),
            Margin=("net_profit_margin","mean"),
            Repeat=("repeat_order_rate","mean"),
            Jumlah=("class","count")
        )
        .reset_index()
    )
    st.dataframe(
        cohort,
        use_container_width=True
    )
    insight_box(
        insight_cohort(cohort, value_col="Margin")
    )
    st.divider()

    # =====================================================
    # 5 KORELASI
    # =====================================================
    st.subheader(
        "5. Faktor yang Mempengaruhi Margin"
    )
    corr = correlation_matrix(df)
    corr = corr["net_profit_margin"]
    corr = corr.drop(
        "net_profit_margin"
    )
    corr = corr.sort_values(
        key=abs,
        ascending=False
    )
    fig = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        color=corr.values,
        color_continuous_scale="RdBu",
        range_color=[-1,1]
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )
    insight_box(
        insight_correlation_driver(corr, target_label="Net Profit Margin")
    )
    st.divider()

    # =====================================================
    # 6 FORECAST
    # =====================================================

    st.subheader(
        "6. Forecast Revenue"
    )
    X = df[
        ["business_tenure_years"]
    ].values
    y = df[
        "monthly_revenue"
    ].values
    model = train_regression(
        X,
        y
    )
    future = np.linspace(
        0,
        df["business_tenure_years"].max()+3,
        50
    ).reshape(-1,1)
    pred = model.predict(future)
    sample = get_sample(df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample["business_tenure_years"],
            y=sample["monthly_revenue"],
            mode="markers",
            marker=dict(
                size=4,
                opacity=.35
            ),
            name="Data"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future.flatten(),
            y=pred,
            mode="lines",
            line=dict(
                color="red",
                width=3
            ),
            name="Forecast"
        )
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )
    insight_box(
        insight_forecast(future, pred, unit="Rp")
    )
    st.divider()

    # =====================================================
    # 7 CLUSTERING
    # =====================================================
    st.subheader(
        "7. Segmentasi UMKM"
    )
    k = st.slider(
        "Jumlah Cluster",
        2,
        6,
        4
    )
    features = [
        "net_profit_margin",
        "digital_adoption_score",
        "repeat_order_rate",
        "kepuasan_pelanggan"
    ]
    label = train_kmeans(
        df[features],
        k
    )
    cluster = df.copy()
    cluster["segment"] = label.astype(str)
    sample_cluster = get_sample(cluster)
    c1,c2 = st.columns([2,1])
    with c1:
        fig = px.scatter(
            sample_cluster,
            x="monthly_revenue",
            y="net_profit_margin",
            color="segment",
            hover_data=["class"]
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )
        insight_box(
            insight_cluster(cluster, "segment", "net_profit_margin", unit="%")
        )
    with c2:
        cross = pd.crosstab(
            cluster["segment"],
            cluster["class"]
        )
        st.dataframe(
            cross,
            use_container_width=True
        )
    st.success(
        "Analisis selesai."
    )
