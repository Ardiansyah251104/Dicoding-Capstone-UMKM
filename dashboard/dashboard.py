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

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="UMKM Investment Dashboard", layout="wide")

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'dataset_bersih.csv')

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Bin tenure (tahun) untuk keperluan tren & forecast, dipakai di banyak halaman
        bins = [0, 2, 4, 6, 8, 10, 12, 16]
        labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12', '12+']
        df['tenure_bin'] = pd.cut(df['business_tenure_years'], bins=bins, labels=labels, include_lowest=True)
        return df
    else:
        st.error(f"File tidak ditemukan di: {file_path}. Pastikan file 'dataset_bersih.csv' ada di folder dashboard.")
        return None

df_full = load_data()

if df_full is not None:

    # =========================================================
    # SIDEBAR: NAVIGASI + FILTER INTERAKTIF (berlaku global)
    # =========================================================
    st.sidebar.title("Navigasi Dashboard")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["Overview Keseluruhan", "Pertanyaan Bisnis", "Analisis Naratif Kelas", "Tren & Adaptive Intelligence"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Filter Data")

    class_options = sorted(df_full['class'].unique().tolist())
    selected_classes = st.sidebar.multiselect("Kelas Bisnis", class_options, default=class_options)

    tenure_options = sorted(df_full['business_tenure_months'].unique().tolist())
    selected_tenure = st.sidebar.multiselect("Kategori Masa Operasional", tenure_options, default=tenure_options)

    latency_options = sorted(df_full['peak_hour_latency'].unique().tolist())
    selected_latency = st.sidebar.multiselect("Peak Hour Latency", latency_options, default=latency_options)

    rev_min, rev_max = float(df_full['monthly_revenue'].min()), float(df_full['monthly_revenue'].max())
    selected_rev = st.sidebar.slider(
        "Rentang Monthly Revenue (Rp)",
        min_value=rev_min, max_value=rev_max, value=(rev_min, rev_max)
    )

    # Terapkan filter
    df = df_full[
        (df_full['class'].isin(selected_classes)) &
        (df_full['business_tenure_months'].isin(selected_tenure)) &
        (df_full['peak_hour_latency'].isin(selected_latency)) &
        (df_full['monthly_revenue'].between(selected_rev[0], selected_rev[1]))
    ]

    if df.empty:
        st.warning("Tidak ada data yang cocok dengan kombinasi filter ini. Coba longgarkan filternya.")
        st.stop()

    st.sidebar.caption(f"Menampilkan {len(df):,} dari {len(df_full):,} UMKM ({len(df)/len(df_full)*100:.1f}%)")

    # =========================================================
    # HALAMAN 1: OVERVIEW KESELURUHAN
    # =========================================================
    if page == "Overview Keseluruhan":
        st.title("📊 Representasi Keseluruhan Data UMKM")
        st.write("Gambaran umum performa UMKM berdasarkan filter yang dipilih di sidebar.")

        # --- KPI Cards dengan delta terhadap baseline (data tanpa filter) ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total UMKM", f"{len(df):,}", delta=f"{len(df) - len(df_full):,} vs total")
        col2.metric(
            "Rata-rata Revenue",
            f"Rp{df['monthly_revenue'].mean():,.0f}",
            delta=f"{df['monthly_revenue'].mean() - df_full['monthly_revenue'].mean():,.0f}"
        )
        col3.metric(
            "Rata-rata Profit Margin",
            f"{df['net_profit_margin'].mean():.2f}%",
            delta=f"{df['net_profit_margin'].mean() - df_full['net_profit_margin'].mean():.2f}%"
        )
        col4.metric(
            "Adopsi Digital",
            f"{df['digital_adoption_score'].mean():.2f}/10",
            delta=f"{df['digital_adoption_score'].mean() - df_full['digital_adoption_score'].mean():.2f}"
        )
        col5.metric(
            "Repeat Order Rate",
            f"{df['repeat_order_rate'].mean():.2f}%",
            delta=f"{df['repeat_order_rate'].mean() - df_full['repeat_order_rate'].mean():.2f}%"
        )

        st.divider()

        # --- Distribusi Kelas & Perbandingan Kategori ---
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribusi Kelas Bisnis")
            fig_pie = px.pie(df, names='class', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.subheader("Repeat Order Rate berdasarkan Peak Hour Latency")
            latency_comp = df.groupby('peak_hour_latency')['repeat_order_rate'].mean().reindex(['Low', 'Med', 'High']).reset_index()
            fig_lat = px.bar(latency_comp, x='peak_hour_latency', y='repeat_order_rate',
                              labels={'peak_hour_latency': 'Latency', 'repeat_order_rate': 'Repeat Order Rate (%)'})
            st.plotly_chart(fig_lat, use_container_width=True)

        # --- Visualisasi Distribusi ---
        st.subheader("Visualisasi Distribusi")
        col_c, col_d = st.columns(2)
        with col_c:
            fig_hist = px.histogram(df, x='monthly_revenue', nbins=40, title="Distribusi Monthly Revenue")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_d:
            fig_box = px.box(df, x='class', y='net_profit_margin', color='class', title="Sebaran Net Profit Margin per Kelas")
            st.plotly_chart(fig_box, use_container_width=True)

        # --- Ringkasan Performa (otomatis, berbasis angka) ---
        st.subheader("Ringkasan Performa")
        top_class = df.groupby('class')['net_profit_margin'].mean().idxmax()
        top_class_margin = df.groupby('class')['net_profit_margin'].mean().max()
        dominant_class = df['class'].value_counts().idxmax()
        dominant_share = df['class'].value_counts(normalize=True).max() * 100
        low_margin_share = (df['net_profit_margin'] < 0).mean() * 100

        st.markdown(f"""
- Kelas dengan rata-rata margin tertinggi pada data terfilter adalah **{top_class}** ({top_class_margin:.2f}%).
- Kelas paling dominan dari sisi jumlah adalah **{dominant_class}**, mencakup **{dominant_share:.1f}%** dari total UMKM yang ditampilkan.
- **{low_margin_share:.1f}%** UMKM pada data ini masih mencatat margin negatif (rugi bersih bulanan).
- Rata-rata skor adopsi digital berada di **{df['digital_adoption_score'].mean():.2f} dari 10**, dengan rata-rata kepuasan pelanggan **{df['kepuasan_pelanggan'].mean():.2f} dari 5**.
        """)

        st.subheader("Sampel Data")
        st.dataframe(df.head(10))

    # =========================================================
    # HALAMAN 2: PERTANYAAN BISNIS
    # =========================================================
    elif page == "Pertanyaan Bisnis":
        st.title("💡 Jawaban Pertanyaan Bisnis")

        # 1. Pertanyaan 1 (Top Growth)
        st.subheader("Pertanyaan 1: UMKM 'Top Growth' untuk Investasi")

        avg_rev = df['monthly_revenue'].mean()
        min_margin = 15

        top_growth = df[(df['class'] == 'Growth') & (df['net_profit_margin'] > min_margin) & (df['monthly_revenue'] > avg_rev)]

        st.info(f"Rata-rata Monthly Revenue (data terfilter): Rp{avg_rev:,.2f}")
        st.success(f"Jumlah UMKM Top Growth (Target Investor): **{len(top_growth)}** UMKM")

        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            fig_scatter = px.scatter(
                df,
                x='monthly_revenue',
                y='net_profit_margin',
                color='class',
                title="Distribusi Revenue vs Margin UMKM",
                labels={'monthly_revenue': 'Monthly Revenue (Rp)', 'net_profit_margin': 'Net Profit Margin (%)'}
            )

            fig_scatter.add_trace(go.Scatter(
                x=top_growth['monthly_revenue'],
                y=top_growth['net_profit_margin'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='star'),
                name='Top Growth (Target)'
            ))

            fig_scatter.add_vline(x=avg_rev, line_dash="dash", line_color="gray", annotation_text="Avg Revenue")
            fig_scatter.add_hline(y=min_margin, line_dash="dash", line_color="orange", annotation_text="Min Margin: 15%")

            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_chart2:
            total_umkm = len(df)
            proporsi_data = pd.DataFrame({
                'Kategori': ['Top Growth', 'Others'],
                'Jumlah': [len(top_growth), total_umkm - len(top_growth)]
            })

            fig_pie_target = px.pie(
                proporsi_data,
                values='Jumlah',
                names='Kategori',
                title="Proporsi Target Investor",
                color_discrete_sequence=['red', '#4285F4'],
                hole=0.3
            )
            fig_pie_target.update_traces(pull=[0.2, 0], textinfo='percent+label')
            st.plotly_chart(fig_pie_target, use_container_width=True)

        with st.expander("Lihat daftar UMKM Top Growth"):
            st.dataframe(top_growth.sort_values('net_profit_margin', ascending=False))

        st.divider()

        # 2. Pertanyaan 2 (Tenure vs Loyalty) - pakai bin tahun, lebih granular dari sekadar 2 kategori
        st.subheader("Pertanyaan 2: Dampak Masa Operasional terhadap Loyalitas")
        tenure_comparison = df.groupby('tenure_bin', observed=True)['repeat_order_rate'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x='tenure_bin',
            y='repeat_order_rate',
            data=tenure_comparison,
            hue='tenure_bin',
            palette='viridis',
            legend=False,
            ax=ax
        )
        ax.set_title('Loyalitas Berdasarkan Masa Operasional (tahun)')
        ax.set_xlabel('Masa Operasional (tahun)')
        ax.set_ylabel('Repeat Order Rate (%)')
        st.pyplot(fig)

    # =========================================================
    # HALAMAN 3: ANALISIS NARATIF KELAS (+ Drill-down)
    # =========================================================
    elif page == "Analisis Naratif Kelas":
        st.title("📝 Perbandingan Karakteristik Antar Kelas")

        metrics = ['net_profit_margin', 'digital_adoption_score', 'repeat_order_rate', 'kepuasan_pelanggan']
        class_stats = df.groupby('class')[metrics].mean()

        missing_classes = [c for c in class_options if c not in class_stats.index]
        if missing_classes:
            st.info(f"Kelas berikut tidak muncul pada data terfilter: {', '.join(missing_classes)}")

        df_norm = (class_stats - class_stats.min()) / (class_stats.max() - class_stats.min())
        df_norm = df_norm.fillna(0)

        def create_radar_chart(target_class, color):
            categories = ['Margin', 'Digital', 'Repeat Order', 'Kepuasan']
            values = df_norm.loc[target_class].values.tolist()
            values += values[:1]
            categories += categories[:1]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values, theta=categories, fill='toself',
                name=target_class, line_color=color
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False, title=f"DNA {target_class}"
            )
            return fig

        st.subheader("Analisis Profil Bisnis")
        class_info_all = [
            ("Elite", "#FFD700", "🏆"),
            ("Growth", "#00FF00", "📈"),
            ("Struggling", "#FFA500", "⚠️"),
            ("Critical", "#FF0000", "🚨")
        ]
        class_info = [c for c in class_info_all if c[0] in class_stats.index]
        tabs = st.tabs([c[0] for c in class_info])

        for (name, color, icon), tab in zip(class_info, tabs):
            with tab:
                col_text, col_plot = st.columns(2)
                with col_text:
                    st.markdown(f"### {icon} Kelas {name}")
                    st.write(f"**Margin**: {class_stats.loc[name, 'net_profit_margin']:.2f}%")
                    st.write(f"**Digital**: {class_stats.loc[name, 'digital_adoption_score']:.2f}/10")
                    st.write(f"**Loyalty**: {class_stats.loc[name, 'repeat_order_rate']:.2f}%")
                    st.write(f"**Kepuasan**: {class_stats.loc[name, 'kepuasan_pelanggan']:.2f}/5")
                with col_plot:
                    st.plotly_chart(create_radar_chart(name, color), use_container_width=True)

                # --- Drill-down: lihat data mentah kelas ini ---
                st.markdown("#### 🔍 Drill-down Data")
                sort_by = st.selectbox(
                    f"Urutkan berdasarkan ({name})",
                    ['net_profit_margin', 'monthly_revenue', 'repeat_order_rate', 'digital_adoption_score'],
                    key=f"sort_{name}"
                )
                n_rows = st.slider(f"Jumlah baris ditampilkan ({name})", 5, 50, 10, key=f"nrows_{name}")
                class_data = df[df['class'] == name].sort_values(sort_by, ascending=False).head(n_rows)
                st.dataframe(class_data)

    # =========================================================
    # HALAMAN 4: TREN & ADAPTIVE INTELLIGENCE
    # =========================================================
    elif page == "Tren & Adaptive Intelligence":
        st.title("🧠 Tren & Adaptive Intelligence")
        st.caption(
            "Dataset ini tidak memiliki kolom tanggal/waktu. Sebagai proksi tren, "
            "halaman ini menggunakan masa operasional (business tenure) sebagai sumbu kohort, "
            "bukan deret waktu historis yang sebenarnya."
        )

        # --- A. Tren berdasarkan masa operasional ---
        st.subheader("1. Tren Berdasarkan Masa Operasional")
        trend_data = df.groupby('tenure_bin', observed=True).agg(
            avg_revenue=('monthly_revenue', 'mean'),
            avg_margin=('net_profit_margin', 'mean'),
            avg_digital=('digital_adoption_score', 'mean')
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend_data['tenure_bin'], y=trend_data['avg_revenue'],
                                        mode='lines+markers', name='Avg Revenue', yaxis='y1'))
        fig_trend.add_trace(go.Scatter(x=trend_data['tenure_bin'], y=trend_data['avg_margin'],
                                        mode='lines+markers', name='Avg Margin (%)', yaxis='y2'))
        fig_trend.update_layout(
            xaxis_title="Masa Operasional (tahun)",
            yaxis=dict(title="Avg Revenue (Rp)"),
            yaxis2=dict(title="Avg Margin (%)", overlaying='y', side='right'),
            legend=dict(orientation='h')
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.divider()

        # --- B. Deteksi perubahan tren otomatis ---
        st.subheader("2. Deteksi Perubahan Tren Otomatis")
        trend_valid = trend_data.dropna(subset=['avg_margin'])
        if len(trend_valid) >= 2:
            x_idx = np.arange(len(trend_valid))
            slope_margin = np.polyfit(x_idx, trend_valid['avg_margin'], 1)[0]
            slope_revenue = np.polyfit(x_idx, trend_valid['avg_revenue'], 1)[0]

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                arah = "naik 📈" if slope_margin > 0.05 else ("turun 📉" if slope_margin < -0.05 else "relatif stabil ➡️")
                st.metric("Arah Tren Margin (per bin tenure)", arah, delta=f"{slope_margin:.3f} poin/bin")
            with col_t2:
                arah_rev = "naik 📈" if slope_revenue > 0 else ("turun 📉" if slope_revenue < 0 else "relatif stabil ➡️")
                st.metric("Arah Tren Revenue (per bin tenure)", arah_rev, delta=f"Rp{slope_revenue:,.0f}/bin")
        else:
            st.info("Data terfilter kurang bervariasi untuk mendeteksi tren (butuh minimal 2 kelompok tenure).")

        st.divider()

        # --- C. Alert ambang batas KPI ---
        st.subheader("3. Peringatan Ambang Batas KPI")
        col_th1, col_th2 = st.columns(2)
        margin_threshold = col_th1.slider("Ambang batas margin minimum yang diinginkan (%)", -20, 30, 0)
        digital_threshold = col_th2.slider("Ambang batas skor adopsi digital minimum", 0.0, 10.0, 3.0)

        pct_below_margin = (df['net_profit_margin'] < margin_threshold).mean() * 100
        pct_below_digital = (df['digital_adoption_score'] < digital_threshold).mean() * 100

        if pct_below_margin > 30:
            st.error(f"⚠️ {pct_below_margin:.1f}% UMKM berada di bawah ambang margin {margin_threshold}%. Perlu perhatian.")
        elif pct_below_margin > 10:
            st.warning(f"{pct_below_margin:.1f}% UMKM berada di bawah ambang margin {margin_threshold}%.")
        else:
            st.success(f"Hanya {pct_below_margin:.1f}% UMKM di bawah ambang margin {margin_threshold}%. Kondisi aman.")

        if pct_below_digital > 30:
            st.error(f"⚠️ {pct_below_digital:.1f}% UMKM di bawah ambang adopsi digital {digital_threshold}. Perlu program digitalisasi.")
        elif pct_below_digital > 10:
            st.warning(f"{pct_below_digital:.1f}% UMKM di bawah ambang adopsi digital {digital_threshold}.")
        else:
            st.success(f"Hanya {pct_below_digital:.1f}% UMKM di bawah ambang adopsi digital {digital_threshold}.")

        st.divider()

        # --- D. Perbandingan performa antar periode (kohort tenure) ---
        st.subheader("4. Perbandingan Performa Antar Periode (Kohort)")
        cohort = df.groupby('business_tenure_months', observed=True).agg(
            avg_revenue=('monthly_revenue', 'mean'),
            avg_margin=('net_profit_margin', 'mean'),
            avg_repeat=('repeat_order_rate', 'mean'),
            jumlah=('class', 'count')
        ).reset_index()
        st.dataframe(cohort.style.format({'avg_revenue': 'Rp{:,.0f}', 'avg_margin': '{:.2f}%', 'avg_repeat': '{:.2f}%'}))

        if len(cohort) == 2:
            diff_margin = cohort['avg_margin'].iloc[1] - cohort['avg_margin'].iloc[0]
            st.write(f"Selisih rata-rata margin antar kedua kategori masa operasional: **{diff_margin:.2f} poin persentase**.")

        st.divider()

        # --- E. Analisis kontribusi faktor terhadap margin ---
        st.subheader("5. Kontribusi Faktor terhadap Net Profit Margin")
        numeric_cols = ['digital_adoption_score', 'repeat_order_rate', 'kepuasan_pelanggan',
                         'review_volatility', 'monthly_revenue', 'business_tenure_years']
        corr = df[numeric_cols + ['net_profit_margin']].corr()['net_profit_margin'].drop('net_profit_margin')
        corr = corr.sort_values(key=abs, ascending=False)

        fig_corr = px.bar(
            x=corr.values, y=corr.index, orientation='h',
            labels={'x': 'Korelasi terhadap Net Profit Margin', 'y': 'Faktor'},
            color=corr.values, color_continuous_scale='RdBu', range_color=[-1, 1]
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Korelasi bukan sebab-akibat. Angka ini menunjukkan kekuatan hubungan linear, bukan bukti kausalitas.")

        st.divider()

        # --- F. Prediksi sederhana (forecast ilustratif) ---
        st.subheader("6. Proyeksi Sederhana Revenue vs Masa Operasional")
        st.caption("Proyeksi ini bersifat ilustratif, dibangun dari pola cross-sectional data saat ini, bukan forecast time-series riil.")

        X = df[['business_tenure_years']].values
        y = df['monthly_revenue'].values
        model = LinearRegression().fit(X, y)

        max_tenure = df['business_tenure_years'].max()
        future_range = np.linspace(0, max_tenure + 3, 50).reshape(-1, 1)
        pred = model.predict(future_range)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df['business_tenure_years'], y=df['monthly_revenue'],
                                           mode='markers', marker=dict(opacity=0.15), name='Data Aktual'))
        fig_forecast.add_trace(go.Scatter(x=future_range.flatten(), y=pred, mode='lines',
                                           line=dict(color='red', width=3), name='Garis Proyeksi'))
        fig_forecast.add_vline(x=max_tenure, line_dash='dash', line_color='gray', annotation_text='Batas data saat ini')
        fig_forecast.update_layout(xaxis_title='Masa Operasional (tahun)', yaxis_title='Monthly Revenue (Rp)')
        st.plotly_chart(fig_forecast, use_container_width=True)

        proj_value = model.predict([[max_tenure + 2]])[0]
        st.write(f"Jika pola saat ini berlanjut, rata-rata revenue pada tenure +2 tahun dari batas data ini diproyeksikan sekitar **Rp{proj_value:,.0f}**.")

        st.divider()

        # --- G. Segmentasi otomatis (clustering) ---
        st.subheader("7. Segmentasi Otomatis (K-Means)")
        k = st.slider("Jumlah segmen (k)", 2, 6, 4)

        cluster_features = ['net_profit_margin', 'digital_adoption_score', 'repeat_order_rate', 'kepuasan_pelanggan']
        X_cluster = df[cluster_features].copy()
        X_scaled = StandardScaler().fit_transform(X_cluster)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_cluster_result = df.copy()
        df_cluster_result['segment'] = kmeans.fit_predict(X_scaled).astype(str)

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            fig_cluster = px.scatter(
                df_cluster_result, x='monthly_revenue', y='net_profit_margin',
                color='segment', hover_data=['class'],
                title="Segmen Otomatis vs Revenue & Margin"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        with col_s2:
            st.markdown("**Perbandingan segmen vs kelas manual**")
            crosstab = pd.crosstab(df_cluster_result['segment'], df_cluster_result['class'])
            st.dataframe(crosstab)

        st.caption("Segmentasi ini dibentuk otomatis dari pola data (margin, adopsi digital, loyalitas, kepuasan), independen dari label kelas yang sudah ada, untuk melihat apakah muncul pengelompokan baru yang bermakna.")

else:
    st.warning("Silakan periksa kembali ketersediaan dataset di repository GitHub Anda.")
