import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="UMKM Investment Dashboard", layout="wide")

# --- FUNGSI LOAD DATA (DIPERBAIKI) ---
@st.cache_data
def load_data():
    # Mendapatkan jalur absolut ke folder tempat dashboard.py berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'dataset_bersih.csv')
    
    # Cek apakah file ada sebelum dibaca
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        st.error(f"File tidak ditemukan di: {file_path}. Pastikan file 'dataset_bersih.csv' ada di folder dashboard.")
        return None

df = load_data()

# Lanjutkan hanya jika data berhasil dimuat
if df is not None:
    # --- NAVIGASI SIDEBAR ---
    st.sidebar.title("Navigasi Dashboard")
    page = st.sidebar.radio("Pilih Halaman:", ["Overview Keseluruhan", "Pertanyaan Bisnis", "Analisis Naratif Kelas"])

    # --- HALAMAN 1: OVERVIEW KESELURUHAN ---
    if page == "Overview Keseluruhan":
        st.title("📊 Representasi Keseluruhan Data UMKM")
        st.write("Halaman ini menampilkan gambaran umum performa seluruh UMKM dalam dataset.")

        # KPI Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total UMKM", f"{len(df):,}")
        col2.metric("Rata-rata Revenue", f"Rp{df['monthly_revenue'].mean():,.0f}")
        col3.metric("Rata-rata Profit Margin", f"{df['net_profit_margin'].mean():.2f}%")
        col4.metric("Adopsi Digital", f"{df['digital_adoption_score'].mean():.2f}/10")

        # Visualisasi Distribusi Kelas
        st.subheader("Distribusi Kelas Bisnis")
        fig_pie = px.pie(df, names='class', title="Persentase Berdasarkan Kelas", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Data Table
        st.subheader("Sampel Data")
        st.dataframe(df.head(10))

    # --- HALAMAN 2: PERTANYAAN BISNIS ---
    elif page == "Pertanyaan Bisnis":
        st.title("💡 Jawaban Pertanyaan Bisnis")

        # 1. Pertanyaan 1 (Top Growth)
        st.subheader("Pertanyaan 1: UMKM 'Top Growth' untuk Investasi")
        
        avg_rev = df['monthly_revenue'].mean()
        min_margin = 15
        
        # Filter data Top Growth
        top_growth = df[(df['class'] == 'Growth') & (df['net_profit_margin'] > min_margin) & (df['monthly_revenue'] > avg_rev)]
        
        st.info(f"Rata-rata Monthly Revenue: Rp{avg_rev:,.2f}")
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

        st.divider()

        # 2. Pertanyaan 2 (Tenure vs Loyalty)
        st.subheader("Pertanyaan 2: Dampak Masa Operasional terhadap Loyalitas")
        tenure_comparison = df.groupby('business_tenure_months')['repeat_order_rate'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x='business_tenure_months',
            y='repeat_order_rate',
            data=tenure_comparison,
            palette='viridis',
            ax=ax
        )
        ax.set_title('Loyalitas Berdasarkan Masa Operasional')
        st.pyplot(fig)
# --- HALAMAN 3: ANALISIS NARATIF KELAS ---
    elif page == "Analisis Naratif Kelas":
        st.title("📝 Perbandingan Karakteristik Antar Kelas")
        
        metrics = ['net_profit_margin', 'digital_adoption_score', 'repeat_order_rate', 'kepuasan_pelanggan']
        class_stats = df.groupby('class')[metrics].mean()

        # Normalisasi untuk Radar Chart
        df_norm = (class_stats - class_stats.min()) / (class_stats.max() - class_stats.min())

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
        tabs = st.tabs(["Elite", "Growth", "Struggling", "Critical"])
        class_info = [
            ("Elite", tabs[0], "#FFD700", "🏆"),
            ("Growth", tabs[1], "#00FF00", "📈"),
            ("Struggling", tabs[2], "#FFA500", "⚠️"),
            ("Critical", tabs[3], "#FF0000", "🚨")
        ]

        for name, tab, color, icon in class_info:
            with tab:
                col_text, col_plot = st.columns(2)
                with col_text:
                    st.markdown(f"### {icon} Kelas {name}")
                    st.write(f"**Margin**: {class_stats.loc[name, 'net_profit_margin']:.2f}%")
                    st.write(f"**Digital**: {class_stats.loc[name, 'digital_adoption_score']:.2f}/10")
                    st.write(f"**Loyalty**: {class_stats.loc[name, 'repeat_order_rate']:.2f}%")
                with col_plot:
                    st.plotly_chart(create_radar_chart(name, color), use_container_width=True)
else:
    st.warning("Silakan periksa kembali ketersediaan dataset di repository GitHub Anda.")
