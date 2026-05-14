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
    
    # Menghitung statistik per kelas untuk narasi (Disesuaikan dengan kolom dataset_bersih.csv)
    # Catatan: Karena dataset_bersih.csv tidak memiliki Burn_Rate_Ratio, kita gunakan metrik tersedia
    metrics = ['net_profit_margin', 'digital_adoption_score', 'repeat_order_rate', 'kepuasan_pelanggan']
    class_stats = df.groupby('class')[metrics].mean()

    # 2. Normalisasi data untuk visualisasi radar (Skala 0-1)
    df_norm = (class_stats - class_stats.min()) / (class_stats.max() - class_stats.min())

    # 3. Fungsi untuk membuat Radar Chart
    def create_radar_chart(target_class, color):
        categories = ['Profit Margin', 'Digital Adoption', 'Repeat Order', 'Kepuasan']
        values = df_norm.loc[target_class].values.tolist()
        values += values[:1]  # Menutup garis radar
        categories += categories[:1]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'DNA {target_class}',
            line_color=color
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title=f"Radar DNA Performa: {target_class}",
            height=400
        )
        return fig

    st.subheader("Analisis Profil Bisnis")
    tab1, tab2, tab3, tab4 = st.tabs(["Elite", "Growth", "Struggling", "Critical"])
    class_colors = {"Elite": "#FFD700", "Growth": "#00FF00", "Struggling": "#FFA500", "Critical": "#FF0000"}

    with tab1:
        col_text, col_plot = st.columns([1, 1])
        with col_text:
            st.markdown(f"""
        ### 🏆 Kelas Elite (The Market Leader)
        Ini adalah kelompok UMKM terbaik dengan rata-rata Profit Margin sebesar **{class_stats.loc['Elite', 'net_profit_margin']:.2f}%**. 
        Mereka memiliki efisiensi operasional yang matang dan tingkat loyalitas pelanggan tertinggi.
        
        Tipe Investor: Cocok untuk investor yang mencari dividen stabil dan risiko rendah (Blue Chip UMKM).
        """)
        with col_plot:
            st.plotly_chart(create_radar_chart("Elite", class_colors["Elite"]), use_container_width=True)

    with tab2:
        col_text, col_plot = st.columns([1, 1])
        with col_text:
            st.markdown(f"""
        ### 📈 Kelas Growth (The Rising Star)
        Kelompok ini menunjukkan potensi ekspansi yang besar. Meskipun margin profit berada di angka **{class_stats.loc['Growth', 'net_profit_margin']:.2f}%**, 
        skor adopsi digital mereka menunjukkan kesiapan untuk skala yang lebih besar.

        Tipe Investor: Sangat cocok untuk Venture Capital atau Angel Investor yang mencari pertumbuhan nilai aset (Capital Gain).
        """)
        with col_plot:
            st.plotly_chart(create_radar_chart("Growth", class_colors["Growth"]), use_container_width=True)

    with tab3:
        col_text, col_plot = st.columns([1, 1])
        with col_text:
            st.markdown(f"""
        ### ⚠️ Kelas Struggling (Needs Optimization)
        UMKM di kelas ini memiliki margin profit yang tipis (**{class_stats.loc['Struggling', 'net_profit_margin']:.2f}%**). 
        Mereka membutuhkan intervensi pada strategi retensi pelanggan dan optimalisasi biaya operasional.

        Tipe Investor: Cocok untuk Investor Turnaround yang juga berperan sebagai mentor untuk memperbaiki struktur biaya bisnis.
        """)
        with col_plot:
            st.plotly_chart(create_radar_chart("Struggling", class_colors["Struggling"]), use_container_width=True)

    with tab4:
        col_text, col_plot = st.columns([1, 1])
        with col_text:
            st.markdown(f"""
        ### 🚨 Kelas Critical (The High Risk)
        Ini adalah zona risiko tinggi dengan margin profit sangat tertekan (**{class_stats.loc['Critical', 'net_profit_margin']:.2f}%**). 
        Loyalitas pelanggan berada pada titik rendah, menandakan perlunya pivot atau perubahan model bisnis.

        Tipe Investor: Cocok untuk Investor Spekulatif yang melihat aset strategis tersembunyi di balik kegagalan operasional (Distressed Asset).
        """)
        with col_plot:
            st.plotly_chart(create_radar_chart("Critical", class_colors["Critical"]), use_container_width=True))
else:
    st.warning("Silakan periksa kembali ketersediaan dataset di repository GitHub Anda.")
