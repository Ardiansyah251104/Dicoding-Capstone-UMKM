import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="UMKM Investment Dashboard", layout="wide")

# Fungsi Load Data
@st.cache_data
def load_data():
    # Menggunakan dataset_bersih.csv sesuai permintaan
    df = pd.read_csv('dataset_bersih.csv')
    return df

df = load_data()

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
    
    # Filter data Top Growth (Target Investor)
    top_growth = df[(df['class'] == 'Growth') & (df['net_profit_margin'] > min_margin) & (df['monthly_revenue'] > avg_rev)]
    
    st.info(f"Rata-rata Monthly Revenue: Rp{avg_rev:,.2f}")
    st.success(f"Jumlah UMKM Top Growth (Target Investor): **{len(top_growth)}** UMKM")

    # --- VISUALISASI TAMBAHAN (Sesuai Gambar) ---
    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        # Scatter Plot: Distribusi Revenue vs Margin
        fig_scatter = px.scatter(
            df, 
            x='monthly_revenue', 
            y='net_profit_margin',
            color='class',
            color_discrete_map={
                'Growth': 'rgba(100, 149, 237, 0.5)', 
                'Struggling': 'rgba(173, 216, 230, 0.5)',
                'Elite': 'rgba(144, 238, 144, 0.5)',
                'Critical': 'rgba(211, 211, 211, 0.5)'
            },
            title="Distribusi Revenue vs Margin UMKM",
            labels={'monthly_revenue': 'Monthly Revenue (Rp)', 'net_profit_margin': 'Net Profit Margin (%)'}
        )

        # Menambahkan titik merah untuk Top Growth (Target)
        fig_scatter.add_trace(go.Scatter(
            x=top_growth['monthly_revenue'],
            y=top_growth['net_profit_margin'],
            mode='markers',
            marker=dict(color='red', size=6),
            name='Top Growth (Target)'
        ))

        # Menambahkan garis bantu (Average Revenue & Min Margin)
        fig_scatter.add_vline(x=avg_rev, line_dash="dash", line_color="gray", annotation_text="Avg Revenue")
        fig_scatter.add_hline(y=min_margin, line_dash="dash", line_color="orange", annotation_text="Min Margin: 15%")
        
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_chart2:
        # Pie Chart: Proporsi UMKM Top Growth
        total_umkm = len(df)
        n_top_growth = len(top_growth)
        n_others = total_umkm - n_top_growth
        
        proporsi_data = pd.DataFrame({
            'Kategori': ['Top Growth', 'Others'],
            'Jumlah': [n_top_growth, n_others]
        })
        
        fig_pie_target = px.pie(
            proporsi_data, 
            values='Jumlah', 
            names='Kategori',
            title="Proporsi UMKM Top Growth (Target Investor)",
            color='Kategori',
            color_discrete_map={'Top Growth': 'red', 'Others': '#4285F4'},
            hole=0.3
        )
        # Efek 'pull' untuk Top Growth agar menonjol seperti di gambar
        fig_pie_target.update_traces(pull=[0.2, 0], textinfo='percent+label')
        
        st.plotly_chart(fig_pie_target, use_container_width=True)

    st.divider()

    # 2. Pertanyaan 2 (Tenure vs Loyalty)
    st.subheader("Pertanyaan 2: Dampak Masa Operasional (Tenure) terhadap Loyalitas")

    # Menghitung rata-rata Repeat Order per grup menggunakan kolom yang sudah ada di CSV
    tenure_comparison = df.groupby('business_tenure_months')['repeat_order_rate'].mean().reset_index()

    # Visualisasi menggunakan Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='business_tenure_months',
        y='repeat_order_rate',
        data=tenure_comparison,
        palette=['#4e79a7', '#59a14f'],
        order=['Di atas 24 Bulan', 'Di bawah 24 Bulan'],
        ax=ax
    )

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points',
                    fontweight='bold')

    ax.set_title('Perbandingan Loyalitas Pelanggan Berdasarkan Masa Operasional', fontsize=14, pad=20)
    ax.set_xlabel('Kelompok Masa Operasional (Tenure)', fontsize=12)
    ax.set_ylabel('Rata-rata Repeat Order Rate (%)', fontsize=12)
    st.pyplot(fig)

# --- HALAMAN 3: ANALISIS NARATIF KELAS ---
elif page == "Analisis Naratif Kelas":
    st.title("📝 Perbandingan Karakteristik Antar Kelas")
    
    # Menghitung statistik per kelas (Kolom disesuaikan dengan dataset_bersih.csv)
    # Catatan: dataset_bersih.csv tidak memiliki Burn_Rate_Ratio, maka sementara dihapus atau diganti
    metrics = ['net_profit_margin', 'digital_adoption_score', 'repeat_order_rate', 'kepuasan_pelanggan']
    class_stats = df.groupby('class')[metrics].mean()

    # Normalisasi data untuk radar chart
    df_norm = (class_stats - class_stats.min()) / (class_stats.max() - class_stats.min())

    def create_radar_chart(target_class, color):
        categories = ['Profit Margin', 'Digital Adoption', 'Repeat Order', 'Kepuasan']
        values = df_norm.loc[target_class].values.tolist()
        values += values[:1]
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

    # List Kelas untuk iterasi narasi
    classes = [("Elite", tab1, "🏆"), ("Growth", tab2, "📈"), ("Struggling", tab3, "⚠️"), ("Critical", tab4, "🚨")]

    for name, tab, icon in classes:
        with tab:
            col_text, col_plot = st.columns([1, 1])
            with col_text:
                st.markdown(f"### {icon} Kelas {name}")
                st.markdown(f"""
                * **Rata-rata Margin**: {class_stats.loc[name, 'net_profit_margin']:.2f}%
                * **Skor Digital**: {class_stats.loc[name, 'digital_adoption_score']:.2f}/10
                * **Loyalitas (Repeat Order)**: {class_stats.loc[name, 'repeat_order_rate']:.2f}%
                * **Kepuasan Pelanggan**: {class_stats.loc[name, 'kepuasan_pelanggan']:.2f}/5
                """)
            with col_plot:
                st.plotly_chart(create_radar_chart(name, class_colors[name]), use_container_width=True)