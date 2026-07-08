
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================================
# TAMPILAN INSIGHT BOX (SERAGAM DI SELURUH DASHBOARD)
# ==========================================================

def insight_box(text: str, icon: str = "💡"):
    """Menampilkan insight otomatis dengan gaya seragam."""
    st.markdown(
        f"""
        <div style="
            background-color:#EFF6FF;
            border-left:4px solid #3B82F6;
            padding:12px 16px;
            border-radius:6px;
            margin-top:6px;
            margin-bottom:6px;
        ">
        <span style="font-weight:600;color:#1D4ED8;">{icon} Insight </span><br>
        <span style="color:#1E293B;">{text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def _safe(fn, fallback="Insight tidak tersedia untuk data pada filter ini."):
    """Wrapper agar dashboard tidak crash kalau data hasil filter terlalu sedikit/kosong."""
    try:
        return fn()
    except Exception:
        return fallback


# ==========================================================
# 1. INSIGHT DISTRIBUSI KATEGORIKAL (Pie / Bar kategori)
# ==========================================================

def insight_categorical_distribution(series: pd.Series, label: str = "data") -> str:
    def _run():
        counts = series.value_counts(normalize=True) * 100
        top, top_pct = counts.index[0], counts.iloc[0]
        bottom, bottom_pct = counts.index[-1], counts.iloc[-1]
        if len(counts) == 1:
            return f"Seluruh {label} berada pada satu kategori yaitu {top} ({top_pct:.1f}%)."
        return (
            f"{top} mendominasi dengan {top_pct:.1f}% dari total {label}, "
            f"sementara {bottom} menjadi kelompok terkecil ({bottom_pct:.1f}%)."
        )
    return _safe(_run)


# ==========================================================
# 2. INSIGHT PERBANDINGAN ANTAR GRUP (Bar chart rata-rata per kategori)
# ==========================================================

def insight_group_comparison(df: pd.DataFrame, group_col: str, value_col: str,
                              unit: str = "", label: str = None) -> str:
    def _run():
        nice_name = label or value_col.replace("_", " ")
        grouped = df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
        top_group, top_val = grouped.index[0], grouped.iloc[0]
        bottom_group, bottom_val = grouped.index[-1], grouped.iloc[-1]
        diff = top_val - bottom_val
        pct_diff = (diff / abs(bottom_val) * 100) if bottom_val != 0 else np.nan
        pct_text = f" (selisih {pct_diff:.1f}%)" if np.isfinite(pct_diff) else ""
        return (
            f"{top_group} memiliki rata-rata {nice_name} tertinggi ({top_val:.2f}{unit}), "
            f"unggul {diff:.2f}{unit}{pct_text} dibanding {bottom_group} "
            f"({bottom_val:.2f}{unit}) yang terendah."
        )
    return _safe(_run)


# ==========================================================
# 3. INSIGHT DISTRIBUSI NUMERIK (Histogram)
# ==========================================================

def insight_distribution(series: pd.Series, name: str, unit: str = "") -> str:
    def _run():
        mean = series.mean()
        median = series.median()
        skew = series.skew()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_pct = ((series < lower) | (series > upper)).mean() * 100

        if skew > 0.5:
            shape = "menceng ke kanan (right-skewed) — sebagian kecil UMKM bernilai jauh di atas rata-rata"
        elif skew < -0.5:
            shape = "menceng ke kiri (left-skewed) — sebagian kecil UMKM bernilai jauh di bawah rata-rata"
        else:
            shape = "cenderung simetris mendekati distribusi normal"

        return (
            f"Distribusi {name} {shape}. Rata-rata berada di {mean:,.2f}{unit}, "
            f"sedangkan median {median:,.2f}{unit}, dengan sekitar {outlier_pct:.1f}% data "
            f"tergolong outlier (di luar rentang IQR)."
        )
    return _safe(_run)


# ==========================================================
# 4. INSIGHT BOXPLOT ANTAR GRUP
# ==========================================================

def insight_boxplot_by_group(df: pd.DataFrame, group_col: str, value_col: str, unit: str = "") -> str:
    def _run():
        nice_name = value_col.replace("_", " ")
        med = df.groupby(group_col)[value_col].median().sort_values(ascending=False)
        top, top_val = med.index[0], med.iloc[0]
        bottom, bottom_val = med.index[-1], med.iloc[-1]
        neg_share = None
        if (df[value_col] < 0).any():
            neg_share = (df[value_col] < 0).mean() * 100
        text = (
            f"Median {nice_name} tertinggi ada pada kelas {top} ({top_val:.2f}{unit}), "
            f"sedangkan kelas {bottom} memiliki median terendah ({bottom_val:.2f}{unit})."
        )
        if neg_share is not None and neg_share > 0:
            text += f" Sekitar {neg_share:.1f}% dari seluruh data bahkan bernilai negatif."
        return text
    return _safe(_run)


# ==========================================================
# 5. INSIGHT KORELASI / DRIVER UTAMA
# ==========================================================

def insight_correlation_driver(corr_series: pd.Series, target_label: str = "Net Profit Margin") -> str:
    def _run():
        strongest = corr_series.abs().idxmax()
        val = corr_series[strongest]
        arah = "positif" if val > 0 else "negatif"
        kekuatan = "kuat" if abs(val) >= 0.5 else ("sedang" if abs(val) >= 0.3 else "lemah")
        second = corr_series.drop(strongest).abs().idxmax()
        second_val = corr_series[second]
        return (
            f"Faktor dengan pengaruh {kekuatan} terhadap {target_label} adalah "
            f"{strongest.replace('_', ' ')} (korelasi {arah} {val:.2f}), "
            f"diikuti oleh {second.replace('_', ' ')} ({second_val:.2f})."
        )
    return _safe(_run)


# ==========================================================
# 6. INSIGHT TREN (naik / turun seiring waktu-proksi)
# ==========================================================

def insight_trend(y_values, name: str, unit: str = "") -> str:
    def _run():
        clean = pd.Series(y_values).dropna().values
        if len(clean) < 2:
            raise ValueError("data tidak cukup")
        x = np.arange(len(clean))
        slope = np.polyfit(x, clean, 1)[0]
        total_change = clean[-1] - clean[0]
        if slope > 0:
            arah = "cenderung meningkat 📈"
        elif slope < 0:
            arah = "cenderung menurun 📉"
        else:
            arah = "relatif stabil ➡"
        return (
            f"Tren {name} seiring bertambahnya masa operasional {arah}, dengan total perubahan "
            f"sekitar {total_change:,.2f}{unit} dari kategori pertama ke terakhir."
        )
    return _safe(_run)


# ==========================================================
# 7. INSIGHT FORECAST REGRESI
# ==========================================================

def insight_forecast(future_x, pred_y, unit: str = "Rp") -> str:
    def _run():
        growth = pred_y[-1] - pred_y[0]
        arah = "meningkat" if growth > 0 else "menurun"
        return (
            f"Model regresi memproyeksikan monthly revenue {arah} sekitar {unit}{abs(growth):,.0f}** "
            f"dari masa operasional {future_x[0][0]:.0f} tahun hingga {future_x[-1][0]:.0f} tahun. "
            f"Proyeksi ini bersifat linear sederhana dan sebaiknya digunakan sebagai indikasi arah, bukan angka pasti."
        )
    return _safe(_run)


# ==========================================================
# 8. INSIGHT CLUSTERING / SEGMENTASI
# ==========================================================

def insight_cluster(cluster_df: pd.DataFrame, segment_col: str, value_col: str, unit: str = "") -> str:
    def _run():
        nice_name = value_col.replace("_", " ")
        seg_stats = cluster_df.groupby(segment_col)[value_col].mean().sort_values(ascending=False)
        top, top_val = seg_stats.index[0], seg_stats.iloc[0]
        bottom, bottom_val = seg_stats.index[-1], seg_stats.iloc[-1]
        sizes = cluster_df[segment_col].value_counts(normalize=True) * 100
        top_size = sizes.get(top, np.nan)
        return (
            f"Segmen {top} memiliki rata-rata {nice_name} tertinggi ({top_val:.2f}{unit}), "
            f"dan mencakup {top_size:.1f}% dari data terfilter, sedangkan segmen {bottom} "
            f"paling rendah ({bottom_val:.2f}{unit})."
        )
    return _safe(_run)


# ==========================================================
# 9. INSIGHT RADAR PER KELAS
# ==========================================================

def insight_radar(class_stats: pd.DataFrame, target_class: str) -> str:
    def _run():
        row = class_stats.loc[target_class]
        strongest = row.idxmax()
        weakest = row.idxmin()
        return (
            f"Kelas {target_class} paling unggul pada aspek {strongest.replace('_', ' ')}, "
            f"namun relatif lemah pada aspek {weakest.replace('_', ' ')} dibanding aspek lainnya."
        )
    return _safe(_run)


# ==========================================================
# 10. INSIGHT SCATTER + HIGHLIGHT SUBSET (mis. Top Growth)
# ==========================================================

def insight_scatter_highlight(df: pd.DataFrame, highlight_df: pd.DataFrame,
                               x_col: str, y_col: str, highlight_label: str) -> str:
    def _run():
        share = len(highlight_df) / len(df) * 100 if len(df) else 0
        corr = df[[x_col, y_col]].corr().iloc[0, 1]
        arah = "positif" if corr > 0 else "negatif"
        return (
            f"{highlight_label} mencakup {share:.1f}% dari total data terfilter ({len(highlight_df):,} UMKM). "
            f"Secara umum, {x_col.replace('_',' ')} dan {y_col.replace('_',' ')} menunjukkan korelasi "
            f"{arah} sebesar {corr:.2f}."
        )
    return _safe(_run)


# ==========================================================
# 11. INSIGHT KPI ALERT
# ==========================================================

def insight_kpi_alert(margin_pct: float, digital_pct: float,
                       margin_limit: float, digital_limit: float) -> str:
    def _run():
        level = "tinggi" if max(margin_pct, digital_pct) > 30 else (
            "sedang" if max(margin_pct, digital_pct) > 10 else "rendah"
        )
        return (
            f"Sebanyak {margin_pct:.1f}% UMKM berada di bawah margin {margin_limit}%, dan "
            f"{digital_pct:.1f}% berada di bawah skor digital {digital_limit}. "
            f"Tingkat risiko keseluruhan tergolong {level} — semakin tinggi persentase ini, "
            f"semakin banyak UMKM yang berpotensi butuh pendampingan."
        )
    return _safe(_run)


# ==========================================================
# 12. INSIGHT COHORT TABLE
# ==========================================================

def insight_cohort(cohort_df: pd.DataFrame, value_col: str = "Margin") -> str:
    def _run():
        best_row = cohort_df.loc[cohort_df[value_col].idxmax()]
        worst_row = cohort_df.loc[cohort_df[value_col].idxmin()]
        cohort_col = cohort_df.columns[0]
        return (
            f"Kelompok {best_row[cohort_col]} mencatat rata-rata {value_col.lower()} tertinggi "
            f"({best_row[value_col]:.2f}), sementara kelompok {worst_row[cohort_col]} paling rendah "
            f"({worst_row[value_col]:.2f})."
        )
    return _safe(_run)