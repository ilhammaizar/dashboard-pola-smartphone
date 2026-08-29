import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Penggunaan Smartphone", page_icon="📱", layout="wide"
)

st.title("📱 Sistem Analisis Pola Penggunaan Smartphone & Kesehatan Digital")
st.markdown(
    "Implementasi Metode Hibrida **Principal Component Analysis (PCA)** dan"
    " **K-Means Clustering**."
)

# ==========================================
# 2. UPLOAD DATASET
# ==========================================
st.sidebar.title("📂 Input Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Unggah File Dataset (CSV / Excel):", type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
  st.info(
      "👋 **Selamat Datang!** Silakan unggah berkas dataset (.csv, .xlsx, atau"
      " .xls) Anda melalui bilah menu di sebelah kiri untuk memuat dashboard."
  )
  st.stop()

# ==========================================
# 3. PIPELINE PEMBACAAN & PEMBERSIHAN DATA
# ==========================================
try:
  file_name = uploaded_file.name.lower()
  if file_name.endswith(".csv"):
    try:
      df = pd.read_csv(uploaded_file)
    except Exception:
      uploaded_file.seek(0)
      df = pd.read_csv(uploaded_file, sep=";")
  elif file_name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
  elif file_name.endswith(".xls"):
    try:
      df = pd.read_excel(uploaded_file, engine="xlrd")
    except Exception:
      uploaded_file.seek(0)
      df = pd.read_excel(uploaded_file)
  else:
    df = pd.read_csv(uploaded_file)
except Exception as e:
  st.error(f"❌ Gagal membaca file: {e}")
  st.stop()

# Bersihkan nama kolom dari spasi
df.columns = df.columns.astype(str).str.strip()

# Pemetaan fleksibel untuk mencocokkan variasi penamaan kolom
# Pemetaan fleksibel dan super lengkap untuk mencocokkan variasi penamaan kolom
col_mapping = {}
for col in df.columns:
  c_clean = col.lower().replace(' ', '').replace('_', '').replace('-', '')
  if 'usia' in c_clean or 'age' in c_clean or 'umur' in c_clean:
    col_mapping[col] = 'Usia Saat Ini'
  elif (
      'weekend' in c_clean
      or 'akhirpekan' in c_clean
      or 'sabtuminggu' in c_clean
      or 'libur' in c_clean
  ):
    col_mapping[col] = 'Weekend_Screen_Time'
  elif (
      'dailyscreen' in c_clean
      or 'screentime' in c_clean
      or 'harian' in c_clean
      or 'durasi' in c_clean
  ):
    col_mapping[col] = 'Daily_Screen_Time'
  elif (
      'social' in c_clean
      or 'medsos' in c_clean
      or 'sosial' in c_clean
      or 'instagram' in c_clean
      or 'tiktok' in c_clean
  ):
    col_mapping[col] = 'Social_Media'
  elif 'game' in c_clean or 'gaming' in c_clean or 'bermain' in c_clean:
    col_mapping[col] = 'Gaming'
  elif (
      'work' in c_clean
      or 'study' in c_clean
      or 'tugas' in c_clean
      or 'kerja' in c_clean
      or 'belajar' in c_clean
  ):
    col_mapping[col] = 'Work_Study'
  elif 'sleep' in c_clean or 'tidur' in c_clean:
    col_mapping[col] = 'Sleep'
  elif 'notif' in c_clean or 'pemberitahuan' in c_clean:
    col_mapping[col] = 'Notifications'
  elif 'open' in c_clean or 'buka' in c_clean or 'akses' in c_clean:
    col_mapping[col] = 'App_Opens'
  elif 'stress' in c_clean or 'stres' in c_clean:
    col_mapping[col] = 'Strees_level'
  elif 'addict' in c_clean or 'adiksi' in c_clean or 'kecanduan' in c_clean:
    col_mapping[col] = 'Addiction_label'

df = df.rename(columns=col_mapping)

num_features = [
    "Usia Saat Ini",
    "Daily_Screen_Time",
    "Social_Media",
    "Gaming",
    "Work_Study",
    "Sleep",
    "Weekend_Screen_Time",
    "Notifications",
    "App_Opens",
]

# Validasi kolom kuantitatif
missing_cols = [c for c in num_features if c not in df.columns]
if missing_cols:
  st.error(
      f"⚠️ Kolom berikut tidak ditemukan pada dataset Anda: **{missing_cols}**."
  )
  st.write("Kolom yang terdeteksi di file Anda adalah:", list(df.columns))
  st.stop()

# Konversi koma desimal ke float numerik
for c in num_features:
  if df[c].dtype == object:
    df[c] = df[c].astype(str).str.replace(",", ".").str.strip()
  df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ==========================================
# 4. KOMPUTASI PCA & K-MEANS DINAMIS
# ==========================================
# Standardisasi Z-Score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_features].astype(float))

# Reduksi Dimensi PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
var_total = var_pc1 + var_pc2

# Evaluasi Nilai K (K=1 s/d K=8)
k_range = list(range(1, 9))
wcss_list = []
sil_list = [np.nan]

for k in k_range:
  km_eval = KMeans(
      n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42
  )
  labels_eval = km_eval.fit_predict(X_pca)
  wcss_list.append(km_eval.inertia_)
  if k > 1:
    sil_list.append(silhouette_score(X_pca, labels_eval))

# Pemodelan K-Means Final (K=4)
kmeans = KMeans(
    n_clusters=4, init="k-means++", n_init=10, max_iter=300, random_state=42
)
df["Cluster"] = kmeans.fit_predict(X_pca)
df_pca["Cluster"] = df["Cluster"]

sil_k4 = silhouette_score(X_pca, df["Cluster"])

persona_mapping = {
    0: "Klaster 0: Heavy / High-Risk Users",
    1: "Klaster 1: Moderate / Casual Users",
    2: "Klaster 2: Adaptive / Healthy Users",
    3: "Klaster 3: Extreme / Specialized Users",
}
df["Persona"] = df["Cluster"].map(persona_mapping)
df_pca["Persona"] = df["Cluster"].map(persona_mapping)

# ==========================================
# 5. MENU NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Tampilan Menu:",
    [
        "📊 1. Ringkasan Data & Statistik",
        "📈 2. Evaluasi K Optimal (Elbow & Silhouette)",
        "🗺️ 3. Visualisasi Spasial PCA & Klaster",
        "📋 4. Profiling & Tabulasi Silang",
        "🔮 5. Prediksi Persona Pengguna Baru",
    ],
)

# ==========================================
# 6. KONTEN BERDASARKAN MENU
# ==========================================

# --- MENU 1: RINGKASAN DATA ---
if menu == "📊 1. Ringkasan Data & Statistik":
  st.header("📊 Ringkasan Dataset & Statistik Deskriptif")
  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Total Responden (N)", f"{len(df)} Orang")
  c2.metric(
      "Rata-Rata Daily Screen Time",
      f"{df['Daily_Screen_Time'].mean():.2f} Jam/Hari",
  )
  c3.metric("Rasio Varians PCA 2D", f"{var_total:.2f}%")
  c4.metric("Silhouette Score (K=4)", f"{sil_k4:.4f}")

  st.markdown("---")
  st.subheader("Data Primer Responden (10 Baris Teratas)")
  st.dataframe(df.head(10), use_container_width=True)

  st.subheader("Statistik Deskriptif 9 Fitur Kuantitatif")
  desc = df[num_features].describe().T[["min", "max", "mean", "std"]]
  desc.columns = [
      "Nilai Min",
      "Nilai Maks",
      "Rata-Rata (Mean)",
      "Standar Deviasi",
  ]
  st.dataframe(desc.round(2), use_container_width=True)

# --- MENU 2: EVALUASI ELBOW & SILHOUETTE ---
elif menu == "📈 2. Evaluasi K Optimal (Elbow & Silhouette)":
  st.header(
      "📈 Penentuan Jumlah Klaster Optimal (Elbow Method & Silhouette Score)"
  )
  st.markdown(
      "Pengujian inersia varians internal (**WCSS**) dan derajat keterpisahan"
      " (**Silhouette Score**) untuk membuktikan konfigurasi $K$ paling ideal."
  )

  col_chart, col_tbl = st.columns([6, 4])

  with col_chart:
    fig_eval, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)

    # Inersia WCSS (Garis Biru)
    color_wcss = "#1f77b4"
    ax1.set_xlabel("Jumlah Klaster (K)", fontweight="bold", fontsize=10)
    ax1.set_ylabel(
        "Inersia WCSS (Elbow)", color=color_wcss, fontweight="bold", fontsize=10
    )
    ax1.plot(
        k_range,
        wcss_list,
        marker="o",
        color=color_wcss,
        linewidth=2.2,
        label="Inersia WCSS",
    )
    ax1.tick_params(axis="y", labelcolor=color_wcss)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Silhouette Score (Garis Merah Sumbu Kanan)
    ax2 = ax1.twinx()
    color_sil = "#d62728"
    ax2.set_ylabel(
        "Silhouette Score", color=color_sil, fontweight="bold", fontsize=10
    )
    ax2.plot(
        k_range[1:],
        sil_list[1:],
        marker="s",
        color=color_sil,
        linestyle="--",
        linewidth=2.2,
        label="Silhouette Score",
    )
    ax2.tick_params(axis="y", labelcolor=color_sil)

    # Garis Penanda K=4 Optimal
    ax1.axvline(
        x=4,
        color="green",
        linestyle=":",
        linewidth=2.5,
        label="K=4 Optimal (Elbow Point)",
    )
    plt.title(
        "Grafik Kurva Elbow Method & Silhouette Score",
        fontweight="bold",
        fontsize=11,
    )
    fig_eval.tight_layout()
    st.pyplot(fig_eval)

  with col_tbl:
    st.write("**Hasil Evaluasi Nilai K:**")
    df_eval_tbl = pd.DataFrame({
        "Jumlah Klaster (K)": [f"K = {k}" for k in k_range],
        "Inersia WCSS": [round(w, 2) for w in wcss_list],
        "Silhouette Score": [
            f"{s:.4f}" if not np.isnan(s) else "N/A" for s in sil_list
        ],
    })
    st.dataframe(df_eval_tbl, use_container_width=True, height=280)
    st.success(
        f"📌 Titik siku (*Elbow Point*) optimal terbentuk pada **K = 4** dengan"
        f" nilai **Silhouette Score = {sil_k4:.4f}** (struktur klaster kuat dan"
        " seimbang)."
    )

# --- MENU 3: VISUALISASI SPASIAL PCA & KLASTER ---
elif menu == "🗺️ 3. Visualisasi Spasial PCA & Klaster":
  st.header("🗺️ Visualisasi Spasial Reduksi Dimensi PCA & Klaster K-Means")

  col_pca, col_km = st.columns(2)

  with col_pca:
    st.subheader("Proyeksi PCA 2D")
    fig_pca, ax_p = plt.subplots(figsize=(6, 5), dpi=300)
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        color="#20639B",
        s=75,
        edgecolor="black",
        alpha=0.85,
        ax=ax_p,
    )
    ax_p.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_p.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_p.set_xlabel(
        f"PC1 ({var_pc1:.2f}% Varians) → Durasi Layar & Medsos",
        fontweight="bold",
        fontsize=9,
    )
    ax_p.set_ylabel(
        f"PC2 ({var_pc2:.2f}% Varians) → Usia & Produktivitas",
        fontweight="bold",
        fontsize=9,
    )
    ax_p.set_title(
        f"Proyeksi Ruang Spasial PCA (Total Varians {var_total:.2f}%)",
        fontweight="bold",
        fontsize=10,
    )
    fig_pca.tight_layout()
    st.pyplot(fig_pca)

  with col_km:
    st.subheader("Klaster K-Means 2D")
    fig_km, ax_k = plt.subplots(figsize=(6, 5), dpi=300)
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    persona_order = [
        "Klaster 0: Heavy / High-Risk Users",
        "Klaster 1: Moderate / Casual Users",
        "Klaster 2: Adaptive / Healthy Users",
        "Klaster 3: Extreme / Specialized Users",
    ]
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="Persona",
        hue_order=persona_order,
        palette=palette,
        s=80,
        edgecolor="black",
        alpha=0.85,
        ax=ax_k,
    )
    centroids = kmeans.cluster_centers_
    ax_k.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=200,
        c="black",
        edgecolor="yellow",
        linewidth=1.5,
        label="Centroids",
    )
    ax_k.set_xlabel(
        f"PC1 ({var_pc1:.2f}% Varians)", fontweight="bold", fontsize=9
    )
    ax_k.set_ylabel(
        f"PC2 ({var_pc2:.2f}% Varians)", fontweight="bold", fontsize=9
    )
    ax_k.legend(
        title="Persona Klaster", fontsize=7, title_fontsize=8, loc="upper right"
    )
    ax_k.set_title(
        "Partisi Spasial 4 Klaster & Centroid", fontweight="bold", fontsize=10
    )
    fig_km.tight_layout()
    st.pyplot(fig_km)

# --- MENU 4: PROFILING & TABULASI SILANG ---
elif menu == "📋 4. Profiling & Tabulasi Silang":
  st.header("📋 Karakteristik Persona & Tabulasi Silang Kesehatan")

  st.subheader("Rata-Rata Karakteristik 9 Fitur Riil per Klaster")
  prof = df.groupby("Persona")[num_features].mean().T
  st.dataframe(prof.round(2), use_container_width=True)

  st.markdown("---")
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Klaster vs Tingkat Stres (%)")
    if "Strees_level" in df.columns:
      ct_stress = (
          pd.crosstab(df["Persona"], df["Strees_level"], normalize="index")
          * 100
      )
      st.dataframe(ct_stress.round(1), use_container_width=True)
    else:
      st.info("Kolom 'Strees_level' tidak ditemukan di dataset.")
  with col2:
    st.subheader("Klaster vs Status Adiksi (%)")
    if "Addiction_label" in df.columns:
      ct_addict = (
          pd.crosstab(df["Persona"], df["Addiction_label"], normalize="index")
          * 100
      )
      st.dataframe(ct_addict.round(1), use_container_width=True)
    else:
      st.info("Kolom 'Addiction_label' tidak ditemukan di dataset.")

# --- MENU 5: PREDIKSI PERSONA BARU ---
elif menu == "🔮 5. Prediksi Persona Pengguna Baru":
  st.header("🔮 Simulasi & Prediksi Persona Pengguna Baru")
  st.write(
      "Masukkan estimasi durasi pemakaian smartphone untuk menguji model"
      " klasifikasi persona secara langsung:"
  )

  col1, col2, col3 = st.columns(3)
  with col1:
    in_age = st.number_input(
        "Usia Saat Ini (Tahun):", min_value=10, max_value=60, value=22
    )
    in_daily = st.slider("Daily Screen Time (Jam/Hari):", 0.0, 24.0, 8.0, 0.5)
    in_social = st.slider("Social Media (Jam/Hari):", 0.0, 24.0, 4.5, 0.5)
  with col2:
    in_gaming = st.slider("Gaming (Jam/Hari):", 0.0, 24.0, 2.0, 0.5)
    in_work = st.slider("Work / Study (Jam/Hari):", 0.0, 24.0, 4.0, 0.5)
    in_sleep = st.slider("Durasi Tidur (Jam/Hari):", 0.0, 24.0, 7.0, 0.5)
  with col3:
    in_weekend = st.slider(
        "Weekend Screen Time (Jam/Hari):", 0.0, 24.0, 11.0, 0.5
    )
    in_notif = st.number_input(
        "Estimasi Notifikasi per Hari:", min_value=0, max_value=1000, value=85
    )
    in_opens = st.number_input(
        "Frekuensi Membuka HP per Hari:", min_value=0, max_value=200, value=20
    )

  if st.button("🚀 Analisis & Prediksi Persona Saya", type="primary"):
    user_df = pd.DataFrame([[
        in_age,
        in_daily,
        in_social,
        in_gaming,
        in_work,
        in_sleep,
        in_weekend,
        in_notif,
        in_opens,
    ]],
                           columns=num_features)
    user_scaled = scaler.transform(user_df)
    user_pca = pca.transform(user_scaled)
    pred_cluster = kmeans.predict(user_pca)[0]
    persona_name = persona_mapping[pred_cluster]

    st.success(f"### Hasil Prediksi: **{persona_name}**")
    st.caption(
        f"Posisi Koordinat Spasial 2D: PC1 = {user_pca[0][0]:.2f}, PC2 ="
        f" {user_pca[0][1]:.2f}"
    )

    if pred_cluster == 0:
      st.warning(
          "⚠️ **Pola Penggunaan Berisiko Tinggi:** Pola penggunaan Anda dominan"
          " pada konsumsi hiburan pasif (medsos & game). Disarankan membatasi"
          " screen time dan melakukan digital detox."
      )
    elif pred_cluster == 1:
      st.info(
          "ℹ️ **Penggunaan Kasual / Moderat:** Pola pemakaian seimbang dan"
          " dominan untuk aktivitas produktif/pekerjaan."
      )
    elif pred_cluster == 2:
      st.success(
          "✅ **Pola Penggunaan Sehat & Adaptif:** Durasi layar seimbang dan jam"
          " tidur optimal."
      )
    elif pred_cluster == 3:
      st.error(
          "🚨 **Penggunaan Ekstrem:** Durasi layar harian sangat tinggi (>15"
          " jam/hari). Waspadai kelelahan leher (*text neck syndrome*) dan"
          " gangguan kualitas tidur."
      )