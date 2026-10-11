# ============================================================
# app.py - Peta Tutupan Lahan Jawa Timur
# Random Forest + Sentinel-2A (hasil dari notebook UTS)
# ============================================================

import os
import tempfile
from pathlib import Path

import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Peta Tutupan Lahan Jawa Timur",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1. KONFIGURASI
# ============================================================

FOLDER_DATA = Path(__file__).parent / "data"
FILE_MODEL = FOLDER_DATA / "model_random_forest.pkl"
FILE_DATASET = FOLDER_DATA / "dataset_uts_klasifikasi.csv"
FILE_RASTER = FOLDER_DATA / "hasil_klasifikasi_lulc.tif"

# Harus sama persis dengan notebook (training dan deployment)
FITUR = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI"]
UKURAN_UJI = 0.2
RANDOM_STATE = 21
NODATA = 255

# Cadangan bila metadata georeferensi di GeoTIFF tidak terbaca
BATAS_CADANGAN = (
    111.20948320931704,   # kiri
    -8.786054103228544,   # bawah
    114.61148320931704,   # kanan
    -7.146054103228545,   # atas
)

NAMA_KELAS = {
    0: "Sawah",
    1: "Bangunan",
    2: "Hutan",
    3: "Mangrove",
    4: "Danau",
    5: "Lautan",
}

KETERANGAN_KELAS = {
    0: "area persawahan",
    1: "permukiman & infrastruktur",
    2: "vegetasi pohon rapat",
    3: "hutan bakau di pesisir",
    4: "badan air daratan",
    5: "laut lepas",
}

WARNA = {
    0: "#F4C430",  # Sawah     - kuning emas
    1: "#D62828",  # Bangunan  - merah
    2: "#2D6A4F",  # Hutan     - hijau tua
    3: "#9D4EDD",  # Mangrove  - ungu
    4: "#4CC9F0",  # Danau     - biru muda
    5: "#1D3557",  # Lautan    - biru navy
}

URL_ESRI = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

MODE_PETA = [
    "Dua peta bertumpuk",
    "Dua peta sejajar",
    "Peta titik sampel saja",
    "Peta wilayah saja",
]

HALAMAN = ["🗺️ Peta", "📐 Luas Lahan", "🎯 Kinerja Model"]

st.markdown(
    """
    <style>
    .stat-box {background:#f8fafc; border:1px solid #e2e8f0; border-top:5px solid #0f766e;
               border-radius:12px; padding:16px 18px; min-height:250px; box-sizing:border-box;}
    .stat-box.merah {border-top-color:#ea580c;}
    .stat-box .lbl {font-size:.72rem; letter-spacing:.1em; font-weight:700; color:#64748b;}
    .stat-box .val {font-size:2.1rem; font-weight:800; color:#0f172a; line-height:1.3;}
    .stat-box .tag {display:inline-block; font-size:.76rem; font-weight:600; color:#0f766e;
                    background:#ccfbf1; border-radius:6px; padding:2px 10px; margin-bottom:6px;}
    .stat-box .tag.warn {color:#9a3412; background:#ffedd5;}
    .stat-box .note {font-size:.82rem; color:#334155; font-weight:600; margin-bottom:8px;}
    .stat-box .exp {font-size:.8rem; color:#64748b; line-height:1.5;
                    border-top:1px dashed #cbd5e1; padding-top:8px;}
    .panel-intro {background:linear-gradient(135deg,#0f766e,#115e59); color:white;
                  border-radius:14px; padding:20px 26px; margin-bottom:6px;}
    .panel-intro h3 {margin:0 0 8px 0; color:white;}
    .panel-intro p {margin:0; font-size:.93rem; line-height:1.6; color:#e6fffa;}
    .kotak-baca {background:#f0fdfa; border-left:5px solid #0f766e; border-radius:8px;
                 padding:12px 16px; font-size:.88rem; color:#134e4a; line-height:1.6;
                 margin:10px 0 14px 0;}
    .tabel-data {width:100%; border-collapse:collapse; font-size:.88rem;}
    .tabel-data th {background:#0f766e; color:white; text-align:left; padding:8px 10px;}
    .tabel-data td {padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:left;}
    .tabel-data tr:nth-child(even) td {background:#f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. CEK KELENGKAPAN FILE
# ============================================================

berkas_hilang = [
    p.name for p in (FILE_MODEL, FILE_DATASET, FILE_RASTER) if not p.exists()
]

if berkas_hilang:
    st.error(
        "File berikut belum ada di folder `data/`: "
        + ", ".join(f"`{n}`" for n in berkas_hilang)
    )
    st.stop()

# ============================================================
# 3. FUNGSI PEMUATAN DATA (DI-CACHE)
# ============================================================


def hex_ke_rgb(kode):
    kode = kode.lstrip("#")
    return tuple(int(kode[i:i + 2], 16) for i in (0, 2, 4))


@st.cache_resource(show_spinner="Memuat model Random Forest...")
def muat_model():
    return joblib.load(FILE_MODEL)


def baca_geotiff(path):
    """Baca GeoTIFF 1 band tanpa rasterio (memakai Pillow).
    Georeferensi dari tag ModelPixelScale (33550) dan ModelTiepoint (33922)."""
    with Image.open(path) as img:
        larik = np.array(img)
        skala = img.tag_v2.get(33550)
        titik = img.tag_v2.get(33922)

    if larik.ndim != 2:
        raise ValueError("Raster hasil klasifikasi harus 1 band.")

    tinggi, lebar = larik.shape

    if skala and titik:
        res_x, res_y = float(skala[0]), float(skala[1])
        kiri, atas = float(titik[3]), float(titik[4])
        kanan = kiri + lebar * res_x
        bawah = atas - tinggi * res_y
    else:
        kiri, bawah, kanan, atas = BATAS_CADANGAN
        res_x = (kanan - kiri) / lebar
        res_y = (atas - bawah) / tinggi

    return larik, (kiri, bawah, kanan, atas), (res_x, res_y)


@st.cache_resource(show_spinner="Membaca raster klasifikasi...")
def muat_raster():
    larik, batas, resolusi = baca_geotiff(FILE_RASTER)
    return {"larik": larik, "batas": batas, "resolusi": resolusi}


@st.cache_data(show_spinner="Mengevaluasi model...")
def muat_evaluasi():
    """Pembagian data identik dengan notebook
    (test_size=0.2, random_state=21, stratify)."""
    model = muat_model()

    df = pd.read_csv(FILE_DATASET, encoding="utf-8-sig")
    df["Label"] = df["Label"].astype(int)

    idx_latih, idx_uji = train_test_split(
        df.index.to_numpy(),
        test_size=UKURAN_UJI,
        random_state=RANDOM_STATE,
        stratify=df["Label"],
    )

    df["Prediksi"] = model.predict(df[FITUR])
    df["Keyakinan"] = model.predict_proba(df[FITUR]).max(axis=1)
    df["Subset"] = "Latih"
    df.loc[idx_uji, "Subset"] = "Uji"

    uji = df[df["Subset"] == "Uji"]
    urutan = list(NAMA_KELAS)
    nama = list(NAMA_KELAS.values())

    laporan = pd.DataFrame(
        classification_report(
            uji["Label"], uji["Prediksi"],
            labels=urutan, target_names=nama,
            output_dict=True, zero_division=0,
        )
    ).T.round(3)

    metrik = {
        "akurasi": accuracy_score(uji["Label"], uji["Prediksi"]),
        "f1_macro": f1_score(
            uji["Label"], uji["Prediksi"],
            labels=urutan, average="macro", zero_division=0,
        ),
        "cm": confusion_matrix(uji["Label"], uji["Prediksi"], labels=urutan),
        "laporan": laporan,
        "jumlah_latih": len(idx_latih),
        "jumlah_uji": len(idx_uji),
    }
    return df, metrik


@st.cache_data(show_spinner="Menghitung luas tiap kelas...")
def hitung_statistik():
    data = muat_raster()
    larik = data["larik"]
    kiri, bawah, kanan, atas = data["batas"]
    res_x, res_y = data["resolusi"]

    # Luas piksel (km2) dikoreksi terhadap lintang tiap baris
    lintang = atas - (np.arange(larik.shape[0]) + 0.5) * res_y
    luas_piksel = (
        (res_x * 111.32 * np.cos(np.radians(lintang)))
        * (res_y * 110.57)
    )

    total = int((larik != NODATA).sum())
    baris = []
    for kode, nama in NAMA_KELAS.items():
        per_baris = (larik == kode).sum(axis=1)
        jumlah = int(per_baris.sum())
        baris.append({
            "Label": kode,
            "Kelas": nama,
            "Jumlah Piksel": jumlah,
            "Luas (km²)": round(float((per_baris * luas_piksel).sum()), 1),
            "Persentase (%)": round(jumlah / total * 100, 2),
        })
    return pd.DataFrame(baris)


@st.cache_data(show_spinner="Menyiapkan lapisan peta...")
def buat_overlay(kelas_aktif):
    """Ubah raster kelas menjadi PNG berwarna (RGBA) untuk ditempel ke peta."""
    larik = muat_raster()["larik"]

    lut = np.zeros((256, 4), dtype=np.uint8)  # default: transparan
    for kode in kelas_aktif:
        r, g, b = hex_ke_rgb(WARNA[kode])
        lut[kode] = (r, g, b, 255)

    rgba = lut[larik]
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.fromarray(rgba).save(path, optimize=True)
    return path


# ============================================================
# 4. FUNGSI PEMBUAT PETA DAN GRAFIK
# ============================================================


def tambah_legenda(peta):
    item = "".join(
        f'<div style="margin:3px 0;"><span style="display:inline-block;'
        f'width:14px;height:14px;background:{WARNA[k]};border:1px solid #555;'
        f'margin-right:8px;vertical-align:middle;"></span>{nama}</div>'
        for k, nama in NAMA_KELAS.items()
    )
    html = (
        '<div style="position:fixed;bottom:24px;right:24px;z-index:9999;'
        'background:white;padding:10px 14px;border-radius:8px;color:#222;'
        'box-shadow:0 1px 6px rgba(0,0,0,.3);font-size:13px;">'
        f'<b>Legenda Tutupan Lahan</b>{item}</div>'
    )
    peta.get_root().html.add_child(folium.Element(html))


def tambah_dasar_peta(peta, esri_dulu=False):
    osm = folium.TileLayer("OpenStreetMap", name="OpenStreetMap")
    esri = folium.TileLayer(
        tiles=URL_ESRI, attr="Esri World Imagery", name="Citra Satelit (Esri)"
    )
    for lapisan in ([esri, osm] if esri_dulu else [osm, esri]):
        lapisan.add_to(peta)


def isi_popup(r):
    nama_asli = NAMA_KELAS[r.Label]
    nama_pred = NAMA_KELAS[r.Prediksi]
    status = "✅ Tepat" if r.Label == r.Prediksi else "❌ Keliru"
    return folium.Popup(
        f"<b>Kelas asli: {nama_asli}</b> ({r.Subset})<br>"
        f"Hasil prediksi: <b>{nama_pred}</b> {status}<br>"
        f"Tingkat keyakinan: {r.Keyakinan * 100:.1f}%<br>"
        f"NDVI: {r.NDVI:.3f} | NDWI: {r.NDWI:.3f}",
        max_width=260,
    )


def buat_peta_sampel(df):
    peta = folium.Map(
        location=[df["Latitude"].mean(), df["Longitude"].mean()],
        zoom_start=8, tiles=None, control_scale=True,
    )
    tambah_dasar_peta(peta)

    for kode, nama in NAMA_KELAS.items():
        sub = df[df["Label"] == kode]
        grup = folium.FeatureGroup(name=f"{nama} - {len(sub)} sampel")
        for r in sub.itertuples():
            folium.CircleMarker(
                [r.Latitude, r.Longitude], radius=5,
                color="#333333", weight=1,
                fill=True, fill_color=WARNA[kode], fill_opacity=0.9,
                tooltip=f"{nama} ({r.Subset})", popup=isi_popup(r),
            ).add_to(grup)
        grup.add_to(peta)

    uji = df[df["Subset"] == "Uji"]
    grup_uji = folium.FeatureGroup(name=f"Sorot data uji - {len(uji)} titik")
    for r in uji.itertuples():
        folium.CircleMarker(
            [r.Latitude, r.Longitude], radius=9,
            color="#1D4ED8", weight=3, fill=False,
        ).add_to(grup_uji)
    grup_uji.add_to(peta)

    salah = uji[uji["Label"] != uji["Prediksi"]]
    grup_salah = folium.FeatureGroup(name=f"Prediksi keliru - {len(salah)} titik")
    for r in salah.itertuples():
        folium.Marker(
            [r.Latitude, r.Longitude],
            icon=folium.Icon(color="red", icon="exclamation", prefix="fa"),
            tooltip=f"{NAMA_KELAS[r.Label]} → {NAMA_KELAS[r.Prediksi]}",
            popup=isi_popup(r),
        ).add_to(grup_salah)
    grup_salah.add_to(peta)

    peta.fit_bounds([
        [df["Latitude"].min(), df["Longitude"].min()],
        [df["Latitude"].max(), df["Longitude"].max()],
    ])
    folium.LayerControl(collapsed=False).add_to(peta)
    tambah_legenda(peta)
    return peta


def buat_peta_regional(kelas_aktif, opasitas):
    kiri, bawah, kanan, atas = muat_raster()["batas"]

    peta = folium.Map(
        location=[(atas + bawah) / 2, (kiri + kanan) / 2],
        zoom_start=8, tiles=None, control_scale=True,
    )
    tambah_dasar_peta(peta, esri_dulu=True)

    kunci = tuple(sorted(kelas_aktif))
    path_png = buat_overlay(kunci)
    if not os.path.exists(path_png):  # jaga-jaga file sementara terhapus
        buat_overlay.clear()
        path_png = buat_overlay(kunci)

    folium.raster_layers.ImageOverlay(
        image=path_png,
        bounds=[[bawah, kiri], [atas, kanan]],
        opacity=opasitas,
        name="Hasil klasifikasi RF",
    ).add_to(peta)

    peta.fit_bounds([[bawah, kiri], [atas, kanan]])
    folium.LayerControl(collapsed=False).add_to(peta)
    tambah_legenda(peta)
    return peta


def gambar_confusion(cm):
    nama = list(NAMA_KELAS.values())
    n = len(nama)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(nama, rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(nama)
    ax.set_xlabel("Hasil prediksi")
    ax.set_ylabel("Kelas sebenarnya")
    ax.set_title("Matriks konfusi (data uji)")

    batas = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > batas else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def gambar_fitur(model):
    pentingnya = pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).sort_values()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(pentingnya.index, pentingnya.values, color="#0f766e")
    ax.set_xlabel("Nilai importance (Gini)")
    ax.set_title("Band & indeks paling berpengaruh")
    fig.tight_layout()
    return fig


def gambar_distribusi(df_stat):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    warna = [WARNA[k] for k in df_stat["Label"]]
    batang = ax.bar(df_stat["Kelas"], df_stat["Persentase (%)"], color=warna,
                    edgecolor="#444444")
    ax.set_ylabel("Persentase piksel (%)")
    ax.set_title("Komposisi tutupan lahan Jawa Timur")
    for b, p in zip(batang, df_stat["Persentase (%)"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{p:.2f}%", ha="center", fontsize=9)
    ax.set_ylim(0, max(df_stat["Persentase (%)"].max() * 1.15, 1))
    fig.tight_layout()
    return fig


def tampilkan_peta(peta, tinggi):
    """Tampilkan peta folium sebagai HTML biasa (tanpa komponen yang butuh PyArrow)."""
    components.html(peta.get_root().render(), height=tinggi, scrolling=False)


def tampilkan_tabel(tabel, index=False):
    """Tampilkan tabel sebagai HTML (tabel bawaan Streamlit butuh PyArrow)."""
    html = tabel.to_html(index=index, border=0, classes="tabel-data")
    st.markdown(html, unsafe_allow_html=True)


def kotak_metrik(label, nilai, tag, catatan, penjelasan, peringatan=False):
    kelas_box = "stat-box merah" if peringatan else "stat-box"
    kelas_tag = "tag warn" if peringatan else "tag"
    return (
        f'<div class="{kelas_box}"><div class="lbl">{label}</div>'
        f'<div class="val">{nilai}</div>'
        f'<span class="{kelas_tag}">{tag}</span>'
        f'<div class="note">{catatan}</div>'
        f'<div class="exp">{penjelasan}</div></div>'
    )


def kotak_baca(teks):
    st.markdown(f'<div class="kotak-baca">{teks}</div>', unsafe_allow_html=True)


def kesalahan_terbanyak(cm):
    terbaik = None
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                if terbaik is None or cm[i, j] > terbaik[0]:
                    terbaik = (int(cm[i, j]), i, j)
    if terbaik is None:
        return "Tidak ada kekeliruan pada data uji"
    jumlah, i, j = terbaik
    return f"Paling sering: {NAMA_KELAS[i]} → {NAMA_KELAS[j]} ({jumlah}x)"


# ============================================================
# 5. MUAT DATA
# ============================================================

model = muat_model()
df, metrik = muat_evaluasi()
df_stat = hitung_statistik()

uji = df[df["Subset"] == "Uji"]
jumlah_salah = int((uji["Label"] != uji["Prediksi"]).sum())
jumlah_benar = len(uji) - jumlah_salah

# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛰️ Pemetaan LULC")
    st.markdown(
        "**Algoritma:** Random Forest  \n"
        "**Citra:** Sentinel-2A MSI L2A  \n"
        "**Sumber:** Copernicus Data Space  \n"
        "**Waktu:** Jan - Sep 2026 (median)"
    )
    st.divider()

    st.markdown("##### Daftar kelas")
    for kode, nama in NAMA_KELAS.items():
        st.markdown(
            f'<span style="color:{WARNA[kode]};font-size:1.2rem;">■</span> '
            f"**{nama}** "
            f'<span style="color:gray;">- {KETERANGAN_KELAS[kode]}</span>',
            unsafe_allow_html=True,
        )
    st.divider()

    st.markdown("##### Opsi tampilan peta")
    mode_peta = st.radio(
        "Susunan peta:",
        MODE_PETA,
        help="Peta titik sampel menampilkan data latih/uji dan prediksi yang keliru. "
             "Peta wilayah menampilkan hasil klasifikasi seluruh piksel Jawa Timur.",
    )
    kelas_tampil = st.multiselect(
        "Kelas pada peta wilayah",
        options=list(NAMA_KELAS),
        default=list(NAMA_KELAS),
        format_func=lambda k: NAMA_KELAS[k],
    )
    opasitas = st.slider("Opasitas lapisan klasifikasi", 0.2, 1.0, 0.8, 0.05)

# ============================================================
# 7. HEADER: PANEL INTRO (KIRI) + 4 METRIK GRID 2x2 (KANAN)
# ============================================================

st.title("🛰️ Peta Tutupan Lahan Provinsi Jawa Timur")

st.markdown(
    '<div class="panel-intro">'
    "<h3>Tentang aplikasi</h3>"
    "<p>Aplikasi ini menampilkan hasil klasifikasi tutupan lahan "
    "Jawa Timur menjadi 6 kelas (sawah, bangunan, hutan, mangrove, danau, "
    "lautan). Klasifikasi dilakukan dengan model Random Forest pada citra "
    "Sentinel-2A, memakai band B02, B03, B04, B08, B11 serta indeks NDVI "
    "dan NDWI. Empat kotak di bawah ini meringkas seberapa baik model bekerja."
    "</p></div>",
    unsafe_allow_html=True,
)
st.write("")

m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    st.markdown(
        kotak_metrik(
            "AKURASI",
            f"{metrik['akurasi'] * 100:.1f}%",
            f"{jumlah_benar} dari {len(uji)} benar",
            "Ketepatan pada data uji",
            "Persentase titik uji yang kelasnya ditebak dengan benar oleh model. "
            "Makin mendekati 100%, makin baik.",
        ),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        kotak_metrik(
            "F1 MACRO",
            f"{metrik['f1_macro']:.3f}",
            "skala 0 - 1",
            "Rata-rata F1 dari 6 kelas",
            "Gabungan precision dan recall, dirata-rata sama rata untuk tiap kelas, "
            "sehingga kelas yang sampelnya sedikit tetap diperhitungkan.",
        ),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        kotak_metrik(
            "JUMLAH SAMPEL",
            f"{len(df)}",
            "titik referensi",
            f"{metrik['jumlah_latih']} latih • {metrik['jumlah_uji']} uji",
            "Titik berlabel dengan nilai band Sentinel-2. Sebagian dipakai melatih "
            "model, sisanya disimpan untuk menguji model.",
        ),
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        kotak_metrik(
            "PREDIKSI KELIRU",
            f"{jumlah_salah}",
            f"{jumlah_salah / len(uji) * 100:.1f}% error",
            kesalahan_terbanyak(metrik["cm"]),
            "Titik uji yang hasil prediksinya berbeda dari kelas sebenarnya. "
            "Pasangan kelas yang sering tertukar ditunjukkan di atas.",
            peringatan=True,
        ),
        unsafe_allow_html=True,
    )

st.write("")

# ============================================================
# 8. NAVIGASI HALAMAN
# ============================================================

halaman = st.radio(
    "Menu", HALAMAN, horizontal=True, label_visibility="collapsed"
)
st.divider()


def blok_peta_sampel(tinggi):
    st.subheader("Titik Sampel & Hasil Uji")
    st.caption(
        "Satu titik = satu sampel. Lingkaran biru menandai data uji, "
        "ikon seru merah menandai data uji yang prediksinya keliru. "
        "Klik titik untuk melihat NDVI, NDWI, dan keyakinan model."
    )
    tampilkan_peta(buat_peta_sampel(df), tinggi)


def blok_peta_regional(tinggi):
    st.subheader("Klasifikasi Seluruh Wilayah")
    st.caption(
        "Prediksi Random Forest untuk tiap piksel (±111 m), "
        "sudah melalui koreksi kelas perairan (Danau vs Lautan)."
    )
    if not kelas_tampil:
        st.warning("Pilih minimal satu kelas pada sidebar agar peta tampil.")
        return
    tampilkan_peta(buat_peta_regional(kelas_tampil, opasitas), tinggi)


# ------------------------------------------------------------
# Halaman 1: Peta
# ------------------------------------------------------------
if halaman == HALAMAN[0]:
    kotak_baca(
        "<b>Cara membaca peta.</b> "
        "<b>Peta wilayah</b> mewarnai setiap piksel Jawa Timur sesuai kelas hasil "
        "prediksi model. <b>Peta titik sampel</b> menampilkan titik referensi yang "
        "dipakai melatih dan menguji model. Ubah susunan peta, kelas yang tampil, "
        "dan opasitas lewat sidebar di kiri. Panel di pojok peta dapat dipakai "
        "untuk menyalakan atau mematikan lapisan."
    )

    tinggi_peta = 480 if mode_peta == MODE_PETA[1] else 580

    if mode_peta == MODE_PETA[1]:
        kiri_k, kanan_k = st.columns(2, gap="medium")
        with kiri_k:
            blok_peta_regional(tinggi_peta)
        with kanan_k:
            blok_peta_sampel(tinggi_peta)
    else:
        if mode_peta in (MODE_PETA[0], MODE_PETA[3]):
            blok_peta_regional(tinggi_peta)
        if mode_peta in (MODE_PETA[0], MODE_PETA[2]):
            blok_peta_sampel(tinggi_peta)

# ------------------------------------------------------------
# Halaman 2: Luas Lahan
# ------------------------------------------------------------
elif halaman == HALAMAN[1]:
    st.subheader("Luas & Proporsi Tiap Kelas")

    terluas = df_stat.loc[df_stat["Luas (km²)"].idxmax()]
    total_km2 = df_stat["Luas (km²)"].sum()
    kotak_baca(
        "<b>Cara membaca.</b> Grafik dan tabel dihitung dari seluruh piksel valid "
        "pada raster hasil klasifikasi. <i>Jumlah piksel</i> adalah banyaknya piksel "
        "per kelas, <i>luas</i> adalah konversinya ke km² (sudah dikoreksi terhadap "
        "lintang), dan <i>persentase</i> adalah porsinya terhadap seluruh area. "
        f"Kelas terluas adalah <b>{terluas['Kelas']}</b> "
        f"({terluas['Persentase (%)']:.2f}% atau sekitar {terluas['Luas (km²)']:,.0f} km²), "
        f"dari total area terklasifikasi sekitar {total_km2:,.0f} km²."
    )

    kol_grafik, kol_tabel = st.columns(2, gap="large")
    with kol_grafik:
        st.markdown("**Komposisi tutupan lahan**")
        st.caption("Tinggi batang = persentase piksel tiap kelas.")
        fig_dist = gambar_distribusi(df_stat)
        st.pyplot(fig_dist)
        plt.close(fig_dist)
    with kol_tabel:
        st.markdown("**Rincian angka**")
        st.caption("Nilai pasti yang menjadi dasar grafik di sebelah kiri.")
        tampilkan_tabel(df_stat.drop(columns="Label"))

# ------------------------------------------------------------
# Halaman 3: Kinerja Model
# ------------------------------------------------------------
else:
    st.subheader("Seberapa Baik Modelnya?")

    fitur_utama = pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).idxmax()
    kotak_baca(
        "<b>Cara membaca.</b> Model diuji pada data yang tidak dipakai saat pelatihan. "
        "Pada <i>matriks konfusi</i>, angka di diagonal adalah prediksi yang benar, "
        "angka di luar diagonal adalah prediksi yang tertukar. Grafik <i>pengaruh fitur</i> "
        f"menunjukkan band atau indeks yang paling menentukan, yaitu <b>{fitur_utama}</b>."
    )

    kol_cm, kol_fitur = st.columns(2, gap="large")
    with kol_cm:
        st.markdown(f"**Matriks konfusi ({len(uji)} titik uji)**")
        st.caption("Baris = kelas sebenarnya, kolom = hasil prediksi.")
        fig_cm = gambar_confusion(metrik["cm"])
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    with kol_fitur:
        st.markdown("**Pengaruh tiap fitur**")
        st.caption("Semakin panjang batang, semakin besar peran fitur itu.")
        fig_fitur = gambar_fitur(model)
        st.pyplot(fig_fitur)
        plt.close(fig_fitur)

    st.markdown("**Laporan klasifikasi per kelas (data uji)**")
    kotak_baca(
        "<b>Precision</b>: dari semua titik yang diprediksi sebagai kelas ini, "
        "berapa yang benar. <b>Recall</b>: dari semua titik yang sebenarnya kelas ini, "
        "berapa yang berhasil ditemukan. <b>F1-score</b>: gabungan keduanya. "
        "<b>Support</b>: jumlah titik uji pada kelas tersebut."
    )
    tampilkan_tabel(metrik["laporan"], index=True)

st.divider()
st.caption(
    "Sumber data: Sentinel-2A L2A (Copernicus Data Space), komposit median "
    "01 Jan - 01 Sep 2026, awan maks. 10%, resolusi 0,001° (±111 m). "
    "Fitur model: B02, B03, B04, B08, B11, NDVI, NDWI."
)