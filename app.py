
import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

st.set_page_config(
    page_title="Klasifikasi Tutupan Lahan",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Klasifikasi Tutupan Lahan")
st.subheader("Sentinel-2A dengan Random Forest")
st.write("Hasil klasifikasi tutupan lahan di Jawa Timur.")

nama_file = "hasil_klasifikasi_lulc.tif"

nama_kelas = {
    0: "Sawah",
    1: "Bangunan",
    2: "Hutan",
    3: "Mangrove",
    4: "Danau",
    5: "Lautan"
}

warna = [
    "#FFE600",
    "#DC2828",
    "#228B22",
    "#00643C",
    "#00BFFF",
    "#00008B"
]

tab1, tab2 = st.tabs(["Peta Klasifikasi", "Informasi"])

with tab1:
    st.markdown("### Peta Hasil Klasifikasi")

    try:
        with rasterio.open(nama_file) as src:
            data = src.read(1)

            tinggi, lebar = data.shape
            faktor = max(1, int(max(tinggi, lebar) / 1500))

            data_tampil = data[::faktor, ::faktor]

            data_tampil = np.ma.masked_where(
                (data_tampil < 0) | (data_tampil > 5),
                data_tampil
            )

            cmap = ListedColormap(warna)
            cmap.set_bad("white")

            fig, ax = plt.subplots(figsize=(12, 7))

            ax.imshow(
                data_tampil,
                cmap=cmap,
                vmin=-0.5,
                vmax=5.5,
                interpolation="nearest"
            )

            ax.set_title("Peta Tutupan Lahan")
            ax.axis("off")

            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                label="Download Peta GeoTIFF",
                data=open(nama_file, "rb").read(),
                file_name=nama_file,
                mime="image/tiff"
            )

    except Exception as e:
        st.error("Peta belum dapat ditampilkan.")
        st.write("Pastikan file GeoTIFF berada di folder aplikasi.")
        st.write(str(e))

    st.markdown("### Legenda")

    for nomor, nama in nama_kelas.items():
        st.markdown(
            f'<span style="color:{warna[nomor]};'
            f'font-size:22px;">■</span> {nama}',
            unsafe_allow_html=True
        )

with tab2:
    st.markdown("### Informasi Penelitian")
    st.write("Data citra: Sentinel-2A")
    st.write("Metode klasifikasi: Random Forest")
    st.write("Jumlah kelas: 6")

    st.markdown("### Kelas Tutupan Lahan")

    for nomor, nama in nama_kelas.items():
        st.write(str(nomor) + " - " + nama)
