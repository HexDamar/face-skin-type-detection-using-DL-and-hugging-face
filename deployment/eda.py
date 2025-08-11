import streamlit as st
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from PIL import Image
from collections import Counter

# Use st.cache_data to cache the DataFrame loading,
# as this function reads many files and can be slow.
@st.cache_data
def load_and_process_data():
    main_path = 'Oily-Dry-Skin-Types' # Main path defined once

    # Check if the main_path exists
    if not os.path.exists(main_path):
        st.error(f"Error: Direktori '{main_path}' tidak ditemukan.")
        st.write("Pastikan folder 'Oily-Dry-Skin-Types' berada di direktori yang sama dengan aplikasi Streamlit Anda, atau perbarui `main_path` ke lokasi yang benar.")
        st.stop() # Stop the app if the directory is not found

    train_path = os.path.join(main_path, 'train')
    test_path = os.path.join(main_path, 'test')
    valid_path = os.path.join(main_path, 'valid')

    # Function to get image paths, class names, and image shapes
    def get_image_data_from_dir(base_dir):
        image_paths = []
        class_names = []
        image_shapes = [] # New list to store [height, width]

        if os.path.exists(base_dir):
            for class_name in os.listdir(base_dir):
                class_dir = os.path.join(base_dir, class_name)
                if os.path.isdir(class_dir):
                    for image_file in os.listdir(class_dir):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                            full_image_path = os.path.join(class_dir, image_file)
                            image_paths.append(full_image_path)
                            class_names.append(class_name)

                            # --- Read image and get its shape ---
                            try:
                                with Image.open(full_image_path) as img:
                                    width, height = img.size
                                    image_shapes.append([height, width]) # Store as [height, width]
                            except Exception as e:
                                # We'll just append [0,0] for unreadable images and let plotting handle it.
                                # Streamlit warnings could spam the console if many images fail.
                                image_shapes.append([0, 0])
        return image_paths, class_names, image_shapes

    # Collect data from all splits
    all_image_paths = []
    all_class_names = []
    all_image_shapes = []

    # Get data from training set
    train_image_paths, train_class_names, train_image_shapes = get_image_data_from_dir(train_path)
    all_image_paths.extend(train_image_paths)
    all_class_names.extend(train_class_names)
    all_image_shapes.extend(train_image_shapes)

    # Get data from test set
    test_image_paths, test_class_names, test_image_shapes = get_image_data_from_dir(test_path)
    all_image_paths.extend(test_image_paths)
    all_class_names.extend(test_class_names)
    all_image_shapes.extend(test_image_shapes)

    # Get data from validation set
    valid_image_paths, valid_class_names, valid_image_shapes = get_image_data_from_dir(valid_path)
    all_image_paths.extend(valid_image_paths)
    all_class_names.extend(valid_class_names)
    all_image_shapes.extend(valid_image_shapes)

    if not all_image_paths:
        st.warning("Tidak ada gambar ditemukan di direktori yang ditentukan. Silakan periksa struktur folder dan `main_path` Anda.")
        st.stop()

    # Create the DataFrame
    df = pd.DataFrame({
        'Image_Path': all_image_paths,
        'Class_Name': all_class_names,
        'Shapes': all_image_shapes
    })

    # Ensure 'Width' and 'Height' columns exist from 'Shapes'
    df['Height'] = df['Shapes'].apply(lambda x: x[0])
    df['Width'] = df['Shapes'].apply(lambda x: x[1])

    # Return the DataFrame AND the paths if needed globally
    return df, train_path, test_path, valid_path # Return paths as well

def run_eda():
    st.title('Eksplorasi Data Analysis ( EDA )')

    # Load data and paths using the cached function
    df, train_path, test_path, valid_path = load_and_process_data() # Unpack the returned values

    # --- Display Raw Data (optional, for debugging or overview) ---
    if st.checkbox('Tampilkan Data Mentah DataFrame'):
        st.dataframe(df)

    
    ## 1. Distribusi Gambar per Kelas
    # =======================================================================================================================
    # EDA 1: Distribusi Kelas
    st.markdown("""
                # 1. Distribusi Class pada Dataset
    """)
    # Calculate the number of images per class
    class_counts = df['Class_Name'].value_counts().sort_index()

    if class_counts.empty:
        st.warning("Tidak ada data kelas yang ditemukan untuk diplot.")
    else:
        # Create the pie chart
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
        ax_pie.set_title('Distribusi Gambar per Kelas', fontsize=16)
        ax_pie.axis('equal')  # Ensures the pie chart is perfectly round
        st.pyplot(fig_pie)
        plt.close(fig_pie) # Close figure to free memory

        # Class Label Proportions
        st.subheader("Proporsi Label Kelas:")
        label_proportions = pd.DataFrame(df['Class_Name'].value_counts(normalize=True))
        label_proportions.columns = ['Proporsi']
        st.dataframe(label_proportions)
    st.markdown("""
                 insight: 

Dari pie chart dan tabel proporsi, kita mendapatkan insight kunci sebagai berikut:

- Mayoritas Data adalah Kelas 'Normal': Sebanyak 40.4% dari gambar berada dalam kategori 'normal'. Ini menunjukkan bahwa dataset Anda memiliki banyak contoh untuk kelas ini, yang umumnya bagus untuk melatih model, karena model akan memiliki banyak variasi untuk dipelajari.

- Kelas 'Oily' Cukup Signifikan: Dengan 35.5% dari data, kelas 'oily' juga memiliki representasi yang kuat. Ini mendekati jumlah data 'normal', yang berarti model juga akan memiliki cukup banyak contoh untuk belajar mengenali karakteristik 'oily'.

- Kelas 'Dry' adalah Kelas Minoritas yang Jelas: Hanya 24.0% dari data termasuk dalam kategori 'dry'. Ini adalah kelas minoritas yang signifikan dibandingkan dengan 'normal' dan 'oily'.

Implikasi dari Insight:

- Potensi Bias Model: Ketidakseimbangan ini sangat mungkin menyebabkan model pembelajaran mesin menjadi bias terhadap kelas mayoritas ('normal' dan 'oily'). Model mungkin akan belajar untuk memprediksi 'normal' atau 'oily' dengan akurasi tinggi hanya karena mereka lebih sering muncul, sementara kinerja prediksi untuk kelas 'dry' bisa jadi buruk.

- Akurasi Menipu: Jika Anda hanya melihat akurasi keseluruhan (overall accuracy), mungkin terlihat tinggi (misalnya, 80-90%), tetapi ini bisa menipu. Akurasi tinggi ini mungkin hanya mencerminkan kemampuan model untuk memprediksi kelas mayoritas dengan baik, sementara ia gagal total dalam mengidentifikasi kelas 'dry'.

Cara Penanggulangan Ketidakseimbangan Kelas (Class Imbalance)

Ada beberapa strategi yang dapat diterapkan untuk mengatasi masalah ketidakseimbangan kelas ini. Pilihan terbaik seringkali bergantung pada ukuran dataset, jenis masalah, dan sumber daya komputasi.

1. Resampling Teknik (Data-level Approaches)
2. Cost-Sensitive Learning (Algorithm-level Approaches)
3. Ensemble Methods
    """)
    
    ## 2. Visualisasi Contoh Gambar per Kelas
    # =======================================================================================================================
    # EDA 2: Visualisasi Contoh Gambar per Kelas
    st.markdown("""
                # 2. Tampilkan Gambar dari masing masing Class
    """)
    def plot_sample_images(path, num_images_per_class=5):
        if not os.path.exists(path):
            st.warning(f"Path tidak ditemukan: {path}")
            return

        labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        if not labels:
            st.warning(f"Tidak ada direktori kelas ditemukan di: {path}")
            return

        for label in labels:
            folder_path = os.path.join(path, label)
            st.subheader(f"Kelas: {label}")

            images_in_class = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            images_in_class.sort() # Sort images to ensure consistent display

            if not images_in_class:
                st.write(f"Tidak ada gambar ditemukan di kelas '{label}'.")
                continue

            # Create a Matplotlib figure to hold the images
            fig_samples, axes_samples = plt.subplots(1, min(num_images_per_class, len(images_in_class)), figsize=(15, 3))
            
            # Ensure axes_samples is always an array
            if min(num_images_per_class, len(images_in_class)) == 1:
                axes_samples = np.array([axes_samples])


            for i in range(min(num_images_per_class, len(images_in_class))):
                image_name = images_in_class[i]
                image_path = os.path.join(folder_path, image_name)
                try:
                    img = Image.open(image_path)
                    axes_samples[i].imshow(img)
                    axes_samples[i].axis("off")
                    axes_samples[i].set_title(f"Gambar {i+1}", fontsize=10)
                except Exception as e:
                    axes_samples[i].text(0.5, 0.5, "Error memuat gambar", horizontalalignment='center', verticalalignment='center', transform=axes_samples[i].transAxes, color='red')
                    axes_samples[i].axis("off")
                    st.warning(f"Tidak dapat memuat gambar {image_name} dari {label}: {e}")

            st.pyplot(fig_samples)
            plt.close(fig_samples) # Close the figure to free up memory
    plot_sample_images(train_path, num_images_per_class=5) # Pass train_path here

    st.markdown("""
                 insight: 

Variasi dalam Setiap Kelas:

- Class: dry: Gambar-gambar menunjukkan subjek dengan kulit kering. Ada variasi dalam pose, pencahayaan, dan latar belakang. Beberapa gambar juga menunjukkan cropping atau data augmentation seperti rotasi atau shifting (misalnya, terlihat pada gambar kedua dan keempat dari kiri di baris 'dry' yang terpotong secara horizontal).

- Class: normal: Gambar-gambar untuk kelas 'normal' juga menunjukkan variasi yang serupa dalam pose dan cropping.

- Class: oily: Begitu pula, kelas 'oily' menampilkan variasi pada individu dan posisi wajah.

Kualitas dan Karakteristik Gambar:

- Gambar-gambar terlihat seperti hasil dari proses preprocessing atau data augmentation (terutama yang terpotong atau sedikit miring). Ini menunjukkan bahwa dataset mungkin telah melalui tahap persiapan untuk meningkatkan variasi dan generalisasi model.

- Ada blok hitam horizontal yang muncul di beberapa gambar, yang bisa jadi merupakan artefak dari data augmentation (misalnya, rotasi yang membuat bagian kosong diisi dengan warna hitam) atau cropping yang tidak sempurna
    """)
    
    ## 3. Distribusi Resolusi Gambar
    # =======================================================================================================================
    # EDA 3: Distribusi Resolusi Gambar
    st.markdown("""
                # 3. Distribusi Resolusi Gambar Dari Masing masing Class
    """)
    # Prepare subplots for each class
    classes = sorted(df['Class_Name'].unique())

    if not classes:
        st.warning("Tidak ada kelas ditemukan dalam data untuk menampilkan resolusi gambar.")
    else:
        fig_res, axes_res = plt.subplots(1, len(classes), figsize=(5 * len(classes), 5), sharey=True)

        # Ensure 'axes_res' is an array if there's only one class
        if len(classes) == 1:
            axes_res = [axes_res]

        for i, class_name in enumerate(classes):
            class_df = df[df['Class_Name'] == class_name]

            # Calculate Width & Height counts
            # Get the count for 640x640, or 0 if not present
            width_count = class_df['Width'].value_counts().get(640, 0)
            height_count = class_df['Height'].value_counts().get(640, 0)

            # Plot as bar chart
            axes_res[i].bar(['Width'], [width_count], color='blue', label='Width', alpha=0.7)
            axes_res[i].bar(['Height'], [height_count], color='green', label='Height', alpha=0.7)

            axes_res[i].set_title(f'Resolusi Gambar - {class_name}')
            axes_res[i].set_ylabel('Jumlah Gambar')
            axes_res[i].legend()

        plt.suptitle('Distribusi Resolusi Gambar (Diasumsikan 640x640)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

        st.pyplot(fig_res)
        plt.close(fig_res) # Close figure to free memory
    st.markdown("""
                 Insight 
1. Resolusi Gambar Seragam:

- Grafik batang untuk "Resolusi Gambar - dry", "Resolusi Gambar - normal", dan "Resolusi Gambar - oily" menunjukkan bahwa baik width (lebar) maupun height (tinggi) gambar memiliki nilai yang konsisten dan tinggi (di sekitar 700-800 unit pada sumbu Y untuk dry, dan di atas 1000 unit untuk normal dan oily).

- Tabel ringkasan df_summary secara eksplisit mengkonfirmasi ini: semua gambar memiliki resolusi 640x640.

2. Jumlah Gambar per Kelas (diperkuat oleh Resolusi):

- Meskipun grafik batang tidak menunjukkan jumlah gambar secara langsung, tinggi bar menunjukkan bahwa kelas 'normal' dan 'oily' memiliki jumlah gambar yang jauh lebih banyak daripada kelas 'dry'. Ini konsisten dengan insight dari pie chart sebelumnya.

- Tabel df_summary memberikan angka pasti:

        - Normal: 1274 gambar

        - Oily: 1120 gambar

        - Dry: 758 gambar

Implikasi
1. Preprocessing yang Konsisten: Fakta bahwa semua gambar memiliki resolusi 640x640 adalah implikasi positif yang sangat kuat. Ini menunjukkan bahwa:

- Data telah di-resize secara konsisten: Ini adalah langkah preprocessing yang sangat baik dan krusial dalam computer vision. Model pembelajaran mesin (terutama Convolutional Neural Networks / CNNs) memerlukan input gambar dengan ukuran yang seragam.

- Mempermudah Pemodelan: Tidak perlu lagi melakukan resize dinamis selama pelatihan, yang dapat menghemat waktu dan sumber daya komputasi.

- Kompatibilitas Model: Resolusi 640x640 adalah ukuran yang cukup umum dan baik untuk banyak arsitektur CNN modern, memungkinkan model untuk menangkap detail yang relevan tanpa terlalu besar (yang bisa boros komputasi) atau terlalu kecil (yang bisa menghilangkan informasi).
    """)

    
    ## 4. Distribusi RGB dan Contoh Gambar per Kelas
    # =======================================================================================================================
    # EDA 4: Distribusi RGB dan Contoh Gambar per Kelas
    st.markdown("""
                # 4. Distribusi RGB
    """)
    # Re-use 'classes' variable from EDA 3
    # Ensure there are enough classes to plot as 3 columns
    if len(df['Class_Name'].unique()) < 3:
        st.warning("Ditemukan kurang dari 3 kelas unik. Hanya kelas yang tersedia yang akan ditampilkan.")
        classes_for_rgb = sorted(df['Class_Name'].unique())
        num_cols_rgb = len(classes_for_rgb)
    else:
        classes_for_rgb = sorted(df['Class_Name'].unique())[:3] # Take up to 3 classes for plotting
        num_cols_rgb = 3

    if not classes_for_rgb:
        st.warning("Tidak ada kelas ditemukan dalam data untuk menampilkan distribusi RGB.")
    else:
        # Prepare the figure for RGB plots
        fig_rgb, axes_rgb = plt.subplots(2, num_cols_rgb, figsize=(5 * num_cols_rgb, 8))

        # Ensure axes_rgb is always a 2D array, even for 1 or 2 columns
        if num_cols_rgb == 1:
            axes_rgb = np.array([[axes_rgb[0]], [axes_rgb[1]]])
        elif num_cols_rgb == 2:
            temp_axes = np.empty((2, 3), dtype=object)
            for r in range(2):
                for c in range(2):
                    temp_axes[r, c] = axes_rgb[r, c]
            axes_rgb = temp_axes
        # If num_cols_rgb is 3, axes_rgb will already be a 2x3 array as desired


        # Loop each class for RGB distribution
        for i, class_name in enumerate(classes_for_rgb):
            class_df = df[df['Class_Name'] == class_name]
            if not class_df.empty:
                # Get a random row from df based on class
                sample_row = class_df.sample(1, random_state=44).iloc[0]
                img_path = sample_row['Image_Path'] # Use 'Image_Path'

                # Open image
                try:
                    img = Image.open(img_path).convert('RGB')  # Ensure RGB
                    img_np = np.array(img)

                    # Display image
                    axes_rgb[0, i].imshow(img_np)
                    axes_rgb[0, i].set_title(f'Contoh Gambar - {class_name}')
                    axes_rgb[0, i].axis('off')

                    # Get RGB channels
                    red_channel = img_np[:, :, 0].flatten()
                    green_channel = img_np[:, :, 1].flatten()
                    blue_channel = img_np[:, :, 2].flatten()

                    # Plot RGB histogram
                    axes_rgb[1, i].hist(red_channel, bins=256, color='red', alpha=0.5, label='Red')
                    axes_rgb[1, i].hist(green_channel, bins=256, color='green', alpha=0.5, label='Green')
                    axes_rgb[1, i].hist(blue_channel, bins=256, color='blue', alpha=0.5, label='Blue')

                    axes_rgb[1, i].set_title(f'Distribusi RGB - {class_name}')
                    axes_rgb[1, i].set_xlim(0, 255)
                    axes_rgb[1, i].set_xlabel('Intensitas Piksel')
                    axes_rgb[1, i].set_ylabel('Jumlah Piksel')
                    axes_rgb[1, i].legend()
                except Exception as e:
                    axes_rgb[0, i].text(0.5, 0.5, "Gambar tidak dapat dimuat", horizontalalignment='center', verticalalignment='center', transform=axes_rgb[0, i].transAxes)
                    axes_rgb[0, i].set_title(f'Contoh Gambar - {class_name}')
                    axes_rgb[0, i].axis('off')
                    axes_rgb[1, i].text(0.5, 0.5, f"Error: {e}", horizontalalignment='center', verticalalignment='center', transform=axes_rgb[1, i].transAxes)
                    axes_rgb[1, i].set_title(f'Distribusi RGB - {class_name}')
                    axes_rgb[1, i].axis('off')
            else:
                axes_rgb[0, i].text(0.5, 0.5, "Tidak ada data untuk kelas ini", horizontalalignment='center', verticalalignment='center', transform=axes_rgb[0, i].transAxes)
                axes_rgb[0, i].set_title(f'Contoh Gambar - {class_name}')
                axes_rgb[0, i].axis('off')
                axes_rgb[1, i].axis('off') # Turn off the empty histogram subplot
                axes_rgb[1, i].set_title(f'Distribusi RGB - {class_name}')

        plt.tight_layout()
        st.pyplot(fig_rgb) # Display the plot in Streamlit
        plt.close(fig_rgb) # Close figure to free memory

    ## Analisis Histogram per Kelas

        st.markdown("""
                 ### 1. Kelas 'dry'
- Distribusi intensitas piksel **lebih luas**.
- Terdapat kecenderungan ke arah **nada merah/hangat**.
- Mengindikasikan karakteristik visual dari **kulit kering**.

### 2. Kelas 'normal'
- Didominasi oleh **puncak tinggi pada intensitas rendah (~0)**.
- Hal ini disebabkan oleh **artefak area hitam** dari proses augmentasi.
- Distribusi ini **tidak mencerminkan karakteristik asli kulit normal**.

### 3. Kelas 'oily'
- Memiliki distribusi yang **lebih menyebar**.
- Menunjukkan adanya variasi dalam **kecerahan atau kilau kulit**.
- Cenderung mencerminkan **reflektivitas/kilauan** pada kulit berminyak.

---

## Implikasi Penting

- Terdapat **perbedaan karakteristik warna** yang dapat dikenali antara kelas `'dry'` dan `'oily'`.
- Distribusi warna pada kelas `'normal'` **terganggu oleh artefak**, sehingga kurang dapat diandalkan untuk analisis eksplorasi berbasis warna.
- Informasi dari histogram dapat membantu **model mempelajari ciri visual** yang membedakan masing-masing kelas, selama data bebas dari artefak.
    """)



# Call the EDA function
if __name__ == '__main__':
    run_eda()