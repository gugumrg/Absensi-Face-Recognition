import streamlit as st
import cv2
import mysql.connector
from mysql.connector import Error
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import toml


# Read the database configuration from the toml file
def read_db_config():
    try:
        config = toml.load('.streamlit/secret.toml')
        db_config = config.get('database', {})
        #st.write("Database configuration:", db_config)
        return db_config
    except Exception as e:
        st.error(f"Error reading the config file: {e}")
        return None

# Koneksi Database MySQL
def create_connection():
    db_config = read_db_config()
    if db_config is None:
        return None

    try:
        connection = mysql.connector.connect(
            host=db_config.get('host'),
            database=db_config.get('database'),
            user=db_config.get('user'),
            password=db_config.get('password')
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Kesalahan saat menghubungkan ke MySQL: {e}")
    return None

# Autentikasi Pengguna
def authenticate_user(username, password):
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        return user
    return None

# Fungsi Login
def login(username, password):
    conn = create_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = user['username']
            st.session_state['role'] = user['role']
            return True
    return False

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def update_password(old_password, new_password, confirm_password):
    ensure_directory_exists("TrainingImageLabel/")
    path = "TrainingImageLabel/password.txt"

    if os.path.isfile(path):
        with open(path, "r") as file:
            stored_password = file.read()
    else:
        if new_password == confirm_password:
            with open(path, "w") as file:
                file.write(new_password)
            st.success("Password berhasil diubah.")
            return
        else:
            st.error("Password baru tidak cocok dengan konfirmasi.")
            return

    if old_password == stored_password:
        if new_password == confirm_password:
            with open(path, "w") as file:
                file.write(new_password)
            st.success("Password berhasil diperbarui.")
        else:
            st.error("Password baru tidak cocok dengan konfirmasi.")
    else:
        st.error("Password lama yang Anda masukkan tidak valid.")

def check_histogram_file_exists():
    if not os.path.isfile("histogram_frontalface_default.xml"):
        st.error("File histogram tidak ditemukan.")
        return False
    return True

def capture_images(user_id, name):
    if not os.path.isfile("histogram_frontalface_default.xml"):
        st.error("Histogram file not found.")
        return

    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    ensure_directory_exists("Details/")
    ensure_directory_exists("TrainingImage/")
    serial = 1

    if os.path.isfile("Details/Details.csv"):
        with open("Details/Details.csv", 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                serial += 1
    else:
        with open("Details/Details.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(columns)

    if all(x.isalpha() or x.isspace() for x in name):
        detector = cv2.CascadeClassifier("histogram_frontalface_default.xml")
        sample_num = 0
        st.title("Menangkap Gambar")
        st.write("Silakan posisikan diri Anda di depan kamera")

        # Create a placeholder for the video feed
        video_placeholder = st.empty()

        # Open video capture
        cam = cv2.VideoCapture(0)

        start_time = time.time()

        while True:
            ret, img = cam.read()
            if not ret:
                st.error("Failed to access the camera.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sample_num += 1
                cv2.imwrite(f"TrainingImage/{name.replace(' ', '_')}.{serial}.{user_id}.{sample_num}.jpg",
                            gray[y:y + h, x:x + w])
            
            # Convert image for Streamlit display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

            # Stop after 8 seconds
            if time.time() - start_time > 60 or sample_num >= 100:
                break

        cam.release()
        cv2.destroyAllWindows()

        if sample_num > 50:
            result_message = f"Gambar berhasil diambil untuk ID: {user_id}, Nama: {name}, Jumlah Sampel: {sample_num}"
            with open('Details/Details.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([serial, '', user_id, '', name])
            st.success(result_message)
        else:
            st.write(f"Gambar berhasil diambil untuk ID: {user_id}, Nama: {name}, Jumlah Sampel: {sample_num}")

    else:
        st.error("Nama yang dimasukkan tidak valid.")

def train_images(password):
    if not check_histogram_file_exists():
        return

    path = "TrainingImageLabel/password.txt"
    if os.path.isfile(path):
        with open(path, "r") as file:
            stored_password = file.read()
    else:
        st.error("Password belum diatur. Harap atur password terlebih dahulu.")
        return

    if password != stored_password:
        st.error("Password yang Anda masukkan tidak valid.")
        return

    ensure_directory_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    cascade_path = "histogram_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces, ids = get_images_and_labels("TrainingImage")

    try:
        recognizer.train(faces, np.array(ids))
    except Exception as e:
        st.error("Silakan daftarkan seseorang terlebih dahulu.")
        st.error(f"Kesalahan: {e}")
        return

    recognizer.save("TrainingImageLabel/Trainner.yml")
    st.success("Model wajah berhasil dilatih dan disimpan.")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(image_np)
        ids.append(user_id)

    return faces, ids

def track_attendance(selected_status):
    if not check_histogram_file_exists():
        return

    ensure_directory_exists("Kehadiran/")
    ensure_directory_exists("Details/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
        st.error("Harap latih model wajah sebelum mencatat kehadiran.")
        return

    recognizer.read("TrainingImageLabel/Trainner.yml")
    cascade_path = "histogram_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    column_names = ['SERIAL', 'NIDN', 'NAMA LENGKAP', 'TANGGAL', 'JAN', 'STATUS']

    if os.path.isfile("Details/Details.csv"):
        df = pd.read_csv("Details/Details.csv")
    else:
        st.error("Data karyawan tidak ditemukan.")
        return

    today_date = datetime.datetime.now().strftime('%d-%m-%Y')
    attendance_file = f"Kehadiran/Kehadiran_{today_date}.csv"

    if os.path.isfile(attendance_file):
        existing_df = pd.read_csv(attendance_file)
        existing_ids = existing_df['NIDN'].tolist()
        existing_status = existing_df.groupby('NIDN')['STATUS'].apply(list).to_dict()
    else:
        existing_ids = []
        existing_status = {}

    attendance = []
    start_time = time.time()  # Record start time

    st.title("Menandai Kehadiran")
    st.write("Arahkan kamera ke wajah untuk mendeteksi kehadiran.")

    # Create a placeholder for the video feed
    frame_placeholder = st.empty()

    # Start video capture
    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        if not ret:
            st.warning("Gagal membaca video feed dari kamera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            user_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            confidence_percentage = round(100 - confidence * 0.1, 2)
            if confidence < 50:
                timestamp = time.time()
                date = datetime.datetime.fromtimestamp(timestamp).strftime('%d-%m-%Y')
                time_stamp = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                user_info = df[df['SERIAL NO.'] == user_id]
                if not user_info.empty:
                    id_number = user_info['ID'].values[0]
                    id_number = str(id_number).replace(',', '')
                    name = user_info['NAME'].values[0]

                    if id_number in existing_ids:
                        if (selected_status == 'Masuk' and 'Masuk' not in existing_status.get(id_number, [])) or \
                           (selected_status == 'Pulang' and 'Pulang' not in existing_status.get(id_number, [])):
                            attendance.append([len(attendance) + 1, id_number, name, date, time_stamp, selected_status])
                            existing_status[id_number] = existing_status.get(id_number, []) + [selected_status]
                    else:
                        attendance.append([len(attendance) + 1, id_number, name, date, time_stamp, selected_status])
                        existing_ids.append(id_number)
                        existing_status[id_number] = [selected_status]

                    text1 = f"{name[:15]}"
                    text2 = f"{confidence_percentage}%"
                    text_position1 = (x, y - 30)
                    text_position2 = (x, y - 10)  # Position text slightly above the face rectangle, below the first line
                    cv2.putText(img, text1, text_position1, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, text2, text_position2, font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(img, "Unknown", (x, y - 10), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(img, "Unknown", (x, y - 10), font, 1, (255, 255, 255), 2)

        # Convert the frame to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, channels="RGB", caption="Menandai Kehadiran")

        # Check if 8 seconds have passed
        if time.time() - start_time > 8:
            break

    cam.release()

    if attendance:
        if os.path.isfile(attendance_file):
            with open(attendance_file, 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                for entry in attendance:
                    writer.writerow(entry)
        else:
            with open(attendance_file, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(column_names)
                for entry in attendance:
                    writer.writerow(entry)

        st.success(f"Kehadiran berhasil dicatat untuk {len(attendance)} orang.")
    else:
        st.warning("Tidak ada kehadiran yang dicatat.")


def view_attendance():
    st.subheader("Lihat Kehadiran")
    ensure_directory_exists("Kehadiran/")
    date = st.date_input("Pilih Tanggal")
    date_str = date.strftime('%d-%m-%Y')
    attendance_file = f"Kehadiran/Kehadiran_{date_str}.csv"

    if os.path.isfile(attendance_file):
        st.write(f"File ditemukan: {attendance_file}")
        try:
            df = pd.read_csv(attendance_file)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Tampilkan DataFrame
            st.write(df)

            # Pilih baris untuk dihapus
            row_to_delete = st.multiselect("Pilih baris untuk dihapus berdasarkan indeks", df.index)

            if st.button("Hapus Baris"):
                if row_to_delete:
                    df = df.drop(row_to_delete)  # Hapus baris yang dipilih
                    df.to_csv(attendance_file, index=False)  # Simpan perubahan ke file CSV
                    st.success("Baris yang dipilih berhasil dihapus.")
                    st.write(df)  # Tampilkan ulang DataFrame yang sudah diperbarui
                else:
                    st.warning("Tidak ada baris yang dipilih untuk dihapus.")

        except Exception as e:
            st.error(f"Kesalahan saat membaca file: {e}")
    else:
        st.error(f"Rekam kehadiran tidak ditemukan untuk {date_str}.")

def get_current_time():
    return datetime.datetime.now().strftime('%H:%M')

def get_current_date():
    return datetime.datetime.now().strftime('%d-%m-%Y')

# Streamlit Frontend
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 2rem;
        font-weight: 500;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .datetime {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E90FF;
    }
    .date {
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Menu yang tidak memerlukan login
    menu_items_no_login = ["Beranda", "Rekam Presensi", "Lihat Presensi"]

    # Menu yang memerlukan login (untuk admin)
    if st.session_state['logged_in']:
        menu_items = menu_items_no_login + ["Register", "Change Password"]
    else:
        menu_items = menu_items_no_login + ["Login"]

    menu = st.sidebar.selectbox("Pilih Menu", menu_items)

    if menu == "Beranda":
        st.markdown('<div class="centered title">Sistem Kehadiran Berbasis Pengenalan Wajah</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="centered subtitle">Selamat datang di Sistem Kehadiran Berbasis Pengenalan Wajah!</div>',
            unsafe_allow_html=True)

        current_date = get_current_date()
        current_time = get_current_time()

        st.markdown(f'<div class="centered datetime date">{current_date}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="centered datetime">{current_time}</div>', unsafe_allow_html=True)

    elif menu == "Rekam Presensi":
        st.title("Track Attendance")
        selected_status = st.selectbox("Pilih Status", ["Masuk", "Pulang"])
        if st.button("Start Tracking"):
            track_attendance(selected_status)

    elif menu == "Lihat Presensi":
        view_attendance()

    elif menu == "Login":
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.success("Login berhasil!")
            else:
                st.error("Username atau password salah.")

    elif menu == "Register":
        st.subheader("Register")
        user_id = st.text_input("Masukkan NIDN")
        name = st.text_input("Masukkan Nama Lengkap")
        if st.button("Ambil Gambar"):
            capture_images(user_id, name)
        password = st.text_input("Masukkan kata sandi untuk melatih gambar", type="password")
        if st.button("Latih Gambar"):
            train_images(password)

    elif menu == "Change Password":
        if not st.session_state['logged_in']:
            st.warning("Anda harus login untuk mengubah kata sandi.")
        elif st.session_state['role'] == 'admin':
            st.title("Change Password")
            old_password = st.text_input("Password Lama", type="password")
            new_password = st.text_input("Password Baru", type="password")
            confirm_password = st.text_input("Konfirmasi Password Baru", type="password")
            if st.button("Update Password"):
                update_password(old_password, new_password, confirm_password)
        else:
            st.warning("Hanya admin yang dapat mengubah kata sandi.")


if __name__ == "__main__":
    main()