import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os
import gdown

# --- Конфигурация ---
# ID вашего файла на Google Диске (замените на свой)
GOOGLE_DRIVE_FILE_ID = "YOUR_FILE_ID_HERE"
MODEL_FILENAME = "best.pt"

@st.cache_resource
def download_and_load_model():
    """
    Скачивает модель с Google Диска (если её нет локально)
    и загружает её с помощью YOLO.
    """
    model_path = MODEL_FILENAME

    if not os.path.exists(model_path):
        with st.spinner("Загружаем модель с Google Диска... (это займёт ~1-2 минуты)"):
            url = f"https://drive.google.com/uc?id={"1UadDRyhUMw1oJXIzWONw2kjERhl0Xq7p"}"
            gdown.download(url, model_path, quiet=False)
            st.success("Модель успешно загружена!")

    # Загружаем модель (кэшируется Streamlit)
    return YOLO(model_path)

def draw_boxes(image, results, class_names):
    """Рисует bounding boxes и подписи на изображении (работает с BGR)."""
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{class_names[cls_id]}: {conf:.2f}"

                # Цвета для классов (можно настроить под свои)
                colors = {
                    0: (255, 0, 0),    # Platelets – синий
                    1: (0, 255, 0),    # RBC – зелёный
                    2: (0, 0, 255)     # WBC – красный
                }
                color = colors.get(cls_id, (255, 255, 255))

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# --- Настройка страницы ---
st.set_page_config(page_title="Blood Cell Detection", layout="wide")
st.title("🩸 Детекция клеток крови (YOLOv8)")
st.markdown("Загрузите изображение для анализа")

# --- Загрузка модели (скачивается автоматически) ---
try:
    model = download_and_load_model()
    class_names = model.names
    st.success("Модель загружена и готова к работе")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# --- Загрузка изображения пользователем ---
uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Чтение изображения
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)

    # Детекция (передаём массив NumPy, временный файл не нужен)
    with st.spinner("Выполняется детекция..."):
        results = model(img_array, conf=0.25)

    # Отрисовка результатов (OpenCV работает в BGR)
    result_img = draw_boxes(img_array.copy(), results, class_names)

    with col2:
        st.subheader("Результат детекции")
        st.image(result_img, use_container_width=True, channels="BGR")

    # Статистика по классам
    st.subheader("Найденные объекты:")
    if len(results[0].boxes) > 0:
        counts = {}
        for box in results[0].boxes:
            cls_name = class_names[int(box.cls[0])]
            counts[cls_name] = counts.get(cls_name, 0) + 1

        for name, cnt in counts.items():
            st.write(f"- **{name}**: {cnt}")
    else:
        st.write("Объекты не обнаружены")