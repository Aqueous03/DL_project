import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os
import gdown

# --- Конфигурация ---
GOOGLE_DRIVE_FILE_ID = "1EXYJf7CHb2PrhHI_6Y8Hmshcl5cFnRn7"   #ID файла на Google Диске
MODEL_FILENAME = "best.pt"

@st.cache_resource
def download_and_load_model():
    """Скачивает модель с Google Диска (если её нет локально) и загружает её."""
    model_path = MODEL_FILENAME
    if not os.path.exists(model_path):
        with st.spinner("Загружаем модель с Google Диска... (это займёт ~1-2 минуты)"):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, model_path, quiet=False)
            st.success("Модель успешно загружена!")
    return YOLO(model_path)

def draw_boxes(image, boxes, class_names):
    """Рисует bounding boxes и подписи на изображении (работает с BGR)."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{class_names[cls_id]}: {conf:.2f}"

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

# --- Загрузка модели ---
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
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)

    with st.spinner("Выполняется детекция..."):
        results = model.predict(
            source=img_bgr,
            imgsz=416,
            conf=0.45,
            device="cpu",
            save=False,
            verbose=False
        )

    result_img = img_bgr.copy()
    counts = {}

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            result_img = draw_boxes(img_bgr.copy(), boxes, class_names)
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                counts[cls_name] = counts.get(cls_name, 0) + 1

    with col2:
        st.subheader("Результат детекции")
        st.image(result_img, use_container_width=True, channels="BGR")

    st.subheader("Найденные объекты:")
    if counts:
        for name, cnt in counts.items():
            st.write(f"- {name}: {cnt}")
    else:
        st.write("Объекты не обнаружены")