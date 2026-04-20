import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def draw_boxes(image, results, class_names):
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{class_names[cls_id]}: {conf:.2f}"
                colors = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255)}
                color = colors.get(cls_id, (255,255,255))
                cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
                cv2.putText(image, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

st.set_page_config(page_title="Blood Cell Detection", layout="wide")
st.title("🩸 Детекция клеток крови (YOLOv8)")

model_path = "C:\\Users\\IVAN\\Desktop\\DL_project\\best.pt"
if not os.path.exists(model_path):
    st.error(f"Модель не найдена: {model_path}")
    st.stop()

model = load_model(model_path)
st.success("Модель загружена")

uploaded_file = st.file_uploader("Выберите изображение", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)

    with st.spinner("Выполняется детекция..."):
        results = model(img_array, conf=0.25)   # ← массив вместо файла

    result_img = draw_boxes(img_array.copy(), results, model.names)

    with col2:
        st.subheader("Результат детекции")
        st.image(result_img, use_container_width=True, channels="BGR")

    st.subheader("Найденные объекты:")
    if len(results[0].boxes) > 0:
        counts = {}
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            counts[cls_name] = counts.get(cls_name, 0) + 1
        for name, cnt in counts.items():
            st.write(f"- **{name}**: {cnt}")
    else:
        st.write("Объекты не обнаружены")