import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import easyocr
from docx import Document
import os
from PIL import Image
import io

# Инициализация EasyOCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ru', 'en'])

reader = load_ocr()

def preprocess_image(image_path):
    """Улучшаем изображение для OCR"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Конвертируем в серый
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Повышаем контрастность
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Увеличиваем резкость
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Бинаризация (черно-белое)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Сохраняем временный файл
    temp_preprocessed = "temp_preprocessed.png"
    cv2.imwrite(temp_preprocessed, binary)
    return temp_preprocessed

def ocr_from_image(image_path):
    """Распознавание через EasyOCR"""
    result = reader.readtext(image_path, detail=0)
    text = " ".join(result)
    return text.strip()

def ocr_from_pdf(pdf_path):
    """Распознавание текста из PDF файла через PyMuPDF"""
    pdf_document = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")

        # Временное изображение
        temp_img_path = f"temp_page_{page_num}.png"
        with open(temp_img_path, "wb") as f:
            f.write(img_data)

        # OCR
        text = ocr_from_image(temp_img_path)
        full_text += text + "\n"

        # Удаление временного файла
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    pdf_document.close()
    return full_text

def create_docx(text, filename):
    """Создаёт .docx файл"""
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filename)

st.title("OCR — Распознавание текста из PDF и изображений")
st.write("Загрузите PDF или изображение (JPG, PNG) для распознавания")

uploaded_files = st.file_uploader("Выберите файлы", type=['pdf', 'jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        st.write(f"Обработка: {uploaded_file.name}")

        # Сохраняем загруженный файл
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.lower().endswith('.pdf'):
            text = ocr_from_pdf(uploaded_file.name)
        else:
            # Предобработка
            temp_path = preprocess_image(uploaded_file.name)
            text = ocr_from_image(temp_path)
            os.remove(temp_path)

        if text:
            all_text += f"\n--- Текст из: {uploaded_file.name} ---\n{text}\n"
        else:
            st.warning(f"⚠️ Не удалось распознать текст из: {uploaded_file.name}")

        os.remove(uploaded_file.name)

    if all_text:
        # Создаём .docx
        output_filename = "результат.docx"
        create_docx(all_text, output_filename)

        # Предоставляем пользователю возможность скачать
        with open(output_filename, "rb") as f:
            st.download_button(
                label="Скачать .docx файл",
                data=f,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        os.remove(output_filename)