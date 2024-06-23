import streamlit as st
import pytesseract
from PIL import Image

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def ocr_image(image, languages):
    # Perform OCR on the image
    text = pytesseract.image_to_string(image, lang=languages)
    return text

def process_image(image, languages):
    # Perform OCR on the image
    text = ocr_image(image, languages)
    return text.strip()  # Strip leading/trailing whitespace

st.title('OCR Web App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Process Image'):
        try:
            # Process the image
            text = process_image(image, 'eng+tam+hin+tel+kan+san+mal')
            st.text_area("Extracted Text", text,height=800,)
        except Exception as e:
            st.error(f"Error: {str(e)}")
