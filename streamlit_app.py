import streamlit as st
import easyocr
import numpy as np
from gtts import gTTS
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import GoogleTranslator
import cv2
import validators  # Added to check for URLs
import time
import os  # For file handling
import asyncio

# Ensure Streamlit doesn't break due to async event loop conflicts
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load summarization model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, do_sample=False)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to auto-rotate an image based on its EXIF data
def auto_rotate_image(image: Image) -> Image:
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if tag == 274:  # Orientation tag
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
                    break
    except (AttributeError, KeyError, IndexError):
        pass  # No EXIF data, do nothing
    return image

# Function to extract QR Code info using OpenCV
def extract_qr_code(image: Image) -> str:
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, _, _ = qr_detector.detectAndDecodeMulti(gray)
    
    if retval:
        for data in decoded_info:
            if validators.url(data):
                return "Invalid QR code: Website redirection is not allowed"
        return decoded_info  # List of QR code data found
    return None

# Streamlit UI
st.set_page_config(page_title="Help Me Read", layout="wide")
st.title("📝 Help Me Read")
st.markdown("No information will be collected or stored.")

# Sidebar: Language selection
languages = {"Blank": "NA", "Chinese": "zh-CN", "Malay": "ms", "Tamil": "ta"}
target_language = st.sidebar.selectbox("Select your language", list(languages.keys()))
input_method = st.sidebar.radio("Choose input method", ("Upload Image", "Use Camera"))

if target_language != "Blank":
    image = None
    if input_method == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
    elif input_method == "Use Camera":
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            image = Image.open(camera_image)
    
    if image:
        image = auto_rotate_image(image)
        st.image(image, caption="Processed Image", width=600)
        
        qr_info = extract_qr_code(image)
        if qr_info:
            if isinstance(qr_info, str) and qr_info.startswith("Invalid QR code"):
                st.error(qr_info)
            else:
                st.subheader("QR Code Information:")
                st.write(" | ".join(qr_info))
        else:
            st.subheader("No QR Code Found. Proceeding with OCR...")
            
            with st.spinner("Extracting text..."):
                time.sleep(1)
                reader = easyocr.Reader(['en'])
                results = reader.readtext(np.array(image))
            
            extracted_text = "\n".join([res[1] for res in results])
            st.subheader("Extracted Text:")
            st.write(extracted_text)
            
            with st.spinner("Summarizing text..."):
                summarized_text = summarize_text(extracted_text)
            st.subheader("Summary:")
            st.write(summarized_text)
            
            with st.spinner("Translating text..."):
                translator = GoogleTranslator(source="en", target=languages[target_language])
                translated_text = translator.translate(summarized_text)
            st.subheader(f"Translated Text ({target_language}):")
            st.write(translated_text)
            st.download_button("Download Translation", translated_text, file_name="translation.txt", mime="text/plain")
            
            with st.spinner("Generating speech..."):
                tts = gTTS(text=translated_text, lang=languages[target_language], slow=False)
                audio_path = "output.mp3"
                tts.save(audio_path)
            st.audio(audio_path, format="audio/mp3")
            st.download_button("Download Audio", audio_path, file_name="speech.mp3", mime="audio/mp3")
            os.remove(audio_path)
