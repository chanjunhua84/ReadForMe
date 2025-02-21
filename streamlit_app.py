import streamlit as st

# Handle ImportError exceptions for the required packages
try:
    import easyocr
    import numpy as np
    from gtts import gTTS
    from PIL import Image
    from transformers import pipeline
    from deep_translator import GoogleTranslator
    import cv2
    import validators
    import time
except ImportError as e:
    st.error(f"Error: {e.name} is not installed. Please install the missing package to proceed.")
    st.stop()  # Stop further execution if a package is missing

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

# Function to extract QR Code info using OpenCV and check for URL
def extract_qr_code(image: Image) -> str:
    # Convert image to RGB and then to NumPy array
    image_cv = np.array(image.convert('RGB'))
    
    if image_cv is None:
        st.error("Image failed to load.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(gray)

    if retval:
        for data in decoded_info:
            if validators.url(data):
                return "Invalid QR code: Website redirection is not allowed"
        return decoded_info
    return None

# Set Streamlit page layout
st.set_page_config(page_title="OCR & QR Code Reader", layout="wide")

# App title
st.markdown("<h1 style='text-align: left; color: #ff6f61;'>📝 Help Me Read</h1>", unsafe_allow_html=True)
st.markdown("<h4>No information will be collected or stored. This app is for your privacy and convenience.</h4>", unsafe_allow_html=True)
st.markdown("---")

# Language selection
languages = {"Blank": "NA", "Chinese": "zh-CN", "Malay": "ms", "Tamil": "ta"}
target_language = st.selectbox("Select your language for translation", list(languages.keys()))

# File uploader
uploaded_file = st.file_uploader("Upload an Image (or Scan QR)", type=["jpg", "jpeg", "png"])

if uploaded_file and target_language != "Blank":
    image = Image.open(uploaded_file)
    image = auto_rotate_image(image)
    img_array = np.array(image)
    st.image(image, caption="Uploaded Image", width=600)
    
    qr_info = extract_qr_code(image)
    
    if qr_info:
        if isinstance(qr_info, str) and qr_info.startswith("Invalid QR code"):
            st.error(qr_info)
        else:
            # Summarizing QR code information
            st.subheader("QR Code Information:")
            qr_summary = " | ".join(qr_info)
            st.write(qr_summary)
            
            # Summarize QR Code Information
            with st.spinner("Summarizing QR code info..."):
                time.sleep(1)
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                summarized_qr_info = summarizer(qr_summary, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
                
            st.subheader("QR Code Information Summary:")
            st.write(summarized_qr_info)
            st.markdown("---")
            
            # Translate QR Code Summary
            with st.spinner("Translating QR code info..."):
                time.sleep(1)
                translator = GoogleTranslator(source="en", target=languages[target_language])
                translated_qr_info = translator.translate(summarized_qr_info)
            
            st.subheader(f"Translated QR Code Info ({target_language}):")
            st.write(translated_qr_info)
            st.download_button("Download Translation", translated_qr_info, file_name="translated_qr_info.txt", mime="text/plain")
            st.markdown("---")
            
            # Generate Audio for Translated QR Code Info
            with st.spinner("Generating speech..."):
                time.sleep(1)
                tts = gTTS(text=translated_qr_info, lang=languages[target_language], slow=False)
                tts.save("output_qr_info.mp3")
            
            st.audio("output_qr_info.mp3", format="audio/mp3")
            st.download_button("Download Audio", "output_qr_info.mp3", file_name="qr_info_speech.mp3")
    else:
        st.subheader("No QR Code Found. Proceeding with OCR...")
        st.markdown("---")
        
        with st.spinner("Extracting text..."):
            time.sleep(1)
            reader = easyocr.Reader(['en'])
            results = reader.readtext(img_array)
        
        extracted_text = "\n".join([res[1] for res in results])
        st.subheader("Extracted Text:")
        st.write(extracted_text)
        st.markdown("---")
        
        with st.spinner("Summarizing text..."):
            time.sleep(1)
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summarized_text = summarizer(extracted_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        
        st.subheader("Summary:")
        st.write(summarized_text)
        st.markdown("---")
        
        with st.spinner("Translating text..."):
            time.sleep(1)
            translator = GoogleTranslator(source="en", target=languages[target_language])
            translated_text = translator.translate(summarized_text)
        
        st.subheader(f"Translated Text ({target_language}):")
        st.write(translated_text)
        st.download_button("Download Translation", translated_text, file_name="translation.txt", mime="text/plain")
        st.markdown("---")
        
        with st.spinner("Generating speech..."):
            time.sleep(1)
            tts = gTTS(text=translated_text, lang=languages[target_language], slow=False)
            tts.save("output.mp3")
        
        st.audio("output.mp3", format="audio/mp3")
        st.download_button("Download Audio", "output.mp3", file_name="speech.mp3")
