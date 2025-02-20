import streamlit as st
import easyocr
import numpy as np
from gtts import gTTS
from PIL import Image
from transformers import pipeline
from deep_translator import GoogleTranslator
import cv2
import validators  # Added to check for URLs
import time

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
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(gray)

    if retval:
        # Filter out URLs and reject QR codes that point to a website
        for data in decoded_info:
            if validators.url(data):  # Check if the data is a valid URL
                return "Invalid QR code: Website redirection is not allowed"
        return decoded_info  # List of QR code data found
    return None

# Set Streamlit page layout for mobile-friendliness
st.set_page_config(page_title="Help Me Read", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        text-align: left;
        font-size: 1.8rem;
        color: #ff6f61;
    }
    .header {
        font-size: 1.2rem;
        color: #333;
    }
    .section-title {
        font-size: 1.1rem;
        color: #3b3b3b;
    }
    .download-btn {
        background-color: #4CAF50;
        color: white;
        padding: 12px 20px;
        font-size: 1.2rem;
        border-radius: 10px;
        text-align: center;
    }
    .download-btn:hover {
        background-color: #45a049;
    }
    .spinner {
        text-align: center;
        margin-top: 20px;
    }

    /* Responsive styles for mobile */
    @media only screen and (max-width: 768px) {
        .title {
            font-size: 1.5rem;
        }
        .header, .section-title {
            font-size: 1rem;
        }
        .download-btn {
            font-size: 1rem;
            padding: 10px 16px;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    }
    </style>
    """, unsafe_allow_html=True
)

# App title with left alignment
st.markdown("<div class='title'>📝 Help Me Read</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>No information will be collected or stored. This app is for your privacy and convenience.</div>", unsafe_allow_html=True)
# Section break
st.markdown("---")

# Sidebar for language selection first, followed by input method
st.sidebar.title("🗣️ Language Selection & Input Method")

# Language selection comes first at the top
languages = {"Blank": "NA", "Chinese": "zh-CN", "Malay": "ms", "Tamil": "ta"}
target_language = st.sidebar.selectbox("Select your language for translation", list(languages.keys()))

# Input method selection
input_method = st.sidebar.radio("Choose input method", ("Upload Image", "Use Camera"))

# Initialize session state if not set
if 'prev_language' not in st.session_state:
    st.session_state.prev_language = ""
if 'prev_qr_info' not in st.session_state:
    st.session_state.prev_qr_info = None
if 'prev_uploaded_file' not in st.session_state:
    st.session_state.prev_uploaded_file = None

# Proceed once language is selected
if target_language != "Blank":
    # Image upload or camera input logic
    image = None

    if input_method == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an Image (or Scan QR)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    elif input_method == "Use Camera":
        camera_image = st.camera_input("Take a picture", key="camera")
        if camera_image is not None:
            image = Image.open(camera_image)

    if image is not None:
        # Auto-rotate the image
        image = auto_rotate_image(image)

        img_array = np.array(image)

        # Display image with responsive width for mobile
        st.image(image, caption="Processed Image", width=600)

        # QR Code Detection using OpenCV
        qr_info = extract_qr_code(image)

        # Store current values in session state
        st.session_state.prev_language = target_language
        st.session_state.prev_uploaded_file = uploaded_file
        st.session_state.prev_qr_info = qr_info

        # Show QR Code Information at the top
        if qr_info:
            if isinstance(qr_info, str) and qr_info.startswith("Invalid QR code"):
                st.error(qr_info)  # Display error message if it's an invalid QR code
            else:
                st.subheader("QR Code Information:")
                st.write(" | ".join(qr_info))
        else:
            st.subheader("No QR Code Found. Proceeding with OCR...")

            # Section break
            st.markdown("---")

            # OCR Processing
            with st.spinner("Extracting text..."):
                time.sleep(1)  # Simulating loading time (remove it for actual processing)
                reader = easyocr.Reader(['en'])
                results = reader.readtext(img_array)

            extracted_text = "\n".join([res[1] for res in results])

            st.subheader("Extracted Text:")
            st.write(extracted_text)

            # Section break
            st.markdown("---")

            # Summarization
            with st.spinner("Summarizing text..."):
                time.sleep(1)  # Simulating loading time (remove it for actual processing)
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                summarized_text = summarizer(extracted_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

            st.subheader("Summary:")
            st.write(summarized_text)

            # Section break
            st.markdown("---")

            # Translation
            with st.spinner("Translating text..."):
                time.sleep(1)  # Simulating loading time (remove it for actual processing)
                translator = GoogleTranslator(source="en", target=languages[target_language])
                translated_text = translator.translate(summarized_text)

            st.subheader(f"Translated Text ({target_language}):")
            st.write(translated_text)
            st.download_button("Download Translation", translated_text, file_name="translation.txt", mime="text/plain")

            # Section break
            st.markdown("---")

            # Text-to-Speech
            with st.spinner("Generating speech..."):
                time.sleep(1)  # Simulating loading time (remove it for actual processing)
                tts = gTTS(text=translated_text, lang=languages[target_language], slow=False)
                tts.save("output.mp3")

            st.audio("output.mp3", format="audio/mp3")

            # Section break
            st.markdown("---")

            # Download buttons
            st.download_button("Download Audio", "output.mp3", file_name="speech.mp3")
