import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
from PIL import Image
import numpy as np
from gtts import gTTS
import tempfile
import os
from keras.models import load_model
# --- Speak function ---
def preprocess_image(image_data):
    # Convert RGBA to RGB if necessary
    if image_data.shape[-1] == 4:
        image_data = image_data[..., :3]

    # Convert to grayscale
    gray = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Invert: EMNIST expects white on black
    inverted = 255 - gray

    # Threshold to binary image
    _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box of content
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # Crop and center in square
        cropped = binary[y:y+h, x:x+w]
        size = max(w, h) + 10  # Add some margin
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    else:
        # If no content, return blank
        square = np.zeros((28, 28), dtype=np.uint8)

    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    # EMNIST-style orientation: rotate and flip
    rotated = np.transpose(normalized)
    flipped = np.flip(rotated, axis=1)

    return flipped.reshape(28, 28, 1)
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        st.audio(tmp.name, format="audio/mp3")

# --- Example words ---
example_words = {
    'a': ['Apple', 'Ant'],
    'b': ['Ball', 'Bat'],
    'c': ['Cat', 'Cup'],
    'd': ['Dog', 'Drum'],
    'e': ['Elephant', 'Egg'],
    'f': ['Fish', 'Fan'],
    'g': ['Grapes', 'Goat'],
    'h': ['Hat', 'Hen'],
    'i': ['Ice', 'Ink'],
    'j': ['Jug', 'Jam'],
    'k': ['Kite', 'Key'],
    'l': ['Lion', 'Leaf'],
    'm': ['Monkey', 'Mug'],
    'n': ['Nest', 'Net'],
    'o': ['Owl', 'Orange'],
    'p': ['Pen', 'Pig'],
    'q': ['Queen', 'Quill'],
    'r': ['Rat', 'Ring'],
    's': ['Sun', 'Sock'],
    't': ['Tiger', 'Tap'],
    'u': ['Umbrella', 'Urn'],
    'v': ['Van', 'Violin'],
    'w': ['Wolf', 'Well'],
    'x': ['Xylophone', 'X-ray'],
    'y': ['Yak', 'Yarn'],
    'z': ['Zebra', 'Zip']
}

# --- Load model ---
model = load_model("emnist_letters_finetuned.keras", compile=False)

# --- Page config ---
st.set_page_config(page_title="Alphabet Learner", layout="centered")
st.markdown("<h1 style='text-align: center;'>✍️ Alphabet Learning App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Draw a lowercase letter (a-z) SLOWLY, click <b>Reaveal</b>, then hear how it sounds and what words start with it!</p>", unsafe_allow_html=True)

# --- Drawing canvas ---
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)



# --- Predict and Reveal Letter ---
if st.button("🪄 Reveal My Magic Letter!"):
    if canvas_result.image_data is not None:
        img = preprocess_image(canvas_result.image_data)
        prediction = model.predict(img.reshape(1, 28, 28, 1))
        predicted_letter = chr(np.argmax(prediction) + ord('a'))

        st.session_state.predicted_letter = predicted_letter
        st.session_state.show_words_permission = True  # Show words immediately
        st.session_state.spoken_once = True  # Prevent auto-speaking words

        # Colorful, animated letter reveal
        st.markdown(f"""
            <div style='text-align: center; margin-top: 20px;'>
                <div style='display: inline-block; padding: 30px 40px;
                            background: linear-gradient(145deg, #ffd93b, #ff7f50);
                            color: white; font-size: 80px; font-weight: bold;
                            border-radius: 30px; box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
                            animation: pop 0.5s ease-out;'>
                    {predicted_letter.upper()}
                </div>
            </div>
            <style>
                @keyframes pop {{
                    0% {{ transform: scale(0); }}
                    100% {{ transform: scale(1); }}
                }}
            </style>
        """, unsafe_allow_html=True)

        speak(predicted_letter)
# Assume example_words and speak() defined earlier

if st.session_state.get('show_words_permission', False) and 'predicted_letter' in st.session_state:
    letter = st.session_state.predicted_letter
    words = example_words.get(letter, [])

    st.markdown("<h3 style='text-align: center;'>Here are two fun words for you:</h3>", unsafe_allow_html=True)

    cols = st.columns(len(words))
    bg_colors = ['#e0f7fa', '#fce4ec']
    text_colors = ['#006064', '#ad1457']

    for i, word in enumerate(words):
        with cols[i]:
            if st.button(word, key=f"word_button_{letter}_{i}"):
                speak(word)  # use the gTTS-based speak that streams audio in browser
            st.markdown(
                f"""
                <style>
                div.stButton > button:first-child {{
                    width: 100%;
                    background: {bg_colors[i]};
                    color: {text_colors[i]};
                    font-size: 24px;
                    font-weight: bold;
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    border: none;
                }}
                div.stButton > button:first-child:hover {{
                    background: {text_colors[i]};
                    color: {bg_colors[i]};
                    cursor: pointer;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
