import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from gtts import gTTS
import tempfile
import tensorflow as tf
import json
import h5py

# --- Streamlit page config MUST be first ---
st.set_page_config(page_title="Alphabet Learner", layout="centered")

# --- Fix model config to handle compatibility issues ---
def fix_model_config(path):
    try:
        with h5py.File(path, 'r+') as f:
            if 'model_config' in f.attrs:
                model_config_data = f.attrs['model_config']
                
                # Handle both string and bytes cases
                if isinstance(model_config_data, bytes):
                    model_config = json.loads(model_config_data.decode('utf-8'))
                else:
                    model_config = json.loads(model_config_data)
                
                # Check if the expected structure exists
                if not model_config or 'config' not in model_config:
                    print("Model config structure is unexpected")
                    return
                
                config = model_config['config']
                if not config or 'layers' not in config:
                    print("Model config layers not found")
                    return
                
                # Fix various compatibility issues
                for layer in config['layers']:
                    if not layer or 'config' not in layer:
                        continue
                        
                    layer_config = layer['config']
                    if not layer_config:
                        continue
                    
                    # Remove batch_shape from InputLayer
                    if layer.get('class_name') == 'InputLayer' and 'batch_shape' in layer_config:
                        del layer_config['batch_shape']
                    
                    # Fix dtype policy issues
                    if 'dtype' in layer_config:
                        if isinstance(layer_config['dtype'], dict):
                            if layer_config['dtype'].get('class_name') == 'DTypePolicy':
                                # Replace with simple string dtype
                                dtype_name = layer_config['dtype']['config'].get('name', 'float32')
                                layer_config['dtype'] = dtype_name
                
                # Save back the fixed config
                f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
            else:
                print("No model_config attribute found in file.")
    except Exception as e:
        print(f"Error fixing model config: {str(e)}")
        # If we can't fix the config, just continue and let the model loading handle it

# --- Load model with custom objects ---
def load_model_safely(model_path):
    try:
        # Check if model file exists
        import os
        if not os.path.exists(model_path):
            return None, f"Model file '{model_path}' not found. Please ensure the model file is in the correct location."
        
        # First try to fix the config
        fix_model_config(model_path)
        
        # Try different loading approaches
        try:
            # Method 1: Standard loading with custom object scope
            with tf.keras.utils.custom_object_scope({}):
                model = tf.keras.models.load_model(model_path, compile=False)
            return model, None
        except Exception as e1:
            try:
                # Method 2: Load without compile
                model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})
                return model, None
            except Exception as e2:
                try:
                    # Method 3: Load weights only if possible
                    # This is a fallback - create a basic model and try to load weights
                    model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(28, 28, 1)),
                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(26, activation='softmax')
                    ])
                    
                    # Try to load weights if the architecture matches
                    model.load_weights(model_path)
                    return model, "Loaded weights only - architecture may not match exactly"
                except Exception as e3:
                    # All methods failed, return fallback
                    error_msg = f"All loading methods failed:\n1. {str(e1)}\n2. {str(e2)}\n3. {str(e3)}"
                    raise Exception(error_msg)
                
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        # Create a simple fallback model for demonstration
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')
        ])
        return model, error_msg

# --- Preprocess the canvas image for EMNIST prediction ---
def preprocess_image(image_data):
    # Convert RGBA to RGB if needed
    if image_data.shape[-1] == 4:
        image_data = image_data[..., :3]

    # Convert to grayscale
    gray = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Invert colors for EMNIST (white on black)
    inverted = 255 - gray

    # Threshold to binary
    _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box of non-zero pixels
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # Crop and center on square canvas
        cropped = binary[y:y+h, x:x+w]
        size = max(w, h) + 10  # margin
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    else:
        # No content, return blank 28x28
        square = np.zeros((28, 28), dtype=np.uint8)

    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0,1]
    normalized = resized.astype(np.float32) / 255.0

    # EMNIST orientation fix: rotate + flip
    rotated = np.transpose(normalized)
    flipped = np.flip(rotated, axis=1)

    # Return with channel dimension
    return flipped.reshape(28, 28, 1)

# --- Text to speech ---
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        st.audio(tmp.name, format="audio/mp3")

# --- Example words dictionary ---
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

# --- Load model safely ---
model_path = "emnist_letters_finetuned.h5"
model, error_msg = load_model_safely(model_path)

# --- Display any model loading errors/warnings ---
if error_msg:
    st.error(error_msg)
    st.warning("Using fallback model. Please check your model file.")

# --- App UI ---
st.markdown("<h1 style='text-align: center;'>✍️ Alphabet Learning App</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Draw a lowercase letter (a-z) SLOWLY, click <b>Reveal</b>, then hear how it sounds and see words starting with it!</p>",
    unsafe_allow_html=True
)

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

# --- Predict and reveal letter ---
if st.button("🪄 Reveal My Magic Letter!"):
    if canvas_result.image_data is not None:
        img = preprocess_image(canvas_result.image_data)
        prediction = model.predict(img.reshape(1, 28, 28, 1))
        predicted_letter = chr(np.argmax(prediction) + ord('a'))

        st.session_state.predicted_letter = predicted_letter
        st.session_state.show_words_permission = True

        # Animated letter reveal UI
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

# --- Show example words and buttons to speak ---
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
                speak(word)
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