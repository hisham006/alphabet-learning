import cv2
import numpy as np

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
