import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 1. Load your custom photograph
image_path = 'my_digit.jpg' 
original_img = Image.open(image_path)

# 2. Preprocess the image to match MNIST standards
# A. Convert to Grayscale (removes RGB color channels)
img = original_img.convert('L')

# B. Invert colors (MNIST expects white ink on a black background)
# This turns your black marker on white paper into white marker on black paper
img = ImageOps.invert(img)

# C. Resize to exactly 28x28 pixels
img = img.resize((28, 28))

# D. Convert to a standard array and normalize pixel values to between 0.0 and 1.0
image_array = np.array(img, dtype=np.float32)
image_array = image_array / 255.0

# E. Reshape to match the ONNX input shape: (Batch Size 1, Channel 1, Height 28, Width 28)
onnx_input = image_array.reshape(1, 1, 28, 28)

# 3. Run the ONNX engine
providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("mnist_model.onnx", providers=providers)
results = session.run(None, {"image_input": onnx_input})

# 4. Calculate probabilities (Softmax)
raw_scores = results[0][0]
probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores))
predicted_digit = np.argmax(probabilities)

# 5. Visualize the transformation and result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Show the transformed 28x28 inverted image so you can see what the model sees
ax1.imshow(img, cmap='gray')
ax1.set_title("How ONNX sees your photo")
ax1.axis('off')

# Show the confidence graph
digits = np.arange(10)
bars = ax2.bar(digits, probabilities * 100, color='lightgray')
ax2.set_xticks(digits)
ax2.set_xlabel('Digit')
ax2.set_ylabel('Confidence (%)')
ax2.set_title(f"Prediction: {predicted_digit}")
bars[predicted_digit].set_color('#007AFF')

plt.tight_layout()
plt.show()