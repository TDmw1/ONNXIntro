import onnxruntime as ort
import numpy as np
import torchvision
import matplotlib.pyplot as plt

# 1. Load a real handwritten digit from the test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Pick a specific image (try changing this index to see different digits!)
image_index = 42 
pil_image, true_label = test_dataset[image_index]

# 2. Format the image exactly how the ONNX model expects it
image_array = np.array(pil_image, dtype=np.float32)
image_array = image_array / 255.0 # Normalize pixels to values between 0 and 1
onnx_input = image_array.reshape(1, 1, 28, 28) # (Batch Size, Channels, Height, Width)

# 3. Run the ONNX engine
providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("mnist_model.onnx", providers=providers)
results = session.run(None, {"image_input": onnx_input})

# 4. Convert the raw output into readable percentages (Softmax)
raw_scores = results[0][0]
probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores))
predicted_digit = np.argmax(probabilities)

# 5. Visualize the inputs and outputs side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left Panel: The actual image
ax1.imshow(pil_image, cmap='gray')
ax1.set_title(f"True Label: {true_label}")
ax1.axis('off')

# Right Panel: The model's confidence scores
digits = np.arange(10)
bars = ax2.bar(digits, probabilities * 100, color='lightgray')
ax2.set_xticks(digits)
ax2.set_xlabel('Digit')
ax2.set_ylabel('Confidence (%)')
ax2.set_title(f"ONNX Prediction: {predicted_digit}")

# Highlight the winning prediction in blue
bars[predicted_digit].set_color('#007AFF')

plt.tight_layout()
plt.show()