import onnxruntime as ort
import numpy as np
from PIL import Image

# 1. Start the ONNX Runtime session
providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("mnist_model.onnx", providers=providers)

# 2. Verify the hardware acceleration
print(f"Executing on: {session.get_providers()[0]}")

# 3. Create a fake image (since we don't have a real one drawn)
# MNIST images are 28x28 pixels, grayscale (1 channel).
# We use NumPy because ONNX Runtime expects standard arrays, not PyTorch tensors.
dummy_image = np.random.rand(1, 1, 28, 28).astype(np.float32)

# 4. Run the model
# We pass the input array matching the exact name we set during export: "image_input"
results = session.run(None, {"image_input": dummy_image})

# 5. Process the output
predictions = results[0] # results is a list, the first item contains our 10 probabilities
predicted_digit = np.argmax(predictions) # Find the index with the highest score

print(f"Raw Output shape: {predictions.shape}")
print(f"The model predicts this random noise is the digit: {predicted_digit}")