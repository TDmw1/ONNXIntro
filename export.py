import torch
import torch.nn as nn

# 1. Rebuild the exact same architecture
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# 2. Instantiate and load the trained weights
model = MNISTClassifier()
model.load_state_dict(torch.load("mnist_model.pt", weights_only=True))

# 3. Set to evaluation mode
model.eval()

# 4. Create a dummy input tensor
# Shape: (Batch Size of 1, 1 Color Channel, 28x28 pixels)
dummy_input = torch.randn(1, 1, 28, 28)

# 5. Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mnist_model.onnx",
    export_params=True,
    input_names=["image_input"],
    output_names=["digit_prediction"]
)
print("Model successfully exported to mnist_model.onnx")