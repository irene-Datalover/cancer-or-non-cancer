import torch
from torchvision import transforms
from PIL import Image
from model import CancerDetectionModel

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CancerDetectionModel(num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Prediction Function
def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return "Cancerous" if predicted.item() == 1 else "Non-Cancerous"

# Test Prediction
print(predict("data/test/sample_image.jpg"))
