import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 3)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# evaluate ___________________________________________________________________________

model.load_state_dict(torch.load("model_weights.pth")) # load model
model.eval()  # actual test mode
from PIL import Image

# normalizes imange so not cooked
img = Image.open("none.jpeg").convert("RGB")
input_tensor = transform(img)

input_batch = input_tensor.unsqueeze(0)

# if gpu avaliable use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_batch = input_batch.to(device)

# evaluate
with torch.no_grad():
    output = model(input_batch)

# makes output readable or sum shi
probabilities = torch.nn.functional.softmax(output[0], dim=0) # makes probability
predicted_class_idx = torch.argmax(probabilities).item()

print(predicted_class_idx)
