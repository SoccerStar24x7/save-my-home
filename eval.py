
def evaluate(img):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import resnet18, ResNet18_Weights
    from PIL import Image
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

    # normalizes imange so not cooked
    img = Image.open(img).convert("RGB")
    input_tensor = transform(img)

    input_batch = input_tensor.unsqueeze(0)

    # evaluate
    with torch.no_grad():
        output = model(input_batch)

    # makes output readable or sum shi
    probabilities = torch.nn.functional.softmax(output[0], dim=0) # makes       probability
    predicted_class_idx = torch.argmax(probabilities).item()

    return predicted_class_idx
