# training

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root='data', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

model = resnet18(weights=ResNet18_Weights.DEFAULT)

nClass = 3
model.fc = torch.nn.Linear(model.fc.in_features, nClass) # makes 2 output possibile things
print("dfa")
# grader and fixer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train() # "yo lock the fuck in"
for images, labels in train_loader: # starts training
    optimizer.zero_grad() # resets from last batch
    outputs = model(images) # "what do you think this is?"
    loss = criterion(outputs, labels) # calculates how off the model was

    # goes and fixes itself
    loss.backward()
    optimizer.step()
    print("a")

torch.save(model.state_dict(), "model_weights.pth") # save model
print("all good in the hood")
