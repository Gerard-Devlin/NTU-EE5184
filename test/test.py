#%% md
# ## Import needed packages
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
#%% md
# ## Hyperparameter
# 
#%%
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size" : 64,
    "learning_rate" : 0.01 ,
    "epochs":15,
}
#%% md
# ## Dataloader
#%%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
#%% md
# ## Visualize
# 
#%%
examples = iter(train_loader)
images, labels = next(examples)

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(images[i][0], cmap='gray')
    # plt.title(f"Label: {labels[i].item()}")
    plt.title(f"label:{labels[i]}")
    plt.axis('off')
plt.show()

#%% md
# ## Model
# 
#%%
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                                 # 展平 28x28 图像
            nn.Linear(28*28, 512),                        # 输入层
            nn.BatchNorm1d(512),                          # 批归一化
            nn.LeakyReLU(0.1),                            # LeakyReLU 防止死神经元
            nn.Dropout(0.3),                              # Dropout 防止过拟合

            nn.Linear(512, 256),                          # 中间层1
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(256, 128),                          # 中间层2
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 10),                           # 输出层
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNN().to(config["device"])

#%% md
# ## Loss Function and Optimizer
#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
#%% md
# ## Trainig loop
#%%
from tqdm import tqdm

for epoch in range(config["epochs"]):
    total_loss = 0.0
    model.train()

    print(f"\n🔄 Epoch {epoch+1}/{config['epochs']}")
    train_bar = tqdm(train_loader, desc="Training", dynamic_ncols=True, leave=False)

    for images, labels in train_bar:
        images, labels = images.to(config["device"]), labels.to(config["device"])

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

#%% md
# ## Testing
#%%
with torch.no_grad():
    total = 0
    correct = 0
    model.eval()
    for images, labels in tqdm(test_loader, desc="Testing", dynamic_ncols=True, leave=False):
        images, labels = images.to(config["device"]),labels.to(config["device"])
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
#%% md
# ## Saving Model
#%%
torch.save(model.state_dict(), "model.pth")
#%% md
# ## Load Model and Predict
#%%
model.load_state_dict(torch.load("model.pth"))
model.eval()

images, labels = next(iter(test_loader))
images, labels = images.to(config["device"]),labels.to(config["device"])
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
print(f"Test Accuracy: {100 * correct / total:.2f}%")
#%% md
# ## Visualize
#%%
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(config["device"]), labels.to(config["device"])
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# calculate
cm = confusion_matrix(all_labels, all_preds)

# Visualize
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
