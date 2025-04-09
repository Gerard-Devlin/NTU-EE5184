#%% md
# ## Import needed packages
#%%
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize, ToTensor
from paddle.vision.datasets import MNIST
from paddle.io import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm

#%% md
# ## Hyperparameter
#%%
config = {
    "device": "gpu" if paddle.is_compiled_with_cuda() else "cpu",
    "batch_size": 64,
    "learning_rate": 0.01,
    "epochs": 5,
}

#%% md
# ## Dataloader
#%%
transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])

train_dataset = MNIST(mode='train', transform=transform)
test_dataset = MNIST(mode='test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

#%% md
# ## Visualize
#%%
examples = next(iter(train_loader))
images, labels = examples

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f"label:{labels[i].numpy()[0]}")
    plt.axis('off')
plt.show()

#%% md
# ## Model
#%%
# class SimpleNN(nn.Layer):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(28*28, 512)
#         self.bn1 = nn.BatchNorm1D(512)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1D(256)
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1D(128)
#         self.output = nn.Linear(128, 10)
#         self.dropout = nn.Dropout(p=0.3)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
#         x = self.dropout(x)
#         x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.1)
#         x = self.dropout(x)
#         x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.1)
#         x = self.dropout(x)
#         return self.output(x)

class CNNModel(nn.Layer):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(1, 32, kernel_size=3, padding=1),   # 1x28x28 → 32x28x28
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2)                   # 32x14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(32, 64, kernel_size=3, padding=1),  # 64x14x14
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2)                   # 64x7x7
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = paddle.flatten(x, start_axis=1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



model = CNNModel()
model.to(config["device"])

#%% md
# ## Loss Function and Optimizer
#%%
criterion = nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=config["learning_rate"])

#%% md
# ## Training Loop
#%%
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0.0

    print(f"\n🔄 Epoch {epoch+1}/{config['epochs']}")
    train_bar = tqdm(train_loader, desc="Training", dynamic_ncols=True, leave=False)

    for batch_id, (images, labels) in enumerate(train_bar):
        images, labels = images, labels.squeeze().astype('int64')
        preds = model(images)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        total_loss += loss.numpy().item()
        train_bar.set_postfix(loss=loss.numpy().item())


    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

#%% md
# ## Testing
#%%
model.eval()
correct = 0
total = 0

with paddle.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", dynamic_ncols=True, leave=False):
        images, labels = images, labels.squeeze().astype('int64')
        preds = model(images)
        predicted = preds.argmax(axis=1)
        correct += (predicted == labels).sum().item()
        total += labels.shape[0]

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

#%% md
# ## Saving Model
#%%
paddle.save(model.state_dict(), "model.pdparams")

#%% md
# ## Load Model and Predict
#%%
model.load_dict(paddle.load("model.pdparams"))
model.eval()

images, labels = next(iter(test_loader))
labels = labels.squeeze().astype('int64')
outputs = model(images)
predicted = outputs.argmax(axis=1)
print(f"Predicted: {predicted.numpy()[:10]}")
print(f"Ground Truth: {labels.numpy()[:10]}")

#%% md
# ## Confusion Matrix Visualization
#%%
model.eval()
all_preds = []
all_labels = []

with paddle.no_grad():
    for images, labels in test_loader:
        labels = labels.squeeze().astype('int64')
        outputs = model(images)
        preds = outputs.argmax(axis=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
