import tkinter as tk
from tkinter import Canvas, Button, Label
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms


# ======== 1. 模型结构 ========= #
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 → 28x28
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),                              # 28x28 → 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 → 14x14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),                              # 14x14 → 7x7
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                                 # 展平为 64 * 7 * 7
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 10)                            # 输出层
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
# ======== 2. 图像预处理函数 ========= #
def preprocess_image(img):
    img = img.convert("L")  # 转换为灰度图 非常重要
    img = ImageOps.pad(img, (28, 28), color=0)  # 填充图像

    # 归一化参数：根据 MNIST 数据集的统计信息
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 数据集的标准化参数
    ])
    return transform(img).unsqueeze(0)

# ======== 3. GUI 应用类 ========= #
class App:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("MNIST")

        self.canvas = Canvas(self.window, width=280, height=280, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind('<B1-Motion>', self.draw)

        self.label = Label(self.window, text="Prediction: None", font=("Arial", 18))
        self.label.grid(row=1, column=0, columnspan=4)
        self.prob_label = Label(self.window, text="Probabilities:", font=("Arial", 12), justify="left", anchor="w")
        self.prob_label.grid(row=3, column=0, columnspan=4, sticky="w")  # 左对齐

        Button(self.window, text="Predict", command=self.predict).grid(row=2, column=0)
        Button(self.window, text="Clear", command=self.clear).grid(row=2, column=1)
        Button(self.window, text="Exit", command=self.window.quit).grid(row=2, column=2)

        self.image = Image.new("L", (280, 280), color=0)
        self.draw_interface = ImageDraw.Draw(self.image)

        # 确保选择设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = SimpleNN().to(self.device)
        self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        self.model.eval()  # 切换到评估模式

        self.window.mainloop()

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
        self.draw_interface.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=0)
        self.draw_interface = ImageDraw.Draw(self.image)
        self.label.config(text="Prediction: None")

    def predict(self):
        img_tensor = preprocess_image(self.image).to(self.device)  # 将图像数据传到设备
        output = self.model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        conf = prob[0][pred].item()
        self.label.config(text=f"Prediction: {pred} ({conf:.2%})")

        # 构建概率文本
        prob_text = "Probabilities:\n"
        for i, p in enumerate(prob[0]):
            prob_text += f"  {i}: {p.item():.2%}\n"

        # 设置到界面上
        self.prob_label.config(text=prob_text)


if __name__ == '__main__':
    App()
