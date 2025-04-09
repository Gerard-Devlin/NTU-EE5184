import tkinter as tk
from tkinter import Canvas, Button, Label
import paddle
import paddle.nn as nn
from PIL import Image, ImageDraw, ImageOps
from paddle.vision import transforms


# ======== 1. 模型结构 ========= #
class SimpleNN(nn.Layer):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1D(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1D(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1D(128)
        self.output = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.1)
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.1)
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.1)
        x = self.dropout(x)
        return self.output(x)


# ======== 2. 图像预处理函数 ========= #
def preprocess_image(img):
    img = img.convert("L")  # 灰度
    img = ImageOps.pad(img, (28, 28), color=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
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
        self.prob_label.grid(row=3, column=0, columnspan=4, sticky="w")

        Button(self.window, text="Predict", command=self.predict).grid(row=2, column=0)
        Button(self.window, text="Clear", command=self.clear).grid(row=2, column=1)
        Button(self.window, text="Exit", command=self.window.quit).grid(row=2, column=2)

        self.image = Image.new("L", (280, 280), color=0)
        self.draw_interface = ImageDraw.Draw(self.image)

        self.device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')

        self.model = SimpleNN()
        state_dict = paddle.load("model.pdparams")
        self.model.set_state_dict(state_dict)
        self.model.eval()

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
        self.prob_label.config(text="Probabilities:")

    def predict(self):
        img_tensor = preprocess_image(self.image)
        output = self.model(img_tensor)
        prob = nn.functional.softmax(output, axis=1)
        pred = paddle.argmax(prob, axis=1).item()
        conf = prob[0][pred].item()

        self.label.config(text=f"Prediction: {pred} ({conf:.2%})")

        # 概率显示
        prob_text = "Probabilities:\n"
        for i, p in enumerate(prob[0]):
            prob_text += f"  {i}: {p.item():.2%}\n"
        self.prob_label.config(text=prob_text)


if __name__ == '__main__':
    App()
