import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 定义 CNN - Transformer 模型
class CNNTransformer(nn.Module):
    def __init__(self, input_channels, num_classes, d_model=64, nhead=4, num_layers=2):
        super(CNNTransformer, self).__init__()

        # CNN 部分
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Transformer 部分
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全连接层用于分类
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # 调整维度以适应 Transformer 的输入要求
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取序列的最后一个时间步的特征
        x = self.fc(x)
        return x


# 自定义数据集类
class GestureDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_excel(file_path)
        self.labels = data.iloc[:, -1].values
        self.data = data.iloc[:, :-1].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        label = torch.tensor(self.labels[idx], dtype=torch.long) - 1  # 标签从 0 开始编号
        return sample, label


# 超参数设置
input_channels = 1  # 输入通道数，根据数据调整
num_classes = 5  # 手势类别数，根据实际情况调整
batch_size = 32
epochs = 50
learning_rate = 0.0001  # 降低学习率

# 文件路径
file_path = r"D:\学习资料\大三下\mec202\处理过后的肌电信号数据\handdata\data.xlsx"

# 加载数据
dataset = GestureDataset(file_path)

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = CNNTransformer(input_channels, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 记录每个 epoch 的 loss
train_losses = []

# 训练模型
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_dataloader)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

# 计算测试集准确率
model.eval()
correct = 0
total = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy}%')

# 绘制训练 loss 曲线
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# 计算每个真实类别对应的预测结果分布
prediction_distribution = {}
for true_label in range(num_classes):
    true_indices = [i for i, label in enumerate(all_labels) if label == true_label]
    true_count = len(true_indices)
    prediction_distribution[true_label] = [0] * num_classes
    for index in true_indices:
        predicted_label = all_predictions[index]
        prediction_distribution[true_label][predicted_label] += 1
    # 转换为比例
    if true_count > 0:
        prediction_distribution[true_label] = [count / true_count for count in prediction_distribution[true_label]]

# 定义颜色映射
colors = ['red', 'orange', 'yellow', 'green', 'blue']

# 绘制每个真实类别对应的预测结果分布（饼状图）
for i in range(num_classes):
    plt.subplot(2, 3, i + 2)
    plt.pie(prediction_distribution[i], labels=[f'Pred {j + 1}' for j in range(num_classes)], autopct='%1.1f%%',
            colors=colors)
    plt.title(f'True Class {i + 1} Prediction Distribution')

plt.tight_layout()
plt.show()
