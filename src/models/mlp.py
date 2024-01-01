import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.data_set import MovieDataset
from models.save_file import find_save_path

# MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, patience=20):
    best_performance = -10.0
    current_patience = 0

    # 用于记录每个 epoch 的指标
    epoch_losses = []
    epoch_accuracies = []
    epoch_performances = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            inputs = batch['features'].to('cuda')
            labels = batch['rating'].long().to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if np.isnan(loss.detach().cpu().numpy()).any():
                print("Loss is NaN. Stopping the program.")
                sys.exit()

            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1) # 返回每一行中最大值的那个元素，且返回其索引
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)

        # 计算准确率
        accuracy = correct_predictions / total_samples

        # 计算性能
        performance = accuracy - avg_loss * 0.25

        # 记录每个 epoch 的指标
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        epoch_performances.append(performance)

        # 如果验证准确率提升，则更新最佳准确率和重置耐心计数器
        if performance > best_performance:
            best_performance = performance
            current_patience = 0
        else:
            current_patience += 1

        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Performance: {performance:.4f}, Patience: {current_patience:.4f}')

        # 如果连续 patience 次 epoch 验证集准确率没有提升，则退出训练
        if current_patience >= patience:
            print(f'Early stopping at epoch {epoch+1} as validation accuracy did not improve.')
            break

    # 绘制图表
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epoch_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy ( % )')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epoch_performances, label='Training Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.legend()

    plt.tight_layout()

    # 保存图表
    file_name = find_save_path('MLP', 'C:\\Users\\15437\\Desktop\\repository\\movie_evaluation\\notebooks\\', ".png")
    plt.savefig(file_name)

def test_model(model, test_loader):
    # 测试模型
    model.eval()
    model = model.to('cuda')  # 将模型移到CUDA上
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            inputs = batch['features'].to('cuda')
            labels = batch['rating'].long().to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')

    return accuracy


def mlp_predict(df):
    # 划分训练集和测试集
    train_data, test_data = train_test_split(df, test_size=0.1)

    # 准备数据加载器
    train_dataset = MovieDataset(train_data)
    test_dataset = MovieDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    # 初始化模型
    input_size = len(train_dataset[0]['features'])
    hidden_size = 4 * input_size
    output_size = 11  # 评分范围在0-10之间，步长 1
    patience = 7
    model = MLP(input_size, hidden_size, output_size)

    model = model.to('cuda')  # 将模型移到CUDA上

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam

    criterion = criterion.to('cuda')

    # 训练模型
    num_epochs = 300
    train_model(model, train_loader, criterion, optimizer, num_epochs, patience)

    # 测试模型
    accuracy = test_model(model, test_loader)

    # 保存模型到文件
    file_name = find_save_path('MLP', 'C:\\Users\\15437\\Desktop\\repository\\movie_evaluation\\weight\\', '.pth')
    torch.save(model.state_dict(), file_name)

    return accuracy