import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.data_set import MovieDataset
from models.save_file import find_save_path


# 定义一维卷积神经网络模型
class CNN(nn.Module):
    def __init__(self, input_size, output_dim, weight_decay=0.01):
        super(CNN, self).__init__()

        # 定义卷积层和池化层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 计算线性层的输入大小
        linear_input_size = int(input_size / 4) * 64
        
        # 定义全连接层
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

        # 定义L2正则化（权重衰减）
        self.weight_decay = weight_decay

    def forward(self, x):
        # 前向传播过程
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        # 添加L2正则化
        l2_reg = torch.tensor(0.).to(x.device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        
        return x + self.weight_decay * l2_reg


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    best_performance = -100.0
    current_patience = 0

    # 用于记录每个 epoch 的指标
    epoch_losses = []
    epoch_accuracies = []
    epoch_performances = []

    train_loader = [{'features': batch['features'].to('cuda'), 'score': batch['score'].float().to('cuda')} for batch in train_loader]
    
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs = batch['features']
            labels = batch['score']

            optimizer.zero_grad()
            outputs = model(inputs)
            scores = outputs.squeeze()
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # 统计差值小于0.5的个数和总数
            diff = torch.abs(scores - labels)
            correct += torch.sum(diff < 1).item()
            total += len(labels)

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)

        # 计算准确率
        accuracy = correct / total

        # 计算性能
        performance = accuracy * 10 - avg_loss * 0.25

        # 记录每个 epoch 的指标
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy * 100)
        epoch_performances.append(performance)

        # 如果验证准确率提升，则更新最佳准确率和重置耐心计数器
        if performance > best_performance:
            best_performance = performance
            current_patience = 0
        else:
            current_patience += 1

        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Performance: {performance:.4f}, Patience: {current_patience:.4f}')

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
    file_name = find_save_path('CNN', 'C:\\Users\\15437\\Desktop\\repository\\movie_evaluation\\notebooks\\', '.png')
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
            labels = batch['score'].float().to('cuda')
            
            # 使用模型预测score
            score_predictions = model(inputs).squeeze()

            # 统计差值小于0.5的个数和总数
            diff = torch.abs(score_predictions - labels)
            correct += torch.sum(diff < 1).item()
            total += len(labels)

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        
        return accuracy


def cnn_1d_predict(df):
    # 划分训练集和测试集
    train_data, test_data = train_test_split(df, test_size=0.1)

    # 准备数据加载器
    train_dataset = MovieDataset(train_data)
    test_dataset = MovieDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    # 初始化模型
    input_size = len(train_dataset[0]['features'])
    output_dim = 1
    model = CNN(input_size, output_dim)
    model = model.to('cuda')  # 将模型移到CUDA上
    model = nn.DataParallel(model)  # 多核并行加速

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.004)  # 使用Adam

    criterion = criterion.to('cuda')

    # 训练模型
    num_epochs = 1800
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 测试模型
    accuracy = test_model(model, test_loader)

    # 保存模型到文件
    file_name = find_save_path('CNN', 'C:\\Users\\15437\\Desktop\\repository\\movie_evaluation\\weight\\', '.pth')
    torch.save(model.state_dict(), file_name)

    return accuracy