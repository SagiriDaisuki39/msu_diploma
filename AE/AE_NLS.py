import csv
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms  # 导入transforms模块，用于数据预处理
import matplotlib.pyplot as plt
import numpy as np
import os  # 导入os模块，用于操作文件路径
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
from tqdm import tqdm  # 进度条模块
from time import sleep
import time

# 设置超参数
EPOCH = 35
BATCH_SIZE = 32
show_size = 12
LR = 0.01              # 学习率
# 设置随机种子
SEED = 2809

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(SEED)

class CustomDataset(Data.Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        super().__init__()  # 调用父类的构造函数
        self.image_dir = image_dir  # 图像数据的路径
        self.label_file = label_file  # 标签文本的路径
        self.transform = transform  # 数据预处理操作
        self.samples = self._load_samples()  # 加载数据集样本信息

    def _load_samples(self):
        samples = []  # 存储样本信息的列表
        with open(self.label_file, 'r') as f:  # 打开标签文本文件
            for line in f:  # 逐行读取标签文本文件中的内容
                image_name, label = line.strip().split('\t ')  # 根据逗号分隔每行内容，获取图像文件名和标签
                image_path = os.path.join(self.image_dir, image_name)  # 拼接图像文件的完整路径
                samples.append((image_path, label))  # 将图像路径和标签组成元组，加入样本列表
        return samples  # 返回样本列表

    def __len__(self):
        return len(self.samples)  # 返回数据集样本的数量

    def __getitem__(self, index):
        image_path, label = self.samples[index]  # 获取指定索引处的图像路径和标签
        image = Image.open(image_path).convert('L')  # 打开图像文件并将其转换为灰度图像
        if self.transform:  # 如果定义了数据预处理操作
            image = self.transform(image)  # 对图像进行预处理操作
        return image, label  # 返回预处理后的图像和标签

# 检查是否安装了CUDA，并且CUDA是否适用于你的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置图片数据路径和标签文本路径
train_image_dir = '../data_train/graphs/'  # 图像数据的路径
train_label_file = '../data_train/labels.txt'  # 标签文本的路径
test_image_dir = '../data_test/graphs/'  # 图像数据的路径
test_label_file = '../data_test/labels.txt'  # 标签文本的路径
show_image_dir = '../data_test/graphs_12_dilation/'
show_label_file = '../data_test/labels_12.txt'

# 定义数据预处理操作，根据需要添加其他预处理操作
transform = transforms.Compose([
    transforms.Resize((40, 40)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
])

# 创建自定义数据集实例
train_dataset = CustomDataset(train_image_dir, train_label_file, transform=transform)
test_dataset = CustomDataset(test_image_dir, test_label_file, transform=transform)
show_dataset = CustomDataset(show_image_dir, show_label_file, transform=transform)

print("Size of train dataset:")
print(train_dataset.__len__())
print("Size of test dataset:")
print(test_dataset.__len__())
print("Size of show dataset:")
print(show_dataset.__len__())

# 创建数据加载器
# train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator = torch.Generator().manual_seed(SEED))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
show_loader = Data.DataLoader(dataset=show_dataset, batch_size=show_size, shuffle=False)

# 激活函数为 ReLU
# 隐空间 4 个神经元
# 10层网络
for NLS in range(100,2100,100):
    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Linear(in_features=40*40, out_features=NLS),
            )
            self.decoder = nn.Sequential(
                nn.Linear(in_features=NLS, out_features=40*40),
                nn.Sigmoid(),       # compress to a range (0, 1)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # 创建自编码器对象并将模型移动到GPU
    model = AutoEncoder().to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss() # 损失函数为均方误差
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss_list = [] # 每个 epoch 的平均训练损失
    test_loss_list = [] # 每个 epoch 的平均测试损失
    loss_list = [] # 每个 epoch 中每个批次的训练损失
    time_list = [] # 每个 epoch 中每个批次训练完的时刻

    # 训练自编码器
    print("------------------ NLS = ", NLS, "------------------")
    start_time = time.time()
    for epoch in range(EPOCH):
        train_loss = 0
        count = 0
        for data in tqdm(train_loader):
            img, _ = data
            img = img.to(device)
            img = img.flatten(2, 3)
            # 前向传播
            output = model(img)
            loss = criterion(output, img)
            time_list.append(time.time() - start_time)
            loss_list.append(loss.data)
            count += 1
            train_loss += loss.data
            # 反向传播和优化器优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sleep(0.001)
        train_loss_list.append(train_loss / count)
        print("Epoch[{}/{}], loss:{:.4f}".format(epoch + 1, EPOCH, loss.data))

        # 每个epoch训练完之后进行测试
        dataiter = iter(test_loader)
        test_loss = 0
        count = 0
        # 循环遍历所有迭代器
        for j in range(len(iter(test_loader))):
            # 从迭代器中获取下一个批次的图像和标签
            images, labels = next(dataiter)
            images = images.flatten(2, 3)

            # 使用模型进行推断，处理获取的图像数据，并将结果保存在output变量中
            output = model(images.to(device))
            count += 1
            test_loss += criterion(output, images.to(device)).data
        test_loss_list.append(test_loss / count)

    with open("./epoch_train_test_param/AE_MSE_ReLU_0.01_"+str(NLS)+".csv","w", newline='') as csvfile:
        writer = csv.writer(csvfile)

        #先写入columns_name
        writer.writerow(["epoch","train_loss","test_loss"])
        #写入多行用writerows
        for i in range(EPOCH):
            writer.writerow([i+1, train_loss_list[i].item(), test_loss_list[i].item()])

    with open("./time_loss_param/AE_MSE_ReLU_0.01_"+str(NLS)+".csv","w", newline='') as csvfile:
        writer = csv.writer(csvfile)

        #先写入columns_name
        writer.writerow(["time","loss"])
        #写入多行用writerows
        for i in range(len(time_list)):
            writer.writerow([time_list[i], loss_list[i].item()])

    # 迭代测试数据集，生成迭代器
    dataiter = iter(show_loader)

    # 循环遍历所有迭代器
    for j in range(len(iter(show_loader))):
        # 从迭代器中获取下一个批次的图像和标签
        images, labels = next(dataiter)
        images = images.flatten(2, 3)

        # 使用模型进行推断，处理获取的图像数据，并将结果保存在output变量中
        output = model(images.to(device))
        output = output.reshape([show_size, 1, 40, 40])
        images = images.reshape([show_size, 1, 40, 40])

        # 创建子图和轴对象，其中第一行显示原始图像，第二行显示重构后的图像
        fig, axes = plt.subplots(nrows=2, ncols=show_size, sharex=True, sharey=True, figsize=(24,4))

        # 循环遍历迭代器中的12个图像，绘制原始图像和重构图像并添加标题
        for i in range(show_size):
            # 显示原始图像
            axes[0,i].imshow(images[i].squeeze().numpy(), cmap='gray')
            axes[0,i].set_title("Original")
            axes[0,i].get_xaxis().set_visible(False)
            axes[0,i].get_yaxis().set_visible(False)

            # 显示重构后的图像
            axes[1,i].imshow(output[i].squeeze().cpu().detach().numpy(), cmap='gray')
            axes[1,i].set_title("Reconstruction")
            axes[1,i].get_xaxis().set_visible(False)
            axes[1,i].get_yaxis().set_visible(False)

        plt.savefig("./pic_reconstruction/AE_MSE_ReLU_0.01_"+str(NLS)+"_ep35.png", dpi=300)
        # 显示生成的子图
        # plt.show()