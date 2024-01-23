import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用cuda进行加速卷积运算
    torch.backends.cudnn.benchmark = True
    # 载入训练集
    train_dataset = torchvision.datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(),
                                               download=True)
    test_dateset = torchvision.datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(),
                                              download=True)
    train_dataloder = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloder = DataLoader(test_dateset, batch_size=32, shuffle=True)


    # 构建教师模型
    class Teacher_model(nn.Module):
        def __init__(self, in_channels=1, num_class=10):
            super(Teacher_model, self).__init__()
            self.fc1 = nn.Linear(784, 1200)
            self.fc2 = nn.Linear(1200, 1200)
            self.fc3 = nn.Linear(1200, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x


    model = Teacher_model()
    model = model.to(device)

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    epoches = 6
    for epoch in range(epoches):
        model.train()
        for image, label in train_dataloder:
            image, label = image.to(device), label.to(device)
            optim.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            optim.step()

        model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for image, label in test_dataloder:
                image = image.to(device)
                label = label.to(device)
                out = model(image)
                pre = out.max(1).indices
                num_correct += (pre == label).sum()
                num_samples += pre.size(0)
            acc = (num_correct / num_samples).item()

        model.train()
        print("teacher-:{},accurate={}".format(epoch, acc))

    teacher_model = model


    # 构建学生模型
    class Student_model(nn.Module):
        def __init__(self, in_channels=1, num_class=10):
            super(Student_model, self).__init__()
            self.fc1 = nn.Linear(784, 20)
            self.fc2 = nn.Linear(20, 20)
            self.fc3 = nn.Linear(20, 10)
            self.relu = nn.ReLU()
            # self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.fc1(x)
            # x = self.dropout(x)
            x = self.relu(x)
            x = self.fc2(x)
            # x = self.dropout(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x


    model = Student_model()
    model = model.to(device)

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    epoches = 6
    for epoch in range(epoches):
        model.train()
        for image, label in train_dataloder:
            image, label = image.to(device), label.to(device)
            optim.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            optim.step()

        model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for image, label in test_dataloder:
                image = image.to(device)
                label = label.to(device)
                out = model(image)
                pre = out.max(1).indices
                num_correct += (pre == label).sum()
                num_samples += pre.size(0)
            acc = (num_correct / num_samples).item()

        model.train()
        print("epoches:{},accurate={}".format(epoch, acc))

        # 开始进行知识蒸馏算法
        teacher_model.eval()
        model = Student_model()
        model = model.to(device)
        # 蒸馏温度
        T = 7
        hard_loss = nn.CrossEntropyLoss()
        alpha = 0.3
        soft_loss = nn.KLDivLoss(reduction="batchmean")
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)

        epoches = 5
        for epoch in range(epoches):
            model.train()
            for image, label in train_dataloder:
                image, label = image.to(device), label.to(device)
                with torch.no_grad():
                    teacher_output = teacher_model(image)
                optim.zero_grad()
                out = model(image)
                loss = hard_loss(out, label)
                ditillation_loss = soft_loss(F.softmax(out / T, dim=1), F.softmax(teacher_output / T, dim=1))
                loss_all = loss * alpha + ditillation_loss * (1 - alpha)
                loss.backward()
                optim.step()

            model.eval()
            num_correct = 0
            num_samples = 0
            with torch.no_grad():
                for image, label in test_dataloder:
                    image = image.to(device)
                    label = label.to(device)
                    out = model(image)
                    pre = out.max(1).indices
                    num_correct += (pre == label).sum()
                    num_samples += pre.size(0)
                acc = (num_correct / num_samples).item()

            model.train()
            print("student--epoches:{},accurate={}".format(epoch, acc))





