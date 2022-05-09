from Net import VGG16, ResNet34
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from torch import nn
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

EPOCH = 10
NUM_CLASSES = 22
LR = 0.00001
BATCH_SIZE = 8
NUM_WORKER = 8

start = time.time()

data_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    # transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor() #将图片转换为Tensor,归一化至[0,1]
])

root = "/mntc/sst/vine_keyframes_img"

full_dataset = datasets.ImageFolder(root, transform=data_transform)
total_count = len(full_dataset)
num_train_data = int(0.8 * total_count)
num_val_data = total_count - num_train_data
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train_data, num_val_data])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
dataloaders = {
    "train": train_loader,
    "val": val_loader
}

def my_plot(in_train, in_val, title:str):
    plt.figure(figsize=(10, 10))
    plt.title("Training and Validation " + title)
    plt.plot(in_train, label="train")
    plt.plot(in_val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    plt.show()

def train(model, train_loader, epoch=EPOCH, batch_size=BATCH_SIZE, lr=LR):
    '''
    # 使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device))
    # 开始训练
    net = VGG16LSTM(num_classes=NUM_CLASSES)
    net = net.to(device)
    '''
    print('Start {} model training...'.format(type(model).__name__))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # 绘图用
    loss_train_list = []
    acc_train_list = []
    loss_val_list = []
    acc_val_list = []

    for e in range(epoch):
        model.train()
        loss = 0.0
        accuracy = 0.0
        for batch_idx, (train_data, train_label) in enumerate(train_loader):
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            train_prediction = model(train_data)

            _, predict = torch.max(train_prediction, dim=1)
            # predict = train_prediction.detach().to("cpu").numpy()
            predict = predict.to("cpu")
            true_label = train_label.detach().to("cpu").numpy()
            accuracy += accuracy_score(predict, true_label)

            batch_loss = loss_func(train_prediction, train_label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.data.cpu().numpy()

        loss = loss * batch_size / num_train_data
        accuracy = accuracy * batch_size / num_train_data
        print('Epoch: ', e, '\n',
              '| train loss: %.5f' % loss,
              '| train accuracy: %.2f' % accuracy)
        loss_train_list.append(loss)
        acc_train_list.append(accuracy)

        # 保存训练好的数据参数
        torch.save(model.state_dict(), type(model).__name__ + '.pt')
        # 验证集验证
        loss_val, accuracy_val = test(model, val_loader)
        loss_val_list.append(loss_val)
        acc_val_list.append(accuracy_val)

        epoch_time = time.time()
        print(' | 运行时间为：%.2f' % (epoch_time - start))

    # 绘图
    my_plot(loss_train_list, loss_val_list, 'Loss: ' + type(model).__name__)
    my_plot(acc_train_list, acc_val_list, 'Accuracy: ' + type(model).__name__)


def test(model, val_loader, batch_size=BATCH_SIZE):
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device))

    m_state_dict = torch.load('VGG16LSTM.pt')
    net = VGG16LSTM(NUM_CLASSES)
    net.load_state_dict(m_state_dict)
    # print(m_state_dict)
    net = net.to(device)
    '''
    model.eval()
    loss = 0.0
    loss_func = nn.CrossEntropyLoss()
    accuracy = 0.0
    for batch_idx, (val_data, val_label) in enumerate(val_loader):
        val_data = val_data.to(device)
        val_label = val_label.to(device)
        val_prediction = model(val_data)

        _, predict = torch.max(val_prediction, dim=1)
        # predict = train_prediction.detach().to("cpu").numpy()
        predict = predict.to("cpu")
        true_label = val_label.detach().to("cpu").numpy()
        accuracy += accuracy_score(predict, true_label)

        batch_loss = loss_func(val_prediction, val_label)
        loss += batch_loss.data.cpu().numpy()

    accuracy = accuracy * batch_size / num_val_data
    loss = loss * batch_size / num_val_data
    print(' | val loss: %.5f' % loss,
          '| val accuracy: %.2f' % accuracy)
    return loss, accuracy


if __name__ == '__main__':
    # 使用GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device))
    # 开始训练
    # VGG16
    net = VGG16(num_classes=NUM_CLASSES)
    net = net.to(device)
    train(net, train_loader)

    # ResNet34
    net = ResNet34(num_classes=NUM_CLASSES)
    net = net.to(device)
    train(net, train_loader)

    end = time.time()
    print("程序的总运行时间为：%.2f" % (end - start))