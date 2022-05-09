import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from vine_datasaet.utils import read_split_data
from vine_datasaet.my_dataset import get_batch
from sklearn.metrics import accuracy_score
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from Net import CNN_LSTM, CNN_LSTM_Attention


start = time.time()

# 初始化参数
# root = 'D:\\workspace_py\\classicCNN\\vine_keyframes_test_3'
root = '/mntc/sst/vine_keyframes'
NUM_CLASSES = 22
EPOCH = 3
LR = 0.00001
BATCH_SIZE = 8

train_videos_path, train_videos_label, val_videos_path, val_videos_label = read_split_data(root)
num_train_data, num_val_data = len(train_videos_label), len(val_videos_label)

train_loader = get_batch(train_videos_path, train_videos_label, BATCH_SIZE, 'train')
val_loader = get_batch(val_videos_path, val_videos_label, BATCH_SIZE, 'val')

writer = SummaryWriter()

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
    print('Start ' + type(model).__name__ + '_' + type(model.cnn).__name__ + 'model training...')
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
        for batch_idx, (train_data, train_label, batch_length) in enumerate(train_loader):
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            # print(batch_length)
            # print(type(batch_length))
            if type(model).__name__ == 'CNN_LSTM':
                batch_length = batch_length.to(device)
                train_prediction = model(train_data, batch_length)
            elif type(model).__name__ == 'CNN_LSTM_Attention':
                train_prediction = model(train_data)
            # print(train_prediction)
            # print(train_data.shape, train_prediction.shape, train_label.shape)
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
        torch.save(model.state_dict(), type(model).__name__ + '_' + type(model.cnn).__name__ + '.pt')
        # 验证集验证
        loss_val, accuracy_val = test(model, val_loader)
        loss_val_list.append(loss_val)
        acc_val_list.append(accuracy_val)

        epoch_time = time.time()
        print(' | 运行时间为：%.2f' % (epoch_time - start))

    # 绘图
    my_plot(loss_train_list, loss_val_list, 'Loss: ' + type(model).__name__ + '_' + type(model.cnn).__name__)
    my_plot(acc_train_list, acc_val_list, 'Accuracy: ' + type(model).__name__ + '_' + type(model.cnn).__name__)



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
    loss_func = nn.CrossEntropyLoss()
    loss = 0.0
    accuracy = 0.0
    for batch_idx, (val_data, val_label, batch_length) in enumerate(val_loader):
        val_data = val_data.to(device)
        val_label = val_label.to(device)
        if type(model).__name__ == 'CNN_LSTM':
            batch_length = batch_length.to(device)
            val_prediction = model(val_data, batch_length)
        elif type(model).__name__ == 'CNN_LSTM_Attention':
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device))
    # 开始训练
    net = CNN_LSTM(num_classes=NUM_CLASSES, model_name='VGG16')
    net = net.to(device)
    train(net, train_loader)

    net = CNN_LSTM_Attention(num_classes=NUM_CLASSES, model_name='ResNet34')
    net = net.to(device)
    train(net, train_loader)

    end = time.time()
    print("程序的总运行时间为：%.2f" % (end - start))