import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from vine_datasaet.utils import read_split_data
from vine_datasaet.my_dataset import get_batch
from VGG16LSTM import VGG16LSTM
from sklearn.metrics import accuracy_score
import time


start = time.time()

# 初始化参数
# root = 'D:\\workspace_py\\classicCNN\\vine_keyframes_test_3'
root = '/mntc/sst/vine_keyframes_test_3'
NUM_CLASSES = 2
EPOCH = 20
LR = 0.00005
BATCH_SIZE = 8

train_videos_path, train_videos_label, val_videos_path, val_videos_label = read_split_data(root)
num_train_data, num_test_data = len(train_videos_label), len(val_videos_label)
train_loader = get_batch(train_videos_path, train_videos_label, BATCH_SIZE, 'train')
val_loader = get_batch(val_videos_path, val_videos_label, BATCH_SIZE, 'val')

# def train_loop(dataloader, model, loss_fn, optimizer, which_model):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         if which_model == 1:
#             X = X.squeeze(1)
#         elif which_model == 2:
#             X = X.unsqueeze(1)
#         else:
#             pass
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
# def test_loop(dataloader, model, loss_fn, which_model):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0
#
#     with torch.no_grad():
#         for X, y in dataloader:
#             if which_model == 1:
#                 X = X.squeeze(1)
#             elif which_model == 2:
#                 X = X.unsqueeze(1)
#             else:
#                 pass
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
#           f"Avg loss: {test_loss:>8f} \n")

def train(model, train_loader, epoch=EPOCH, batch_size=BATCH_SIZE, lr=LR):
    '''
    # 使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device))
    # 开始训练
    net = VGG16LSTM(num_classes=NUM_CLASSES)
    net = net.to(device)
    '''
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    for e in range(epoch):
        model.train()
        accuracy = 0
        for batch_idx, (train_data, train_label) in enumerate(train_loader):
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            train_prediction = model(train_data)
            # print(train_prediction)
            # print(train_data.shape, train_prediction.shape, train_label.shape)
            _, predict = torch.max(train_prediction, dim=1)
            # predict = train_prediction.detach().to("cpu").numpy()
            predict = predict.to("cpu")
            true_label = train_label.detach().to("cpu").numpy()
            accuracy += accuracy_score(predict, true_label)

            # for i in range(batch_size):
            #     # print(i, '\n', train_prediction[i],'|', torch.argmax(train_prediction[i]),'|', train_label[i])
            #     if torch.argmax(train_prediction[i]) == train_label[i]:
            #         precision += 1
                #     print(train_prediction, 'precision=', precision)
                # else:
                #     print(train_prediction)

            loss = loss_func(train_prediction, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: ', e, '\n',
              '| train loss: %.5f' % loss.data.cpu().numpy(),
              '| train accuracy: %.2f' % (accuracy * batch_size / num_train_data))

        # 保存训练好的数据参数
        torch.save(net.state_dict(), 'VGG16LSTM.pt')
        # 验证集验证
        test(net, val_loader)

        epoch_time = time.time()
        print('| 运行时间为：%.2f' % (epoch_time - start))


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
    accuracy = 0
    for batch_idx, (val_data, val_label) in enumerate(val_loader):
        val_data = val_data.to(device)
        val_label = val_label.to(device)
        val_prediction = model(val_data)

        _, predict = torch.max(val_prediction, dim=1)
        # predict = train_prediction.detach().to("cpu").numpy()
        predict = predict.to("cpu")
        true_label = val_label.detach().to("cpu").numpy()
        accuracy += accuracy_score(predict, true_label)

        loss = loss_func(val_prediction, val_label)

    accuracy = accuracy * batch_size / num_test_data
    print('val loss: %.5f' % loss.data.cpu().numpy(),
          '| val accuracy: %.2f' % accuracy)


if __name__ == '__main__':
    # 使用GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device))
    # 开始训练
    net = VGG16LSTM(num_classes=NUM_CLASSES)
    net = net.to(device)

    train(net, train_loader)
    # test(net, val_loader)

    end = time.time()
    print("程序的总运行时间为：{}".format(end-start))