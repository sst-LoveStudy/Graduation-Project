import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class VGG16Variant(nn.Module):
    def __init__(self):
        super(VGG16Variant, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # print((x_3d[:, t, :, :, :]).shape)   [8, 3, 160, 160]
        pool1 = self.block1(x)
        pool2 = self.block2(pool1)
        pool3 = self.block3(pool2)
        pool4 = self.block4(pool3)
        pool5 = self.block5(pool4)
        # print(pool5.shape)    [8, 512, 5, 5] 8是batch_size
        return pool5.reshape(pool5.shape[0], -1)

class ResidualBlock(nn.Module):
    # 显式的继承自nn.Module
    # resnet是卷积的一种
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        # shortcut是直连，resnet和densenet的精髓所在
        # 层的定义都在初始化里
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34Variant(nn.Module):
    # 包括34，50，101等多种结构，可以按需实现，这里是Resnet34
    def __init__(self, num_classes=22):
        super(ResNet34Variant, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),  # 这个64是指feature_num
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        short_cut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers = []
        layers.append(ResidualBlock(in_channel, out_channel, stride, short_cut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channel, out_channel))  # 输入和输出要一致
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 5)  # 注意F和原生的区别
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x



class CNN_LSTM(nn.Module):
    # 总共22个场景类别
    # test是4个场景类别
    def __init__(self, num_classes=22, model_name='VGG16'):
        super(CNN_LSTM, self).__init__()
        if model_name == 'VGG16':
            self.cnn = VGG16Variant()
        elif model_name == 'ResNet34':
            self.cnn = ResNet34Variant()
        self.lstm = nn.LSTM(
            input_size=512 * 5 * 5,
            hidden_size=256,
            num_layers=3,
            batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # batch_length: 每一组batch里面真实帧数
    def forward(self, x_3d, batch_length):
        cnn_output_list = list()
        # print(x_3d.shape) [8, 5, 3, 160, 160]

        # 用于计算length
        batch_length_list = batch_length.tolist()
        batch_size = len(batch_length_list)

        # cnn特征提取
        for t in range(x_3d.shape[1]):
            # print((x_3d[:, t, :, :, :]).shape)   [8, 3, 160, 160]
            cnn_output = self.cnn(x_3d[:, t, :, :, :])
            # print(pool5.shape)    [8, 512 * 5 * 5]
            cnn_output_list.append(cnn_output)
        x = torch.stack(tuple(cnn_output_list), dim=1)
        # print(x.shape)    [8, 5, 12800]
        out, (hn, cn) = self.lstm(x)
        # x = hn

        # 操作：x = out[:, length, :]
        # print(out.dtype)
        temp = [x-1 for x in batch_length_list]
        batch_length_list = temp
        # print(batch_length_list)
        x = []
        for i in range(batch_size):
            temp = (out[i][batch_length_list[i]]).tolist()
            x.append(temp)
        # x = torch.from_numpy(np.array(x))
        x = torch.tensor(x, dtype=torch.float32)
        x = x.cuda()
        # print(x.dtype)

        # x shape: [8, 256]
        x = self.fc1(x)
        x = F.relu(x) # shape: [8, 128]
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

class AttentionAggregator(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionAggregator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1),
        )
        self.__init_parameter()

    def __init_parameter(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, mask=None):
        """
        :parameter:
            x: input data, bills embedding or users embedding, [B, N ,D]
            mask: a group contains different number users. A user may support diff number bill. Use Mask to mask
             supernumerary data, [B, N, D]
        """
        out = torch.tanh(self.linear(x))
        # print(out.shape)
        if mask is None:
            weight = torch.softmax(out, dim=1)
        else:
            weight = torch.softmax(out + mask.unsqueeze(2), dim=1)
        # weight = F.softmax(out.view(1, -1), dim=1)
        ret = torch.matmul(x.transpose(2, 1), weight)
        return ret


class CNN_LSTM_Attention(nn.Module):
    def __init__(self, num_classes=22, model_name = 'VGG16'):
        super(CNN_LSTM_Attention, self).__init__()
        if model_name == 'VGG16':
            self.cnn = VGG16Variant()
        elif model_name == 'ResNet34':
            self.cnn = ResNet34Variant()
        self.lstm = nn.LSTM(
            input_size=512 * 5 * 5,
            hidden_size=256,
            num_layers=3,
            batch_first=True)
        self.att = AttentionAggregator(embedding_dim=256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        cnn_output_list = list()

        # cnn特征提取
        for t in range(x_3d.shape[1]):
            cnn_output = self.cnn(x_3d[:, t, :, :, :])
            cnn_output_list.append(cnn_output)
        x = torch.stack(tuple(cnn_output_list), dim=1)
        out, (hn, cn) = self.lstm(x)
        # x = hn

        # 使用Attention进行特征融合
        x = self.att(out)
        x = x.reshape(x.shape[0], -1)

        # x shape: [8, 256]
        x = self.fc1(x)
        x = F.relu(x) # shape: [8, 128]
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg_model=CNN().to(device)
    # summary(vgg_model, (3, 160, 160)) #打印网络结构

    vgg16lstm = CNN_LSTM_Attention(model_name='VGG16').to(device)
    print(type(vgg16lstm.cnn).__name__)
    resnet34lstm = CNN_LSTM_Attention(model_name='ResNet34').to(device)
    print(type(resnet34lstm.cnn).__name__)