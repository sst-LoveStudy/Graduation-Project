import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

class VGG16LSTM(nn.Module):
    # 总共22个场景类别
    # test是4个场景类别
    def __init__(self, num_classes=22):
        super(VGG16LSTM, self).__init__()
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
        self.lstm = nn.LSTM(
            input_size=512 * 5 * 5,
            hidden_size=256,
            num_layers=3,
            batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        cnn_output_list = list()
        # print(x_3d.shape) [4, 5, 3, 160, 160]
        for t in range(x_3d.shape[1]):
            # print((x_3d[:, t, :, :, :]).shape)   [4, 3, 160, 160]
            pool1 = self.block1(x_3d[:, t, :, :, :])
            pool2 = self.block2(pool1)
            pool3 = self.block3(pool2)
            pool4 = self.block4(pool3)
            pool5 = self.block5(pool4)
            # print(pool5.shape)    [4, 512, 5, 5] 4是batch_size
            cnn_output = pool5.reshape(pool5.shape[0], -1)
            # print(pool5.shape)    [4, 512 * 5 * 5]
            cnn_output_list.append(cnn_output)
        x = torch.stack(tuple(cnn_output_list), dim=1)
        # print(x.shape)    [4, 5, 12800]
        out, (hn, cn) = self.lstm(x)
        # x = hn
        x = out[:, -1, :]
        # x = F.relu(x) # shape: [4, 256]
        x = self.fc1(x)
        x = F.relu(x) # shape: [4, 128]
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

    # def forward(self, x_3d):
    #     hidden = None
    #     for t in range(x_3d.shape[1]):
    #         with torch.no_grad():
    #             pool1 = self.block1(x_3d[:, t, :, :, :])
    #             pool2 = self.block2(pool1)
    #             pool3 = self.block3(pool2)
    #             pool4 = self.block4(pool3)
    #             pool5 = self.block5(pool4)
    #         out, hidden = self.lstm(pool5.unsqueeze(0), hidden)
    #
    #     x = out[:, -1, :]
    #     x = F.relu(x) # shape: [4, 256]
    #     x = self.fc1(x)
    #     x = F.relu(x) # shape: [4, 128]
    #     x = self.fc2(x)
    #     return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg_model=VGG16().to(device)
    # summary(vgg_model, (3, 160, 160)) #打印网络结构

    vgg16lstm = VGG16LSTM().to(device)
    # summary(vgg_model, (3, 160, 160))