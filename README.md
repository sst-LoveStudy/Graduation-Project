# Graduation-Project

东南大学本科毕业设计，采用VGG16-LSTM进行基于关键帧的视频场景识别

## CNN

1. 包含VGG16和ResNet34，输入格式[batch_size, 3, 160, 160]
2. 包含train和validation，并在训练完成后有绘图loss和acc

## CNN_LSTM_Attention

1. 包含VGG16, ResNet34和LSTM, LSTM_Attention的组合，共4种模型
2. 包含train和validation，并在训练完成后有绘图loss和acc

## vine_dataset

1. 自建定义Dataset和Dataloader
2. 以0.8：0.2划分训练集和验证集
