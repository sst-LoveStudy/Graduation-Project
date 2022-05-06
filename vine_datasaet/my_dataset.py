# 重写init等三个方法
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import vine_datasaet.VineVideo
import os
from torchvision import transforms

class MyDataSet(Dataset):
    '''自定义数据集'''

    def __init__(self, video_path:list, video_class:list, transform=None):
        self.video_path = video_path
        self.video_class = video_class
        self.transform = transform

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, index):
        # 对每个video这个示实例完成构初始化
        # self.video_path[index]是VineVideo类型
        # v = self.video_path[index]
        # 输出的是图片的路径
        # print(v.vid)
        # print(v.frames_num)
        # print(v.frames_list)
        video = self.video_path[index]
        label = self.video_class[index]
        # print(video.vid)
        if(isinstance(video.frames_list[0],str)):
            temp_frames_list = []
            for frame in video.frames_list:
                img = Image.open(frame)
                if img.mode != 'RGB':
                    raise ValueError("image: {} isn't RGB mode".format(self.video_path[index]))
                temp_frames_list.append(img)
            video.frames_list = temp_frames_list
            # 对应标签

            if self.transform is not None:
                for i in range(video.frames_num):
                    video.frames_list[i] = self.transform(video.frames_list[i])

        return video, label

    @staticmethod
    def collate_fn(batch):
        videos, labels = tuple(zip(*batch))
        # 原教学视频中图片是（3，h，w）
        # 使用images=torch.stack(images, dim=0)是变成了(batch_size, 3, h, w)
        temp_videos_tensor = []
        for v in videos:
            temp_videos_tensor.append(torch.stack(v.frames_list, dim=0))
        videos = temp_videos_tensor
        # 现在每一个视频变成了4维tensor [帧数, 3, 160, 160]
        # 由于每个视频的关键帧数目不相同，在这里进行0 padding补至相同

        videos = pad_sequence(videos, batch_first=True) # 输入变成了(batch_size, t, 3, h, w), t指的是frames_num
        # videos = torch.stack(videos, dim=0)
        labels = torch.as_tensor(labels)
        return videos, labels

def get_batch(videos_path, videos_label, batch_size, str):
    # 开始装载数据
    data_transform = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(160), # 随机裁剪一个area然后再resize
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(), # 随机水平翻转
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化到[-1.0,1.0]
        ]),
        'val': transforms.Compose([
            transforms.Resize((160, 160)),
            # transforms.RandomResizedCrop(160),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    if str == 'train':
        dataset = MyDataSet(
            video_path=videos_path,
            video_class=videos_label,
            transform=data_transform['train']
        )
    else:
        dataset = MyDataSet(
            video_path=videos_path,
            video_class=videos_label,
            transform=data_transform['val']
        )

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(nw))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        collate_fn=dataset.collate_fn
    ) # 装载数据完成
    #  shuffle是指每个epoch都随机打乱数据排列再分batch，
    #  这里一定要设置成false，否则之前的排序会直接被打乱，
    #  drop_last是指不利用最后一个不完整的batch（数据大小不能被batch_size整除）
    return dataloader
