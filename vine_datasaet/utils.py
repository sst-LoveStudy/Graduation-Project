# 划分vine_keyframes训练集和验证集
# input: 根路径root, 划分率(默认0.2)
# output: 训练集、训练集标签、验证集、验证集标签的列表
# 每个列表中存储的是路径、标签索引
# 本数据集每个视频文件夹下面最多有7个关键帧
import os
import json
import pickle
import random
import matplotlib.pyplot as plt
from vine_datasaet.VineVideo import VineVideo

def read_split_data(root:str, val_rate:float=0.2):
    random.seed(0)
    assert os.path.exists(root), 'dataset root: {} does not exist.'.format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    scene_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    scene_class = ['General Entertainment', 'Art Museum']
    # 排序保证顺序一致
    scene_class.sort()
    # 生成类别名以及对应的数字索引0-21，存储到json格式文件中
    class_indexs = dict((k, v) for v, k in enumerate(scene_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indexs.items()), indent=4)
    with open('class_indexs.json', 'w') as json_file:
        json_file.write(json_str)   # 输出为class_indexs.json

    train_videos_path = []  # 存储训练集所有视频路径
    train_videos_label = []  # 存储训练集所有视频对应索引信息
    val_videos_path = []  # 存储验证集所有视频路径
    val_videos_label = []  # 存储验证集所有视频对应索引信息
    every_class_num = []    # 存储每个类别的样本总数
    supported = ['.jpg', '.JPG', '.png', '.PNG'] # 支持的文件后缀类型

    # 遍历每个类别文件夹下的视频子文件夹的图片
    max_num_frames = 0
    for cla in scene_class:
        cla_path = os.path.join(root, cla)
        videos = []
        for vid in os.listdir(cla_path):
            video_path = os.path.join(cla_path, vid)
            frames = os.listdir(video_path)
            frames_num = len(frames)    # 获取该视频文件夹下关键帧数目
            if frames_num == 0:
                print(vid)
            if max_num_frames < frames_num:
                max_num_frames = frames_num
            frames_list = [os.path.join(video_path, i) for i in os.listdir(video_path)
                  if os.path.splitext(i)[-1] in supported]  # 获取该视频文件夹下关键帧列表
            temp_video = VineVideo(vid, frames_num, frames_list)
            videos.append(temp_video)
            # 测试看实例化是否成功
            # if(temp_video.vid == '922900179101822976'):
            #     temp_video.display()

        # 获取该类别对应的索引
        video_class = class_indexs[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(videos))
        # 按输入的比例随机采样验证样本
        val_path = random.sample(videos, k=int(len(videos) * val_rate))

        for video_path in videos:
            if video_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_videos_path.append(video_path)
                val_videos_label.append(video_class)
            else:   # 否则存入训练集
                train_videos_path.append(video_path)
                train_videos_label.append(video_class)
    print('{} videos were found in the dataset.'.format(sum(every_class_num)))
    print('{} videos for training.'.format(len(train_videos_path)))
    print('{} videos for validation.'.format(len(val_videos_path)))
    print('max_num_frames =', max_num_frames)

    plot_video = False
    if plot_video:
        plt.bar(range(len(scene_class)), every_class_num, align='center')
        plt.xticks(range(len(scene_class)), scene_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v+5, s=str(v), ha='center')
        plt.xlabel('video class')
        plt.ylabel('number of videos')
        plt.title('scene class distribution')
        plt.show()

    return train_videos_path, train_videos_label, val_videos_path, val_videos_label

if __name__ == '__main__':
    train_videos_path, train_videos_label, val_videos_path, val_videos_label = \
        read_split_data('D:/workspace_py/classicCNN/vine_keyframes', 0.2)
    # print(train_videos_path)
    # print(train_videos_label)