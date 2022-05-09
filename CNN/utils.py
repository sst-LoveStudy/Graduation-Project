# 仅运行一次即可，已完成
# 根据视频文件夹，转化为Image文件夹
import os
import json
import random
import PIL.Image
import matplotlib.pyplot as plt

root = '/mntc/sst/vine_keyframes'
target = '/mntc/sst/vine_keyframes_img'

def move_data(root:str, target:str):

    # 遍历文件夹，一个文件夹对应一个类别
    scene_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序保证顺序一致
    scene_class.sort()
    # 生成类别名以及对应的数字索引0-21，存储到json格式文件中
    class_indexs = dict((k, v) for v, k in enumerate(scene_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indexs.items()), indent=4)
    with open('class_indexs.json', 'w') as json_file:
        json_file.write(json_str)   # 输出为class_indexs.json

    for cla in scene_class:
        cla_path = os.path.join(root, cla)
        target_folder = os.path.exists(os.path.join(target, cla))
        if not target_folder:
            os.makedirs(os.path.join(target, cla))
            print('new folder: ', os.path.join(target, cla))
        for vid in os.listdir(cla_path):
            video_path = os.path.join(cla_path, vid)
            frames = os.listdir(video_path)
            for frame in frames:
                frame_path = os.path.join(video_path, frame)
                img = PIL.Image.open(frame_path)
                img.save(os.path.join(target, cla, frame))
                # img.save(target+'/'+cla+'/'+frame)



def read_split_data(root:str, val_rate:float=0.2):
    random.seed(0)
    assert os.path.exists(root), 'dataset root: {} does not exist.'.format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    scene_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序保证顺序一致
    scene_class.sort()
    # 生成类别名以及对应的数字索引0-21，存储到json格式文件中
    class_indexs = dict((k, v) for v, k in enumerate(scene_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indexs.items()), indent=4)
    with open('class_indexs.json', 'w') as json_file:
        json_file.write(json_str)   # 输出为class_indexs.json

    train_images_path = []  # 存储训练集所有视频路径
    train_images_label = []  # 存储训练集所有视频对应索引信息
    val_images_path = []  # 存储验证集所有视频路径
    val_images_label = []  # 存储验证集所有视频对应索引信息
    every_class_num = []    # 存储每个类别的样本总数
    supported = ['.jpg', '.JPG', '.png', '.PNG'] # 支持的文件后缀类型

    for cla in scene_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indexs[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))


        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:   # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    print('{} images were found in the dataset.'.format(sum(every_class_num)))
    print('{} images for training.'.format(len(train_images_path)))
    print('{} images for validation.'.format(len(val_images_path)))

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

    return train_images_path, train_images_label, val_images_path, val_images_label

if __name__ == '__main__':
    train_images_path, train_images_label, val_images_path, val_images_label = \
        read_split_data(target, 0.2)
    # print(train_videos_path)
    # print(train_videos_label)