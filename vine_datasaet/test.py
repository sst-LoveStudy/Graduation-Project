from PIL import Image
import torch
from torchvision import transforms

# img_path = 'D:\\workspace_py\\classicCNN\\vine_keyframes_test_4\\Art Museum\\908852512239525888\\908852512239525888@@0.0-6.5.jpg'
#
# # transforms.ToTensor()
# transform1 = transforms.Compose([
#     transforms.Resize((240, 240)),
#     transforms.ToTensor() # range [0, 255] -> [0.0,1.0]
# ])
#
# ##PIL
# img = Image.open(img_path).convert('RGB') # 读取图像
# img2 = transform1(img) # 归一化到 [0.0,1.0]
# print("img2 = ", img2)
#

a = torch.rand((2,3))
print(a)
a = a.reshape(-1, 6)
print(a)