from PIL import Image
import torch
import numpy as np
from torchvision import transforms

# img_path = 'D:\\workspace_py\\classicCNN\\vine_keyframes_test_4\\Art Museum\\908852512239525888\\908852512239525888@@0.0-6.5.jpg'

# out = torch.randn(4, 3, 2)
# print(out)
# batch_length_list = [1, 2, 2, 2]
# a = []
# for i in range(out.shape[0]):
#     temp = (out[i][batch_length_list[i]]).tolist()
#     a.append(temp)
# # y = torch.from_numpy(np.array(y))
# print(a)
# a = torch.tensor(a)
#
# print(a)

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

val_losses = []
train_losses = []
train_losses = range(1, 100)

val_losses = range(101, 200)

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()