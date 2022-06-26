import torch
import numpy as np
import os
from PIL import Image


img_size = (64, 64)
img_dir = './images'


img_path = sorted(os.listdir(img_dir))
images = []
for path in img_path:
    img = Image.open('data/' + path)
    img = img.resize(img_size).convert('RGB')
    images.append(np.array(img))

images = np.array(images)
images = images / 255.
images = images*2 - 1
images = np.transpose(images, (0, 3, 1, 2))
images = torch.tensor(images, dtype=torch.float32)
torch.save(images, 'img_tensor.pt')