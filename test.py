# image to tensor  &&  tensor to image
# -*-coding:utf-8-*-
import torch
# from torchvision import transforms
# from PIL import Image
# import cv2
#
# import numpy as np
# np.set_printoptions(threshold=np.inf)
#
# img_path = "./cat.png"
#
# transform1 = transforms.Compose([
#     transforms.CenterCrop((224, 224)),  # 只能对PIL图片进行裁剪
#     transforms.ToTensor(),
# ]
# )
#
# # PIL图片与Tensor互转
# img_PIL = Image.open(img_path).convert('RGB')
# img_PIL.show()  # 原始图片
# img_PIL_Tensor = transform1(img_PIL)
# print(type(img_PIL))
#
#
# x=img_PIL_Tensor.numpy()
# with open('test.txt', 'w') as f:  # 打开test.txt   如果文件不存在，创建该文件。
#     # f.write(str(img_PIL_Tensor.data))  # 把变量var写入test.txt。这里var必须是str格式，如果不是，则可以转一下。
#     # f.write('\n\n')
#     f.write(str(x))
# print(img_PIL_Tensor.data)
#
# # Tensor转成PIL.Image重新显示
# new_img_PIL = transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')
# new_img_PIL.show()  # 处理后的PIL图片
#
# # opencv读取的图片与Tensor互转
# # transforms中，没有对np.ndarray格式图像的操作
# img_cv = cv2.imread(img_path)
# transform2 = transforms.Compose([
#     transforms.ToTensor(),
# ]
# )
#
# img_cv_Tensor = transform2(img_cv)
# print(type(img_cv))
# print(type(img_cv_Tensor))


from PIL import Image
import torchvision.transforms as tfs

im=Image.open('./cat.png')

# im.show()
# print(im.size)

# Resize
# new_im=tfs.Resize((100,255))(im)
# new_im.show()
# print(new_im.size)
# new_im.save('./new.png')

# random crop
# new_im=tfs.ColorJitter(brightness=2)(im)
# new_im.show()



# matplotlib.pyplot

# im_aug = tfs.Compose([
#     tfs.Resize(120),
#     # tfs.RandomHorizontalFlip(),
#     tfs.RandomCrop(96),
#     tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
# ])
#
# import matplotlib.pyplot as plt
#
# nrows = 3
# ncols = 3
# figsize = (10, 10)
# _, figs = plt.subplots(nrows, ncols, figsize=figsize)
# for i in range(nrows):
#     for j in range(ncols):
#         figs[i][j].imshow(im_aug(im))
#         figs[i][j].axes.get_xaxis().set_visible(False)
#         figs[i][j].axes.get_yaxis().set_visible(False)
# plt.show()

import numpy

