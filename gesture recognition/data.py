import numpy as np
from skimage import io
import tensorflow as tf

batch_size = 64

data_x = []
data_y = []
import os
from PIL import Image

for filename in os.listdir("D:\\handpicture\\0"):
    im = Image.open('D:\\handpicture\\0\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(0)

for filename in os.listdir("D:\\handpicture\\1"):
    im = Image.open('D:\\handpicture\\1\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(1)

for filename in os.listdir("D:\\handpicture\\2"):
    im = Image.open('D:\\handpicture\\2\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(2)

for filename in os.listdir("D:\\handpicture\\3"):
    im = Image.open('D:\\handpicture\\3\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(3)
for filename in os.listdir("D:\\handpicture\\4"):
    im = Image.open('D:\\handpicture\\4\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(4)
for filename in os.listdir("D:\\handpicture\\5"):
    im = Image.open('D:\\handpicture\\5\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(5)
for filename in os.listdir("D:\\handpicture\\6"):
    im = Image.open('D:\\handpicture\\6\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(6)

for filename in os.listdir("D:\\handpicture\\7"):
    im = Image.open('D:\\handpicture\\7\\'+filename)
    im2 = np.array(im)
    data_x.append(im2)
    data_y.append(7)

for filename in os.listdir("D:\\handpicture\\8"):
    im = Image.open('D:\\handpicture\\8\\'+filename)
    im2 = np.array(im)
    data_x.append(im2)
    data_y.append(8)

for filename in os.listdir("D:\\handpicture\\9"):
    im = Image.open('D:\\handpicture\\9\\'+filename)
    im2 = np.array(im)
    data_x.append(im2)
    data_y.append(9)

data_x = np.array(data_x)
data_y = np.array(data_y)
print(data_x)
print(data_y)
print(data_x.shape, data_y.shape)