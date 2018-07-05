import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import os

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def Test_Image(path,size_weight,size_height):
    Test_X=[]
    for filename in os.listdir(r"./Test"):
        image_path=os.path.join(path, filename)
        train_image=image = np.array(ndimage.imread(image_path, flatten=False))
        image = scipy.misc.imresize(train_image, size=(size_weight, size_height)).reshape((1, size_weight * size_height * 3)).T
        Test_X.append(image)  # 将图像数组添加到训练数组里
        # print(filename)

    X = np.array(Test_X).T  # 将训练数组转为矩阵行列X
    X = np.squeeze(X)  # 删去矩阵里为1的维度
    return X

path='./Test/'#路径 path='./image/'#路径
size_weight=64#图片的宽
size_height=64#图片的高

X=Test_Image(path,size_weight,size_weight)

W=np.load('W.npy')
b=np.load('b.npy')

#print("W",W.shape)
#print("X",X.shape)

Z=np.dot(W.T,X)+b
yi=sigmoid(Z)
print("yi:",yi)