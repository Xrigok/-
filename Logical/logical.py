import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import os


def Train_Image(pathCat,pathNoCat,size_weight,size_height):
    Train_X=[]#训练数组
    Train_Y=[]#标签

    #读取是猫的训练集
    for filename in os.listdir(r"./Cat"):
        image_path=os.path.join(pathCat, filename)#图片路径名
        train_image=image = np.array(ndimage.imread(image_path, flatten=False))#将突破转为数组
        image = scipy.misc.imresize(train_image, size=(size_weight, size_height)).reshape((1, size_weight * size_height * 3)).T#将三维数组转为一维数组
        Train_X.append(image)#将图像数组添加到训练数组里
        #print(filename)
        Train_Y.append([1])#添加训练标签

    #读取不是猫的训练集
    for filename in os.listdir(r"./NoCat"):
        image_path = os.path.join(pathNoCat, filename)
        train_image = image = np.array(ndimage.imread(image_path, flatten=False))
        image = scipy.misc.imresize(train_image, size=(size_weight, size_height)).reshape((1, size_weight * size_height * 3)).T
        Train_X.append(image)
        #print(filename)
        Train_Y.append([0])

    X = np.array(Train_X)#将训练数组转为矩阵行列X
    X = np.squeeze(X)#删去矩阵里为1的维度
    Y = np.array(Train_Y).T#将训练标签转为矩阵行列Y


    return X,Y

def sigmoid(z):
    s=1/(1+np.exp(-z))
    #print(s)
    return s

def propagate(w,b,X,Y,m):#传播梯度函数

    Z=np.dot(w.T,X)+b
    A=sigmoid(Z)

    cost = -np.average(np.dot(Y.T,np.log(A)) + np.dot((1 - Y).T,np.log(1 - A)))#代价函数

    dw=1/m*np.dot(X,(A-Y).T)
    db=1/m*np.sum(A-Y)

    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost = np.squeeze(cost)#删除cost中为1的那个维度
    assert(cost.shape == ())#cost为实数
    grads={'dw':dw,
           'db':db}
    return grads,cost

def optimize(w, b, X, Y, itera, learning_rate,m):
    costs=[]
    for i in range(itera):
        grads,cost=propagate(w,b,X,Y,m)#传播梯度函数
        dw=grads['dw']
        db=grads['db']
        w=w-learning_rate*dw#更新w
        b=b-learning_rate*db#更新b
        #print(w)
        if i%100==0:#将每百次的cost保存下来
            costs.append(cost)

    #保存w,b,dw,db
    params={'w':w,
            'b':b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs
def Logical():
    pathCat='./Cat/'#是猫的图片的路径
    pathNoCat = './NoCat/'  #不是猫的图片的路径
    size_weight=64#图片的宽
    size_height=64#图片的高
    X,Y=Train_Image(pathCat,pathNoCat,size_weight,size_height)#获取训练集X，Y

    train_set_x = (X / 1.).T#将X转为浮点数并倒置
    train_set_y = Y / 1.#将Y转为浮点数并倒置
    m = train_set_x.shape[1]#获取x的总数
    itera=2000#迭代次数
    learning_rate=0.005#学习率

    w=np.zeros((train_set_x.shape[0],1))#初始w

    b=0#将比初始为0

    paramenters,grades,costs=optimize(w,b,train_set_x,train_set_y,itera,learning_rate,m)
    w=paramenters["w"]
    b=paramenters["b"]

    print(X.shape)
    print(w.shape)
    #将数据存储到d
    return w,b
w,b=Logical()

#保存d
#np.save("W",w)
#np.save('b',b)
#print('w',d['w'])