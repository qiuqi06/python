import numpy as np
def sigmoid(inx):
    return 1.0/(1+np.e**(-inx))
a=np.array([1,1,0])
e=np.array([[1,2],[1,3],[2,3]])

def gradAscent(dataMatIn,classLabels):
    #将数据集列表转为Numpy矩阵
    dataMatrix=np.mat(dataMatIn)
    # 将数据集标签列表转为Numpy矩阵，并转置
    labelMat=np.mat(classLabels).transpose()
    #获取数据集矩阵的行数和列数
    m,n=np.shape(dataMatrix)
    #学习步长
    alpha=0.001
    #最大迭代次数
    maxCycles=500
    #初始化权值参数向量每个维度均为1.0
    weights=np.ones((n,1))
    #循环迭代次数
    for k in range(maxCycles):
        #求当前的sigmoid函数预测概率
        h=sigmoid(dataMatrix*weights)
        #***********************************************
        #此处计算真实类别和预测类别的差值
        #对logistic回归函数的对数释然函数的参数项求偏导
        error=(labelMat-h)
        #更新权值参数
        weights=weights+alpha*dataMatrix.transpose()*error
        #***********************************************
    return weights
w=gradAscent(e,a)
print(w)
