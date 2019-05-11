import numpy as np
def loadDataSet():
    #创建两个列表
    dataMat=[];labelMat=[]
    #打开文本数据集
    fr=open('testSet.txt')
    #遍历文本的每一行
    for line in fr.readlines():
        #对当前行去除首尾空格，并按空格进行分离
        lineArr=line.strip().split()
        #将每一行的两个特征x1，x2，加上x0=1,组成列表并添加到数据集列表中
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        #将当前行标签添加到标签列表
        labelMat.append(int(lineArr[2]))
    #返回数据列表，标签列表
    return dataMat,labelMat

#定义sigmoid函数
def sigmoid(inx):
    return 1.0/(1+np.e**(-inx))

#梯度上升法更新最优拟合参数
#@dataMatIn：数据集
#@classLabels：数据标签
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
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else: xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker='s')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]- weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()

    def stocGradAscent(dataMatrix, classLabels):
        # 为便于计算，转为Numpy数组
        dataMat = np.array(dataMatrix)
        # 获取数据集的行数和列数
        m, n = np.shape(dataMatrix)
        # 初始化权值向量各个参数为1.0
        weights = np.ones(n)
        # 设置步长为0.01
        alpha = 0.01
        # 循环m次，每次选取数据集一个样本更新参数
        for i in range(m):
            # 计算当前样本的sigmoid函数值
            h = sigmoid(dataMatrix[i] + weights)
            # 计算当前样本的残差(代替梯度)
            error = (classLabels[i] - h)
            # 更新权值参数
            weights = weights + alpha * error * dataMatrix[i]
        return weights





