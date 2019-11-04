# coding:utf-8
import numpy as np
import pandas as pd
import math
import random
from scipy.sparse.linalg import gmres, lgmres
from scipy.sparse import csr_matrix


def genData():
    data = pd.read_csv('test1/200509', header=None, sep='\t')
    data.columns = ['date', 'user', 'item', 'label']
    data.drop('date', axis=1, inplace=True)
    data = data[:10000]
    print
    "genData successed!"
    return data


def getUItem_label(data):
    UI_label = dict()
    for i in range(len(data)):
        lst = list(data.iloc[i])
        user = lst[0]
        item = lst[1]
        label = lst[2]
        addToMat(UI_label, (user, item), label)
    print
    "UI_label successed!"
    return UI_label


def addToMat(d, x, y):
    d.setdefault(x, []).append(y)


def SplitData(Data, M, k, seed):
    '''
    划分训练集和测试集
    :param data:传入的数据
    :param M:测试集占比
    :param k:一个任意的数字，用来随机筛选测试集和训练集
    :param seed:随机数种子，在seed一样的情况下，其产生的随机数不变
    :return:train:训练集 test：测试集，都是字典，key是用户id,value是电影id集合
    '''
    data = Data.keys()
    test = []
    train = []
    random.seed(seed)
    # 在M次实验里面我们需要相同的随机数种子，这样生成的随机序列是相同的
    for user, item in data:
        if random.randint(0, M) == k:
            # 相等的概率是1/M，所以M决定了测试集在所有数据中的比例
            # 选用不同的k就会选定不同的训练集和测试集
            for label in Data[(user, item)]:
                test.append((user, item, label))
        else:

            for label in Data[(user, item)]:
                train.append((user, item, label))
    print
    "splitData successed!"
    return train, test


def getTU(user, test, N):
    items = set()
    for user1, item, tag in test:
        if user1 != user:
            continue
        if user1 == user:
            items.add(item)
    return list(items)


def new_getTU(user, test, N):
    for user1, item, tag in test:
        if user1 != user:
            continue
        if user1 == user:
            if (user, item) not in user_items:
                user_items.setdefault((user, item), 1)
            else:
                user_items[(user, item)] += 1
    testN = sorted(user_items.items(), key=lambda x: x[1], reverse=True)[0:N]
    items = []
    for i in range(len(testN)):
        items.append(testN[i][0][1])
    # if len(items)==0:print "TU is None"
    return items


def Recall(train, test, AA, M, G, alpha, N, user_items):
    '''
    :param train: 训练集
    :param test: 测试集
    :param N: TopN推荐中N数目
    :param k:
    :return:返回召回率
    '''
    hit = 0  # 预测准确的数目
    totla = 0  # 所有行为总数
    for user, item, tag in train:
        tu = getTU(user, test, N)
        rank = GetRecommendation(AA, M, G, alpha, user, N, user_items)
        for item in rank:
            if item in tu:
                hit += 1
        totla += len(tu)
    print
    "Recall successed!", hit / (totla * 1.0)
    return hit / (totla * 1.0)


def Precision(train, test, AA, M, G, alpha, N, user_items):
    '''
    :param train:
    :param test:
    :param N:
    :param k:
    :return:
    '''
    hit = 0
    total = 0
    for user, item, tag in train:
        tu = getTU(user, test, N)
        rank = GetRecommendation(AA, M, G, alpha, user, N, user_items)
        for item in rank:
            if item in tu:
                hit += 1
        total += N
    print
    "Precision successed!", hit / (total * 1.0)
    return hit / (total * 1.0)


def Coverage(train, AA, M, G, alpha, N, user_items):
    '''
    计算覆盖率
    :param train:训练集 字典user->items
    :param test: 测试机 字典 user->items
    :param N: topN推荐中N
    :param k:
    :return:覆盖率
    '''
    recommend_items = set()
    all_items = set()
    for user, item, tag in train:
        all_items.add(item)
        rank = GetRecommendation(AA, M, G, alpha, user, N, user_items)
        for item in rank:
            recommend_items.add(item)
    print
    "Coverage successed!", len(recommend_items) / (len(all_items) * 1.0)
    return len(recommend_items) / (len(all_items) * 1.0)


def Popularity(train, AA, M, G, alpha, N, user_items):
    '''
    计算平均流行度
    :param train:训练集 字典user->items
    :param test: 测试机 字典 user->items
    :param N: topN推荐中N
    :param k:
    :return:覆盖率
    '''
    item_popularity = dict()
    for user, item, tag in train:
        if item not in item_popularity:
            item_popularity[item] = 0
        item_popularity[item] += 1
    ret = 0
    n = 0
    for user, item, tag in train:
        rank = GetRecommendation(AA, M, G, alpha, user, N, user_items)
        for item in rank:
            if item != 0 and item in item_popularity:
                ret += math.log(1 + item_popularity[item])
                n += 1
    if n == 0: return 0.0
    ret /= n * 1.0
    print
    "Popularity successed!", ret
    return ret


def CosineSim(item_tags, item_i, item_j):
    ret = 0
    for b, wib in item_tags[item_i].items():
        if b in item_tags[item_j]:
            ret += wib * item_tags[item_j][b]
    ni = 0
    nj = 0
    for b, w in item_tags[item_i].items():
        ni += w * w
    for b, w in item_tags[item_j].items():
        nj += w * w
    if ret == 0:
        return 0
    return ret / math.sqrt(ni * nj)


def Diversity(train, AA, M, G, alpha, N, item_tags, user_items):
    ret = 0.0
    n = 0
    for user, item, tag in train:
        rank = GetRecommendation(AA, M, G, alpha, user, N, user_items)
        for item1 in rank:
            for item2 in rank:
                if item1 == item2:
                    continue
                else:
                    ret += CosineSim(item_tags, item1, item2)
                    n += 1
    print
    "Diversity successed!", ret / (n * 1.0)
    return ret / (n * 1.0)


def buildGrapha(record):
    graph = dict()
    user_tags = dict()
    tag_items = dict()
    user_items = dict()
    item_tags = dict()
    for user, item, tag in record:
        if user not in graph:
            graph[user] = dict()
        if item not in graph[user]:
            graph[user][item] = 1
        else:
            graph[user][item] += 1

        if item not in graph:
            graph[item] = dict()
        if user not in graph[item]:
            graph[item][user] = 1
        else:
            graph[item][user] += 1

        if user not in user_items:
            user_items[user] = dict()
        if item not in user_items[user]:
            user_items[user][item] = 1
        else:
            user_items[user][item] += 1

        if user not in user_tags:
            user_tags[user] = dict()
        if tag not in user_tags[user]:
            user_tags[user][tag] = 1
        else:
            user_tags[user][tag] += 1

        if tag not in tag_items:
            tag_items[tag] = dict()
        if item not in tag_items[tag]:
            tag_items[tag][item] = 1
        else:
            tag_items[tag][item] += 1

        if item not in item_tags:
            item_tags[item] = dict()
        if tag not in item_tags[item]:
            item_tags[item][tag] = 1
        else:
            item_tags[item][tag] += 1

    print
    "buildGrapha successed!"

    return graph, user_items, user_tags, tag_items, item_tags


def buildMatrix_M(G):
    M = []
    for key in G.keys():
        lst = []
        key_out = len(G[key])
        for key1 in G.keys():
            if key1 in G[key]:
                w = G[key][key1]
                lst.append(w / (1.0 * key_out))
            else:
                lst.append(0)
        M.append(lst)
    print
    "buildMatrix_M successed!"
    return np.matrix(M)


def before_GetRec(M):
    n = M.shape[0]

    A = np.eye(n) - alpha * M.T

    data = list()
    row_ind = list()
    col_ind = list()
    for row in range(n):
        for col in range(n):
            if (A[row, col] != 0):
                data.append(A[row, col])
                row_ind.append(row)
                col_ind.append(col)
    AA = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    print
    "before_GetRec successed!"
    return AA


def GetRecommendation(AA, M, G, alpha, root, N, user_items):
    items = []
    vertex = G.keys()
    index = G.keys().index(root)
    n = M.shape[0]
    zeros = np.zeros((n, 1))
    zeros[index][0] = 1
    r0 = np.matrix(zeros)
    b = (1 - alpha) * r0

    r = gmres(AA, b, tol=1e-08, maxiter=1)[0]
    rank = {}
    for j in range(n):
        rank[vertex[j]] = r[j]
    li = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    for i in range(N):
        item = li[i][0]
        if '/' in item and item not in user_items[root]:
            items.append(item)
    return items


def evaluate(train, test, AA, M, G, alpha, N, item_tags, user_items):
    ##计算一系列评测标准

    recall = Recall(train, test, AA, M, G, alpha, N, user_items)
    precision = Precision(train, test, AA, M, G, alpha, N, user_items)
    coverage = Coverage(train, AA, M, G, alpha, N, user_items)
    popularity = Popularity(train, AA, M, G, alpha, N, user_items)
    diversity = Diversity(train, AA, M, G, alpha, N, item_tags, user_items)
    return recall, precision, coverage, popularity, diversity


if __name__ == '__main__':
    data = genData()
    UI_label = getUItem_label(data)
    (train, test) = SplitData(UI_label, 10, 5, 10)
    N = 20;
    max_depth = 50;
    alpha = 0.8
    G, user_items, user_tags, tag_items, item_tags = buildGrapha(train)
    M = buildMatrix_M(G)
    AA = before_GetRec(M)
    recall, precision, coverage, popularity, diversity = evaluate(train, test, AA, M, G, alpha, N, item_tags,
                                                                  user_items)
