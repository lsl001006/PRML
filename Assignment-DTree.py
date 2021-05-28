# Author : Li Shanglin
# StudentID : 18231088
# Date: 2021-05-28
# Teacher: Zengchang.Qin (Ph.D)
from math import log2, floor
from graphviz import Digraph


def dataReform():
    """
    读入iris dataset并将其转化为二维数组
    :return: data:整理为二维数组类型的数据集 150行*5列
    """
    with open('iris_data.txt', 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        tmp = data[i][:-1]
        tmp2 = tmp.split(',')
        data[i] = [float(tmp2[i]) for i in range(4)] + [tmp2[4]]
    return data


def entropy(data):
    """
    计算信息熵
    :param data:数据集
    :return: Information Entropy：信息熵
    """
    iris = [0, 0, 0]
    for i in range(len(data)):
        if data[i][-1] == 'Iris-setosa':
            iris[0] += 1
        elif data[i][-1] == 'Iris-versicolor':
            iris[1] += 1
        else:
            iris[2] += 1
    p = [0, 0, 0]
    e = [0, 0, 0]
    for i in range(len(p)):
        p[i] = iris[i] / len(data)
        if p[i] == 0:
            e[i] = 0  # 解决数学运算中的除零错误
        else:
            e[i] = -p[i] * log2(p[i])
    Info_ent = sum(e)
    return Info_ent


def findBestIG(data, k):
    """
    找出最佳信息增益对应的特征分类阈值
    :param data:拥有k个特征的数据集
    :param k: 特征个数
    :return: ideal_threshold:理想阈值数组 长度为k
             ideal_info_gain:每个特征对应的最佳信息增益
    """
    threshold = []  # 存放不同特征的分类阈值
    cData = [[] for _ in range(k)]  # cData分别存放每个特征下所有的数值，以研究在某一特征中信息增益最大的值(分类值)
    for i in range(k):
        cData[i] = [each[i] for each in data]
        threshold.append(min(cData[i]))
    ideal_threshold = [0] * k
    ideal_info_gain = [0] * k
    info_gain_list = [[] for _ in range(k)]
    # 对每一个特征，我们以0.1为步长逐渐从最小值增加，对每个数值进行信息增益计算，选取最大增益情况下对应的数值
    for i in range(k):
        for j in range(floor((max(cData[i]) - min(cData[i])) / 0.1) - 1):
            threshold[i] += 0.1
            info_gain_list[i].append(InfoGainFcn(data, threshold, k)[0][i])
    for i in range(k):  # 求出最佳分类阈值以及对应的信息增益
        ideal_info_gain[i] = max(info_gain_list[i])
        ideal_threshold[i] = (info_gain_list[i].index(ideal_info_gain[i]) + 1) * 0.1 + min(cData[i])

    return ideal_threshold, ideal_info_gain


def InfoGainFcn(data, threshold, k):
    """
    信息增益计算函数
    :param data: Reformed data
    :param k:特征个数
    :param threshold: k个特征的二分类阈值 e.g. threshold = [5.1, 3.5, 1.4, 0.2]
    :return:Info_gain:k个特征的信息增益
    """
    character = [[0, []] for _ in range(k)]
    dataClass1 = [[] for _ in range(k)]
    dataClass2 = [[] for _ in range(k)]
    Info_gain = [0] * k
    Info_ent = entropy(data)

    for i in range(len(data)):  # 求出每个特征对应的大于阈值的个数,以及对应data中的index
        for j in range(k):
            if data[i][j] > threshold[j]:
                character[j][0] += 1  # character[j]为iris的一种特征（共4种）
                character[j][1].append(i)
    for i in range(k):  # 给出大于阈值的数据dataClass1
        for j in range(len(character[i][1])):
            dataClass1[i].append(data[character[i][1][j]])
    # print(dataClass1[0])
    for i in range(k):  # 取data中关于dataClass1的补集dataClass2
        tmp = data.copy()
        for each in character[i][1]:
            tmp.pop(tmp.index(data[each]))
        dataClass2[i] = tmp
    # print(dataClass2[0])
    for i in range(k):  # 分别计算信息增益
        Info_gain[i] = Info_ent - 1.0 / len(data) * (character[i][0] * entropy(dataClass1[i]) +
                                                     (len(data) - character[i][0]) * entropy(dataClass2[i]))
    return [Info_gain, dataClass1, dataClass2]


def decisionTree(data, ck):
    """
    基于ID3算法的决策树
    :param data:需要进行决策树分类的数据集
    :param ck: 数据集特征数
    :return:
    """
    TreeDict = {'root': {}}
    threshold, infoGain = findBestIG(data, k=ck)
    # 选择信息增益最大的节点构建决策树的根节点
    root_index = infoGain.index(max(infoGain))
    print('Root is : {}   classification threshold is : {:.1f}   information gain is : {:.3f}'.format(
        characterName[root_index], threshold[root_index], infoGain[root_index]))
    characterName.pop(root_index)
    TreeGrow(data, threshold, ck, root_index)


def TreeGrow(data, threshold, n, parent_node_index):
    """
    决策树生长，迭代生成子节点
    :param data: 生长前需要的数据
    :param threshold: 生长前得到的阈值
    :param n: 生长前的特征数量
    :param parent_node_index: 父节点的最大信息增益特征索引
    :return:
    """
    if n > 1: # 如果特征数大于1, 则继续生长决策树, 否则退出生长
        ig, dClass1, dClass2 = InfoGainFcn(data, threshold, k=n)
        # 去掉dClass1的“低信息增益特征”数据,在“高信息增益特征”分类成功基础上向下生长。dClass1对应n个高于属性阈值的数据集
        nodeData = dClass1[parent_node_index]  # 利用“高信息增益特征”分类后剩余数据作为新的数据进行信息熵计算
        for i in range(len(nodeData)):
            nodeData[i].pop(parent_node_index)
        # tValue represents threshold value, infoGain represents the infoGain list
        tValue, infoGain = findBestIG(nodeData, k=n - 1)
        node_index = infoGain.index(max(infoGain))
        if infoGain[node_index] > 0.1:  # 如果信息增益较大，我们将对此节点进行扩展
            print('Node{} is : {}   classification threshold is : {:.1f}   information gain is : {:.3f}'.format(
                (5 - n), characterName[node_index], tValue[node_index], infoGain[node_index]))
            characterName.pop(node_index)
            TreeGrow(nodeData, tValue, n - 1, node_index)  # 递归扩展节点
        else:  # 如果信息增益过小，我们将不对此节点进行扩展，成为叶节点并结束分类
            print('---Warning! Information gain is too small to grow as a node---\n'
                  '---------------Decision Tree growth completed!----------------')


def plotTree(): # 在结果的基础上构建决策树的可视化图像
    g = Digraph('G', filename='DecisionTree.gv')
    g.node('root', label='ROOT')
    g.node('node1', label='node1')
    g.attr('node', shape='doublecircle')
    g.node('leaf1', label='Setosa',)
    g.node('leaf2', label='Versicolour')
    g.node('leaf3', label='Virginica')
    g.edge('root', 'node1', label='petal length > 1.9')
    g.edge('root', 'leaf1', label='petal length <= 1.9')
    g.edge('node1', 'leaf2', label='petal width > 1.7')
    g.edge('node1', 'leaf3', label='petal width <= 1.7')
    g.view()


if __name__ == '__main__':
    characterName = ['sepal length', 'sepal width', 'petal length', 'petal width']
    iris_data = dataReform()
    decisionTree(iris_data, 4)
    plotTree()
