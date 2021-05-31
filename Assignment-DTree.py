# Author : Li Shanglin
# StudentID : 18231088
# Date: 2021-05-31
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
    if len(data) > 0:
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
    else:
        return 0


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
        if len(info_gain_list[i]) > 0:
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


def decisionTree(data, ck, t):
    """
    基于ID3算法的决策树
    :param data:需要进行决策树分类的数据集
    :param ck: 数据集特征数
    :param t: 生成子节点的最小信息增益阈值，默认为0.1
    :return:
    """
    threshold, infoGain = findBestIG(data, k=ck)
    # 选择信息增益最大的节点构建决策树的根节点
    root_index = infoGain.index(max(infoGain))
    PDict = TreeDict.setdefault(characterName[root_index] + ' : {:.1f}'.format(threshold[root_index]), {})
    print('Root is : {}   classification threshold is : {:.1f}   information gain is : {:.3f}'.format(
        characterName[root_index], threshold[root_index], infoGain[root_index]))
    TreeGrow(data, threshold, ck, root_index, PDict, tIG=t)
    Tree = TreePrune(TreeDict)
    return Tree


def TreeGrow(data, threshold, n, parent_node_index, ParentDict, tIG=0.1):
    """
    决策树生长，迭代生成子节点
    :param data: 生长前需要的数据
    :param threshold: 生长前得到的阈值
    :param n: 生长前的特征数量
    :param parent_node_index: 父节点的最大信息增益特征索引
    :param ParentDict: 父字典
    :param tIG: 生成子节点的最小信息增益阈值，默认为0.1
    :return:
    """
    ig, dClass1, dClass2 = InfoGainFcn(data, threshold, k=n)
    #  dClass1是高于阈值的数据集，由四个特征分别对应的高于阈值的数据组成
    # 去掉dClass1的“低信息增益特征”数据,在“高信息增益特征”分类成功基础上向下生长。dClass1对应n个高于属性阈值的数据集
    if len(dClass1) > 0 and len(dClass2) > 0:
        nodeData1 = dClass1[parent_node_index]  # 利用“高信息增益特征”分类后剩余数据作为新的数据进行信息熵计算
        nodeData2 = dClass2[parent_node_index]
        # tValue1 represents threshold value, infoGain1 represents the infoGain1 list
        if len(nodeData1) > 0:
            tValue1, infoGain1 = findBestIG(nodeData1, k=n)
            node_index1 = infoGain1.index(max(infoGain1))
            if sum(infoGain1) != 0:
                dict11 = ParentDict.setdefault(characterName[node_index1] +
                                               ' : {:.1f}'.format(tValue1[node_index1]), {})  # 拓展树字典
                print('infoGain1:', infoGain1)
                print('Node{} is : {}   classification threshold is : {:.1f}   information gain is : {:.3f}'.format(
                    'left', characterName[node_index1], tValue1[node_index1], infoGain1[node_index1]))
            else:
                ParentDict.setdefault('leaf', nodeData1[0][-1])
                print(nodeData1[0][-1])
        else:
            tValue1 = [0, 0, 0, 0]
            node_index1 = -1
            infoGain1 = [0, 0, 0, 0]
            print(nodeData2[0][-1])

        if len(nodeData2) > 0:
            tValue2, infoGain2 = findBestIG(nodeData2, k=n)
            node_index2 = infoGain2.index(max(infoGain2))
            if sum(infoGain2) != 0:
                if characterName[node_index2] not in ParentDict.keys():
                    dict21 = ParentDict.setdefault(characterName[node_index2] +
                                                   ' : {:.1f}'.format(tValue2[node_index2]), {})  # 拓展树字典
                else:
                    dict21 = ParentDict.setdefault(characterName[node_index2] + '1' +
                                                   ' : {:.1f}'.format(tValue2[node_index2]), {})
                print('infoGain2:', infoGain2)
                print('Node{} is : {}   classification threshold is : {:.1f}   information gain is : {:.3f}'.format(
                    'right', characterName[node_index2], tValue2[node_index2], infoGain2[node_index2]))
            else:
                if 'leaf' not in ParentDict.keys():
                    ParentDict.setdefault('leaf', nodeData2[0][-1])
                else:
                    ParentDict.setdefault('leaf1', nodeData2[0][-1])
                print(nodeData2[0][-1])
        else:
            tValue2 = [0, 0, 0, 0]
            node_index2 = -1
            infoGain2 = [0, 0, 0, 0]
            print(nodeData1[0][-1])

        print('--------------------------------------------------------------')

        if infoGain1[node_index1] > tIG:  # 如果信息增益较大，我们将对此节点进行扩展
            TreeGrow(nodeData1, tValue1, n, node_index1, dict11, tIG)  # 递归扩展节点
        else:
            p = [0, 0, 0]
            for each in nodeData1:
                if each[-1] == 'Iris-versicolor':
                    p[1] += 1
                elif each[-1] == 'Iris-virginica':
                    p[-1] += 1
                else:
                    p[0] += 1
            pIndex = p.index(max(p))
            ParentDict.setdefault(iris_name[pIndex] + ' with accuracy {}/{}'.format(max(p), sum(p)), 'leaf')

        if infoGain2[node_index2] > tIG:  # 如果信息增益较大，我们将对此节点进行扩展
            TreeGrow(nodeData2, tValue2, n, node_index2, dict21, tIG)  # 递归扩展节点
        else:  # 如果信息增益过小，我们将不对此节点进行扩展，成为叶节点并结束分类
            p = [0, 0, 0]
            for each in nodeData2:
                if each[-1] == 'Iris-versicolor':
                    p[1] += 1
                elif each[-1] == 'Iris-virginica':
                    p[-1] += 1
                else:
                    p[0] += 1
            pIndex = p.index(max(p))
            ParentDict.setdefault(iris_name[pIndex] + ' with accuracy {}/{}'.format(max(p), sum(p)), 'leaf')
            print('---Warning! Information gain is too small to grow as a node---\n'
                  '---------------Decision Tree growth completed!----------------')

    elif len(dClass2) == 0:
        print(dClass1[0][0][-1])
    else:
        print(dClass2[0][0][-1])


def TreePrune(tree):
    """
    除掉决策树的空枝
    :param tree:初步生成的决策树
    :return:
    """
    tmp = tree.copy()
    KEY = list(tmp.keys())
    for key in KEY:
        if tree[key] == {}:
            del tree[key]
        elif type(tree[key]) == dict:
            TreePrune(tree[key])
        elif key is 'leaf' or key is 'leaf1':
            del tree[key]
    return tree


def plotTree(treeDict, pnode):
    """
    绘制决策树
    :param treeDict: 字典类型存储的决策树
    :param pnode: 上一个节点的名称
    :return:
    """
    global Num
    root_name = str(Num)
    g.node(root_name, label=pnode)
    tmp = treeDict.copy()
    KEY = list(tmp.keys())
    for i in range(len(KEY)):  # 找出本层节点
        Num += 1
        g.node(str(Num), label=KEY[i])
        if type(tmp[KEY[i]]) is dict:  # 如果键值是字典的话则递归以穷尽字典
            g.edge(root_name, str(Num))
            plotTree(tmp[KEY[i]], KEY[i])
        else:
            g.edge(root_name, str(Num))


if __name__ == '__main__':
    TreeDict = {}  # 设置决策树字典全局变量
    Num = 0  # 节点编号
    iris_name = ['Setosa', 'Versicolor', 'Virginica']
    characterName = ['sepal length', 'sepal width', 'petal length', 'petal width']
    g = Digraph('G', filename='DecisionTree.gv')  # 初始化绘图
    iris_data = dataReform()  # 数据整理
    tree = decisionTree(iris_data, 4, t=0.32)  # 生成决策树字典
    plotTree(tree, 'root')  # 绘制决策树
    g.view()
