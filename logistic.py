# -*- coding: UTF-8 -*-
import os
import random
import jieba
from sklearn.naive_bayes import MultinomialNB
#import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
# 手写拉普拉斯修正的朴素贝叶斯
import numpy as np
import pandas as pd



"""
    函数说明:中文文本处理
    Parameters:
        path - 文本存放的路径
        test_size - 测试集占比，默认占所有数据集的百分之20
    Returns:
        all_words_list - 按词频降序排序的训练集列表
        train_data_list - 训练集列表
        test_data_list - 测试集列表
        train_class_list - 训练集标签列表
        test_class_list - 测试集标签列表
"""
def TextProcessing(path, test_size=0.2):
#    folder_list = os.listdir(folder_path)  # 查看folder_path下的文件
    data_list = []  # 数据集数据
    class_list = []  # 数据集类别
    with open(path, 'r', encoding='utf-8') as f:  # 打开txt文件
        for line in f.readlines():
            line = line.strip().split("_!_")
            # print(line)
            if (len(line) >= 5):
                strr = line[3] + line[4]
            else:
                strr = line[3]
            word_cut = jieba.cut(strr, cut_all=False)  # 精简模式，返回一个可迭代的generator
            word_list = list(word_cut)  # generator转换为list
            data_list.append(word_list)
            class_list.append(line[2])

    data_class_list = list(zip(data_list, class_list))  # zip压缩合并，将数据与标签对应压缩
    random.shuffle(data_class_list)  # 将data_class_list乱序
    index = int(len(data_class_list) * test_size) + 1  # 训练集和测试集切分的索引值
    train_list = data_class_list[index:]  # 训练集
    test_list = data_class_list[:index]  # 测试集
    train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)  # 测试集解压缩

    all_words_dict = {}  # 统计训练集词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


"""
函数说明:读取文件里的内容，并去重
Parameters:
    words_file - 文件路径
Returns:
    words_set - 读取的内容的set集合
"""
def MakeWordsSet(words_file):
    words_set = set()  # 创建set集合
    with open(words_file, 'r', encoding='utf-8') as f:  # 打开文件
        for line in f.readlines():  # 一行一行读取
            word = line.strip()  # 去回车
            if len(word) > 0:  # 有文本，则添加到words_set中
                words_set.add(word)
    return words_set  # 返回处理结果


"""
函数说明:文本特征选取
Parameters:
    all_words_list - 训练集所有文本列表
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns:
    feature_words - 特征集
"""
def words_dict(all_words_list, deleteN, stopwords_set=set()):
    feature_words = []  # 特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度为1000
            break
            # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


"""
函数说明:根据feature_words将文本向量化
Parameters:
    train_data_list - 训练集
    test_data_list - 测试集
    feature_words - 特征集
Returns:
    train_feature_list - 训练集向量化列表
    test_feature_list - 测试集向量化列表
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    # for features in train_feature_list:
    #     for index in range(len(features)):
    #         features[index]=str(index)+"_"+str(features[index])
    # for features in test_feature_list:
    #     for index in range(len(features)):
    #         features[index]=str(index)+"_"+str(features[index])

    return train_feature_list, test_feature_list  # 返回结果


"""
函数说明:新闻分类器
Parameters:
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_class_list - 训练集分类标签
    test_class_list - 测试集分类标签
Returns:
    test_accuracy - 分类器精度
"""


if __name__ == '__main__':
    # 文本预处理
    folder_path = "./toutiao.txt"  # 训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,test_size=0.2)
    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    test_accuracy_list = []

    clf=LogisticRegression()

    id2class=['news_finance', 'news_story', 'news_travel', 'news_edu', 'news_military', 'news_game', 'news_agriculture', 'news_house', 'news_sports', 'news_car', 'news_tech', 'stock', 'news_entertainment', 'news_culture', 'news_world']

    class2id = {}
    index = 0
    for i in id2class:
        class2id[i] = index
        index = index + 1
    # print(id2class)
    train_class_list=[class2id[i] for i in train_class_list]
    test_class_list = [class2id[i] for i in test_class_list]

    feature_words = np.load("./feature_words.npy")
    feature_words = list(feature_words)


    #print(feature_words)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    print(train_feature_list[0])
    print(train_class_list[0])
    clf.fit(train_feature_list,train_class_list)
    savapath = "./News/saved_dict/"
    joblib.dump(clf, savapath+"./logr.m")

    predict_y=clf.predict(test_feature_list)
    print(classification_report(test_class_list, predict_y, target_names=id2class))
    # acc = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list,c1)
    # #print(c1.cc)
    # #print(c1.fc)
    # print("acc:",acc)
    # #print("predict lable:",lable)