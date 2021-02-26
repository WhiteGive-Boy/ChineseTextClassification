# ChineseTextClassification
中文文本分类 传统机器学习+深度学习

深度学习部分见https://github.com/649453932/Chinese-Text-Classification-Pytorch
## 目录及文件说明
bert_pretrain存放bert预训练的参数及模型

models存放深度学习对应的模型定义

News文件夹存放中文文本数据，数据处理结果及模型运行结果.其中data文件夹下为文本数据，默认为word级，即文件夹下的数据分好词以空格分割；char文件夹下则不需要分词。

pytorch_pretrained为官方定义的加载bert需要的模块

使用了三种传统机器学习方法 朴素贝叶斯 逻辑斯蒂回归 lightGBM

各方法定义见对应的py文件

datapro.py文件为文本数据预处理程序代码
