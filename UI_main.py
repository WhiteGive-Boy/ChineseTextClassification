from tkinter import*
#初始化Tk()
import torch
import jieba
import numpy as np

from importlib import import_module
import pickle as pkl

myWindow = Tk()
#设置标题
myWindow.title('中文新闻分类')
var = StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
var.set("此处输出预测类别")
#创建一个标签，显示文本
on_hit = False
feature_words = np.load("./News/saved_dict/feature_words.npy")
feature_words = list(feature_words)
id2class = ['金融', '民生', '旅游', '教育', '军事',
            '游戏', '三农','房产', '体育', '汽车',
            '科技', '证券', '娱乐', '文化','国际']


id2class2=['体育', '旅游', '娱乐', '汽车', '文化', '游戏', '金融',
           '三农', '民生', '房产', '教育', '军事', '国际', '证券',
           '科技']

id2class3=['民生', '文化', '体育', '汽车', '科技', '金融', '证券',
           '军事', '游戏', '教育', '娱乐', '三农', '房产', '旅游',
           '国际']

dataset = 'News'  # 数据集
embedding = 'random'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_path = dataset + '/data/vocab.pkl'
vocab = pkl.load(open(vocab_path, 'rb'))
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
PAD2, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def countn(text, feature_words):
    result = 0
    for each in text:
        if (each == feature_words):
            result = result + 1
    return result
def text_features(text):  # 出现在特征集中，则置1
    global feature_words
    text_words = set(text)
    features = [countn(text,word) for word in feature_words]
    return features

def text_features2(text, pad_size=32):
    words_line = []
    token = text
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    words_line = torch.LongTensor(words_line).reshape(1,-1).to(device)
    seq_len = torch.LongTensor(seq_len).reshape(1,-1).to(device)

    return (words_line,seq_len)

def text_features3(text, pad_size=32):

    token = bertconfig.tokenizer.tokenize(text)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = bertconfig.tokenizer.convert_tokens_to_ids(token)

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    token_ids = torch.LongTensor(token_ids).reshape(1,-1).to(device)
    seq_len = torch.LongTensor(seq_len).reshape(1,-1).to(device)
    mask = torch.LongTensor(mask).reshape(1,-1).to(device)
    return (token_ids, seq_len, mask)

def hit_me():
    global Bayes,Lightgbm,LSTM,BERT,CNN,id2class,Logis
    modelv=model.get()
    strr=inputnews.get()
    word_cut = jieba.cut(strr, cut_all=False)  # 精简模式，返回一个可迭代的generator
    word_list = list(word_cut)  # generator转换为list

    if (modelv == 1):
        inputfea = text_features(word_list)
        inputfea=np.array(inputfea)
        inputfea=inputfea.reshape(1,-1)
        result=Bayes.predict(inputfea)
        var.set(id2class[result[0]])
        return
    if (modelv == 2):
        inputfea = text_features(word_list)
        inputfea=np.array(inputfea)
        inputfea=inputfea.reshape(1,-1)
        result=Logis.predict(inputfea)
        var.set(id2class[result[0]])
        return
    if (modelv == 3):
        inputfea = text_features(word_list)
        inputfea = np.array(inputfea)
        inputfea = inputfea.reshape(1, -1)
        result = Lightgbm.predict(inputfea)
        result = np.argmax(result, axis=-1)
        #print(result.shape)
        var.set(id2class[result[0]])
        return
    if (modelv == 4):
        inputfea = text_features2(word_list)

        outputs = CNN(inputfea)
        predic = torch.max(outputs.data, -1)[1].cpu().numpy()
        #print(predic)
        #print(predic.shape)
        var.set(id2class2[predic[0]])
        return
    if (modelv == 5):
        inputfea = text_features2(word_list)

        outputs = LSTM(inputfea)
        predic = torch.max(outputs.data, -1)[1].cpu().numpy()
        var.set(id2class2[predic[0]])
        return
    if (modelv == 6):
        inputfea = text_features3(strr)

        outputs = BERT(inputfea)
        predic = torch.max(outputs.data, -1)[1].cpu().numpy()
        var.set(id2class3[predic[0]])
        return

rowtop=0
coltop=0




from sklearn.naive_bayes import MultinomialNB

from sklearn.externals import joblib
Bayes=MultinomialNB()
Bayes=joblib.load("./News/saved_dict/bayes.m")

from sklearn.linear_model import LogisticRegression
Logis=LogisticRegression()
Logis=joblib.load("./News/saved_dict/logr.m")

import lightgbm as lgb
Lightgbm = lgb.Booster(model_file='./News/saved_dict/lightgbm.txt')

x = import_module('models.' + "TextRNN")
config = x.Config(dataset, embedding)
config.n_vocab = len(vocab)
LSTM=x.Model(config).to(config.device)
LSTM.load_state_dict(torch.load(config.save_path))


x = import_module('models.' + "TextCNN")
config = x.Config(dataset, embedding)
config.n_vocab = len(vocab)
CNN=x.Model(config).to(config.device)
CNN.load_state_dict(torch.load(config.save_path))


x = import_module('models.' + "bert")
bertconfig = x.Config(dataset)
BERT=x.Model(bertconfig).to(bertconfig.device)
BERT.load_state_dict(torch.load(bertconfig.save_path))





myWindow.geometry('500x250')  # 这里的乘是小x
Label(myWindow,text="输入一段新闻文本").grid(row = rowtop,column = coltop)
inputnews = StringVar()
# sb = Scrollbar(myWindow)
# sb.grid(row = rowtop,column = coltop)
news_entry = Entry(myWindow, bd =5, textvariable=inputnews,width=30)
news_entry.grid(row = rowtop,column = coltop+1)
b = Button(myWindow, text="类别预测", command=hit_me)
b.grid(row = rowtop,column = coltop+2)


L1 = Label(myWindow,text="选择一个算法：",justify=LEFT)
L1.grid(row = rowtop+1,column = 1,sticky='W')

model = IntVar()
model.set(1)
Radiobutton(myWindow,variable=model,text="Naive Bayes",value=1,justify=LEFT).grid(row = rowtop+2,column =1,sticky='W')
Radiobutton(myWindow,variable=model,text="Logistic",value=2,justify=LEFT).grid(row = rowtop+3,column = 1,sticky='W')
Radiobutton(myWindow,variable=model,text="LightGBM",value=3,justify=LEFT).grid(row = rowtop+4,column = 1,sticky='W')
Radiobutton(myWindow,variable=model,text="CNN",value=4,justify=LEFT).grid(row = rowtop+5,column = 1,sticky='W')
Radiobutton(myWindow,variable=model,text="LSTM",value=5,justify=LEFT).grid(row = rowtop+6,column = 1,sticky='W')
Radiobutton(myWindow,variable=model,text="Bert",value=6,justify=LEFT).grid(row = rowtop+7,column = 1,sticky='W')

L2 = Label(myWindow,text="输出类别：",justify=LEFT)
L2.grid(row = rowtop+3,column = 2,sticky='W')

classout = Label(myWindow,textvariable=var)
classout.grid(row = rowtop+4,column = 2,sticky='W')

#进入消息循环
myWindow.mainloop()