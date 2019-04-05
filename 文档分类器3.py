#-coding=utf-8
import os
import pandas as pd
import jieba
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    l_labels = []
    l_documents = []
    #os.walk返回三元组(root, dirs, files)
    #root指的是当前正在遍历的这个文件夹本身的地址
    #dirs是一个list，内容是该文件夹中所有的目录的名字
    #files是一个list，内容是该文件夹中所有的文件，不包含子目录
    for root, dirs, files in os.walk(path):
        print( root, dirs, files)
        for l_file in files:
            l_label = root.split('/')[-1]
            l_filepath = os.path.join(root, l_file)
            with open(l_filepath, 'r',encoding='gbk') as l_f:
                try:
                    l_content = l_f.read()
                except Exception as err:
                    print(err)
                    print(l_filepath)
                    continue
                
                l_words = ' '.join(list(jieba.cut(l_content)) )
                l_labels.append(l_label)
                l_documents.append(l_words)
    return l_documents, l_labels

#第一步：对文档进行分词
train_documents, train_labels = load_data('./text_classification-master/text classification/train/')
test_documents, test_labels = load_data('./text_classification-master/text classification/test/')

#第二步：加载停用词
STOP_WORDS = [line.strip() for line in open('./text_classification-master/text classification/stop/stopword.txt' ,'r').readlines()]

#第三步：计算单词的权重
tf = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5)
train_features = tf.fit_transform(train_documents)

#第四步：生成朴素贝叶斯分类器
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)

#第五步：使用生成的分类器做预测
test_tf = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5, vocabulary=tf.vocabulary_)
test_features = test_tf.fit_transform(test_documents)

predict_labels = clf.predict(test_features)

#第六步：计算准确率
print (metrics.accuracy_score(test_labels, predict_labels))
