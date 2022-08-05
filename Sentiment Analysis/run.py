from pymongo import MongoClient
from gensim.models import word2vec
import random
import os
import jieba
import re
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

def getDataFromMongoDB():
    '''
    The main function is to connect MongoDB database and obtain relevant data from it
    :return:
    '''

    # Link MongoDB
    conn = MongoClient('127.0.0.1', 27017)
    db = conn.jdspider
    comments = db.getCommentsOnly

    # De duplication, and through the three item operator, the score of 3-5 points is set as positive (represented by 1), and the score of 1-2 points is set as negative (represented by 0)
    # When optimizing, you can consider the multi classification problem
    commentsData = list(set([(-1,-1) if ("此用户未填写评价内容" in comment["content"]) else (comment["content"],1 if(comment["score"]>=3) else 0) for comment in comments.find()]))

    return commentsData


def dataBalance(data):
    '''
    Balance the positive and negative samples. Here, downward sampling is used
    :param data:
    :return: Tuple type, the first element is positive, the second is negative
    '''

    positive = []
    negative = []
    for eveItem in data:
        if eveItem[1] == 1:
            positive.append(eveItem)
        elif eveItem[1] == 0:
            negative.append(eveItem)

    # Disrupt the sequence to prevent the data obtained during collection from being adjacent to similar products
    random.shuffle(positive)
    random.shuffle(negative)

    positiveNum = len(positive)
    negativeNum = len(negative)

    if positiveNum > negativeNum:
        positive = positive[0:len(negative)]
    else:
        negative = negative[0:len(positive)]

    return (positive,negative,(positiveNum,negativeNum))


def word2vecFun(textName, modelName):
    if os.path.exists(modelName):
        print("Model file already exists")
        print("Loading model file：%s"%(modelName))
        model = word2vec.Word2Vec.load(modelName)
    else:
        model = word2vec.Word2Vec(word2vec.LineSentence(textName), min_count=10, window=10)
        model.save(modelName)
    return model


def getStopWords(stopWordsName):
    with open(stopWordsName) as f:
        stopWords = [word.replace("\n", "") for word in f.readlines()]
    return stopWords


def cutWords(data,stopWords):
    dataCut = []
    for eveSentence in data:
        cutWord = jieba.cut("".join(re.findall(r'[\u4e00-\u9fa5]', eveSentence[0])))
        tempWord = []
        for eveWord in cutWord:
            if eveWord not in stopWords:
                tempWord.append(eveWord)
        dataCut.append((tempWord, eveSentence[1]))
    return dataCut


def getWordLen(data,percent):
    tempList = [len(eveItem[0]) for eveItem in data]
    listLen = len(tempList)
    tempCount = int(listLen * percent)
    tempList.sort()
    drawHist(tempList,'len', 'count')  # Histogram
    return tempList[tempCount]


def maxList(listData):
    temp = 0
    for i in listData:
        if listData.count(i) > temp:
            maxData = i
            temp = listData.count(i)
    return maxData


def drawHist(myList,Xlabel,Ylabel):
    '''
    The parameters are list, title, X-axis label, Y-axis label, and XY axis range
    :return:
    '''
    plt.hist(myList)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.show()

def word2Index(TEXTNAME):
    with open(TEXTNAME) as f:
        tempData = [eveSentence.strip() for eveSentence in f.readlines()]
    textList = []
    for eveSentence in tempData:
        for eveWord in eveSentence.split(" "):
            textList.append(eveWord)
    textList = list(set(textList))
    return textList

def getSentenceVec(sentence,count,textIndex):
    sentenceVec = np.zeros((count), dtype='int32')
    i = 0

    # Because the count value is taken here, the cutting is directly carried out here, which will lead to the decrease of accuracy
    # When optimizing, you can consider using TF IDF here to look at the "influence degree" of each word, and sort according to the influence degree (weight), taking the top count
    for eve in sentence[0:count]:
        try:
            sentenceVec[i] = textIndex.index(eve)
        except:
            sentenceVec[i] = -1

        i = i + 1

    return sentenceVec


def getIndexVectory(model):
    indexData = []
    vectoryData = []
    for eve in model.wv.vocab:
        indexData.append(eve)
        vectoryData.append(list(model[eve]))

    return (indexData, np.array(vectoryData))


# Data processing

TEXTNAME = "jd_comments_181007_cutted.txt"
MODELNAME = "jd_comments_181_7_model.model"
STOPWORDS = "StopwordsCN.txt"

print("STEP Model loading")
model = word2vecFun(TEXTNAME, MODELNAME)
print("STEP Index and word vector generating")
indexData, vectoryData = getIndexVectory(model)
print("STEP Stop words loading")
stopWords = getStopWords(STOPWORDS)
print("STEP Data loading")
data = getDataFromMongoDB()

print("STEP Data balancing")
positive,negative,dataCount = dataBalance(data)
print(dataCount)
print("STEP Word segmentation operating")
positiveCut = cutWords(positive,stopWords)
negativeCut = cutWords(negative,stopWords)

print("STEP Sentence length distribution obtaining")
userData = positiveCut + negativeCut
count = getWordLen(userData,0.80)


print("STEP Data Processing")
sentenceVec = []

for eveSentence in positiveCut:
    sentenceVec.append((getSentenceVec(eveSentence[0],count,indexData),eveSentence[1]))
for eveSentence in negativeCut:
    sentenceVec.append((getSentenceVec(eveSentence[0],count,indexData),eveSentence[1]))


# --------------------------------------------
# Start modeling training

print("STEP Modeling")
BATCHSIZE = 50
LSTMUNITS = 64
NUMCLASSES = 2
ITERATIONS = 50000
MAXSEQLENGTH = count
NUMDIMENSIONS = 300


# batch approach
def getTrainBatch(sentenceVec):
    labels = []
    arr = np.zeros([BATCHSIZE, MAXSEQLENGTH])
    for i in range(BATCHSIZE):
        tempData = random.choice(sentenceVec)
        if tempData[1] == 0:
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = np.array(tempData[0])

    return arr, labels

# line chart
def drawLine(list1,list2,title1,title2):

    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(list1)
    ax.set_title(title1)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(list2)
    ax.set_title(title2)
    plt.show()


tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [BATCHSIZE, NUMCLASSES])
inputData = tf.placeholder(tf.int32, [BATCHSIZE, MAXSEQLENGTH])

data = tf.Variable(tf.zeros([BATCHSIZE, MAXSEQLENGTH, NUMDIMENSIONS]),dtype=tf.float32)
data = tf.nn.embedding_lookup(vectoryData,inputData+1)

lstmCell = tf.contrib.rnn.BasicLSTMCell(LSTMUNITS)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([LSTMUNITS, NUMCLASSES]))
bias = tf.Variable(tf.constant(0.1, shape=[NUMCLASSES]))
value = tf.transpose(value, [1, 0, 2])
#Take the final result value
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


print("STEP Start training")
lossList = []
accuracyList = []
for i in range(ITERATIONS):

    nextBatch, nextBatchLabels = getTrainBatch(sentenceVec);
    sess.run(optimizer, {inputData: nextBatch, labels: nextBatchLabels})

    loss_ = sess.run(loss, {inputData: nextBatch, labels: nextBatchLabels})
    accuracy_ = sess.run(accuracy, {inputData: nextBatch, labels: nextBatchLabels})

    lossList.append(loss_)
    accuracyList.append(accuracy_)

    if (i % 1000 == 0 and i != 0):

        print("iteration {}/{}...".format(i + 1, ITERATIONS),
              "loss {}...".format(loss_),
              "accuracy {}...".format(accuracy_))

    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/%s.lstm_model"%(MODELNAME), global_step=i)
        print("saved to %s" % save_path)

drawLine(lossList,accuracyList,"LOSS LINE", "ACCURACY LINE")