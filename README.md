# SentimentAnalysis
Emotion analysis based on deep learning (LSTM) (JD mall data)

## Purpose
Through the LSTM algorithm, the emotional analysis of e-commerce comments is realized.

## Procedures
* Analyze jd.com website and collect data through distributed crawlers
* Clean the collected data, including deleting duplicate data, deleting garbage data, etc
* Perform word segmentation, word stop and other operations on the cleaned data, and save the results to a new document
* The data after word segmentation is used to establish word vectors and index tables through word2vec
* Conduct data processing on the cleaned data, and set the scores of 1 and 2 as unsatisfactory, and the scores of 3, 4 and 5 as satisfactory
* Conduct data processing on the cleaned data, and set the scores of 1 and 2 as unsatisfactory, and the scores of 3, 4 and 5 as satisfactory
* The combination of word resounding and labels generates sample data for training
* Create a batch function
* LSTM modeling through RNN module in tensorflow
* LSTM modeling through RNN module in tensorflow
* Draw loss and accurate images

## Optimization opinions
* The process of converting collected data into sample data can be more reasonable. For example, keep the original 1-5 grade score as the emotional degree (satisfaction degree), change the existing two classification problem into a multi classification problem, and weight the evaluation through the "useful / useless" judged by other users on the comment, such as useful > useless, the emotional degree is deepened, otherwise the emotional degree is attenuated, which will make the sample data more scientific;
* When segmenting words and removing stop words, some punctuation marks and some modal particles are deleted, but in fact, these words are likely to seriously affect the expression of emotion, so when optimizing, we can consider dealing with this part of vocabulary alone or partially transforming it;
* When selecting the best sentence length, samples exceeding this length are cut, but in fact, this method may cut off some words that have a greater impact, so TF can be used here_ IDF to calculate the weight, and then sort the weight from high to low, and then cut according to the sorted vocabulary, so as to retain the characteristics of the original sentence as much as possible;

## Conclusion
Sentiment analysis is a very important work. It plays an important role in many fields, such as commodity satisfaction, film satisfaction, government satisfaction, or people's emotional orientation. This experiment collects data through large-scale distributed crawlers, obtains target data, and then carries out data processing. Word vectors and indexes are established through word2vec model. Through LSTM algorithm, After the model training, according to the final results, we can see that the whole experimental effect is good. The overall trend is towards the trend of increasing accuracy and decreasing loss, which is the basic goal of this experiment. However, this experiment also has some shortcomings, which have been listed in detail through the optimization opinions.

## Extra details
* This experiment mainly uses scratch redis to build a distributed crawler system, tensorflow to build an LSTM model, gensim to build a word2vec word vector, etc
* Contact: zhulei010414@gmail.com

## Route Map
* Overview of crawler crawling results (due to limited time, not too much data is crawled)
![首页展示](https://github.com/Vincent-Zhul/JD-Comments-Sentiment-Analysis/blob/main/Processing%20Picture/1.png)
* Details of crawler crawling results (part)
![首页展示](https://github.com/Vincent-Zhul/JD-Comments-Sentiment-Analysis/blob/main/Processing%20Picture/2.png)
* Corpus after cleaning
![首页展示](https://github.com/Vincent-Zhul/JD-Comments-Sentiment-Analysis/blob/main/Processing%20Picture/3.png)
* Corpus after word segmentation
![首页展示](https://github.com/Vincent-Zhul/JD-Comments-Sentiment-Analysis/blob/main/Processing%20Picture/4.png)
* Sentence length distribution in the sample
![首页展示](https://github.com/Vincent-Zhul/JD-Comments-Sentiment-Analysis/blob/main/Processing%20Picture/5.png)
* Loss and approximate diagram
![首页展示](https://github.com/Vincent-Zhul/JD-Comments-Sentiment-Analysis/blob/main/Processing%20Picture/6.png)
* Screenshot of operation results (part)
![首页展示](https://github.com/Vincent-Zhul/JD-Comments-Sentiment-Analysis/blob/main/Processing%20Picture/7.png)
