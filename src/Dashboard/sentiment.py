import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import csv
import tweepy 
import re 
import string
import os

import snscrape.modules.twitter as sntwitter
from contextlib import redirect_stdout
import time
import datetime as DT

todayDate = DT.date.today()

todayNow = todayDate.strftime("%Y-%m-%d")
weekAgo = todayDate - DT.timedelta(days=3)
weekAgo = weekAgo.strftime("%Y-%m-%d")



def grabTweets(company):
    # Create list to store/append twitter data to from snscrape
    tweet_list = []

    atUser = ""

    if company == 'Bank of America':
        atUser = "@BankOfAmerica"
    if company == 'Merrill Lynch':
        atUser = '@MerrillLynch'
    else:
        atUser = company

    #+ ' until:' + todayNow

    start_time = time.time()
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(atUser +  ' since:' + weekAgo).get_items()):
        tweet_list.append([tweet.date, tweet.id, tweet.content.replace("\n", ""), tweet.user.location.replace("\n", ""),
                        tweet.user.username])

    # Creates dataframe of scraped tweeets
    all_tweets = pd.DataFrame(tweet_list, columns=['Datetime', 'Tweet ID', 'Tweet', 'Location', 'Username'])

    # if company == 'Bank of America':
    #         # if file exists....
    #     if os.path.isfile('BoA-tweets.csv'):
    #         #Old data
    #         oldFrame = pd.read_csv('BoA-tweets.csv')

    #         #Concat
    #         df_diff = pd.concat([oldFrame, all_tweets],ignore_index=True).drop_duplicates()

    #         #Write new rows to csv file
    #         df_diff.to_csv('BoA-tweets.csv', header=True, index=False)

    #     else: # else it exists so append
    #         all_tweets.to_csv('BoA-tweets.csv', header=True, index=False)

    # Creates CSV file with BoA Tweets scraped from Twitter
    if company == 'Bank of America':
        all_tweets.to_csv('BoA-tweets.csv', sep=',', index=False)
    
    elif company == 'Merrill Lynch':
        all_tweets.to_csv('ML-tweets.csv', sep=',', index=False)
        # if os.path.isfile('ML-tweets.csv'):
        #     #Old data
        #     oldFrame = pd.read_csv('ML-tweets.csv')

        #     #Concat
        #     df_diff = pd.concat([oldFrame, all_tweets],ignore_index=True).drop_duplicates()

        #     #Write new rows to csv file
        #     df_diff.to_csv('ML-tweets.csv', header=True, index=False)

        # else: # else it exists so append
        #     all_tweets.to_csv('ML-tweets.csv', header=True, index=False)


grabTweets('Bank of America')
grabTweets('Merrill Lynch')





def __init__(self):
    self.df = []
    self.tweetTest = []




df = pd.read_csv('newSent.csv')

bankAmericaTweets = pd.read_csv('BoA-tweets.csv')
merillTweets = pd.read_csv('ML-tweets.csv')

timestampBOA = bankAmericaTweets.iloc[:,0]
boaTweetID = bankAmericaTweets.iloc[:,1]
boaTweets = bankAmericaTweets.iloc[:,2]
locations = bankAmericaTweets.iloc[:,3]
usernames = bankAmericaTweets.iloc[:,4]


timestampMer = merillTweets.iloc[:,0]
merTweetID = merillTweets.iloc[:,1]
merTweets = merillTweets.iloc[:,2]
locationsMer = merillTweets.iloc[:,3]
usernamesMer = merillTweets.iloc[:,4]




sentiment = df['Sentiment']
tweet = df['Tweet']

# replace the negative and positive sentiments with a 1 and 3 since there aren't any neutrals
# df['sentiment'] = df['sentiment'].replace('negative','1')
# df['sentiment'] = df['sentiment'].replace('positive','3')

#ngram_range(1,2) means unigram and bigram string values
#max_features is the size of vector
#Calculating the mean and variance of each tweets
#TfidfVectorizer tranform text into meaningful representation of numbers
tfidf = TfidfVectorizer(max_features=53, ngram_range=(3,3))
tweeted = tfidf.fit_transform(tweet)

##print(tweet.shape)
#prints out (233, 5082) = rows, columns in data

sentiment_train, sentiment_test, tweet_train, tweet_test = train_test_split(sentiment, tweeted, test_size=0.2, random_state=0)

##print(tweet_train.shape, tweet_test.shape)
#prints out (186, 5082) (47, 5082) = first one being trained and second one being tested

#LinearSVC classify the data
clf = LinearSVC()
clf.fit(tweet_train, sentiment_train)

#average/ratio of positive compare to negative sentiment
# sentiment_pred = clf.predict(tweet_test)
# print('Average sentiment prediction: ')
# print(sentiment_pred)

# print(' ')

# print('Classification report: ')
# print(classification_report(sentiment_test, sentiment_pred))



header = ['sentiment', 'tweet', 'service']
tweetsBank = []
sentiment = []
service = []


headerMer = ['sentiment', 'tweet', 'service']
tweetsLynch = []
sentimentLynch = []
serviceLynch = []



positive = '3'
negative = '1'
neutral = '2'


aTM = 'ATM'
mobileApp = " app "
mobile = "App"
online = "online"
other = "Other"
teller = 'teller'
bank = "Bank"


  
atmList = ['ATM', 'dispense','cash machine', 'atm machine']
mobileList = ['mobile','phone','app']
onlineList = ['online','web','site','website','web site', 'online banking']


for x in boaTweets:


    vec = tfidf.transform([x])
    clf.predict(vec)
    tweetsBank.append(x)

    # print(clf.predict(vec))
    if clf.predict(vec) == [3]:
        sentiment.append(positive)

    if clf.predict(vec) == [1]:
        sentiment.append(negative)

    if clf.predict(vec) == [2]:
        sentiment.append(neutral)



    if [ele for ele in atmList if(ele in x)]:
        service.append(aTM)

    elif [ele for ele in mobileList if(ele in x)]:
        service.append(mobile)

    elif [ele for ele in onlineList if(ele in x)]:
        service.append(online)
    else:
        service.append(other)



for x in merTweets:


    vec = tfidf.transform([x])
    clf.predict(vec)
    tweetsLynch.append(x)

    # print(clf.predict(vec))
    if clf.predict(vec) == [3]:
        sentimentLynch.append(positive)

    if clf.predict(vec) == [1]:
        sentimentLynch.append(negative)

    if clf.predict(vec) == [2]:
        sentimentLynch.append(neutral)
  

    if [ele for ele in atmList if(ele in x)]:
        serviceLynch.append(aTM)

    elif [ele for ele in mobileList if(ele in x)]:
        serviceLynch.append(mobile)

    elif [ele for ele in onlineList if(ele in x)]:
        serviceLynch.append(online)
    else:
        serviceLynch.append(other)

  


data = {'Sentiment':sentiment,'Service': service,'Tweet':tweetsBank, 'Timestamp': timestampBOA, 'Location':locations}

dataMer = {'Sentiment':sentimentLynch,'Service': serviceLynch,'Tweet':tweetsLynch, 'Timestamp': timestampMer, 'Location':locationsMer}


print(len(sentiment))
print(len(service))

outputBOA = pd.DataFrame(data)

outputBOA['Sentiment'] = outputBOA['Sentiment'].replace('1','negative')
outputBOA['Sentiment'] = outputBOA['Sentiment'].replace('3','positive')
outputBOA['Sentiment'] = outputBOA['Sentiment'].replace('2','neutral')

outputLynch = pd.DataFrame(dataMer)

outputLynch['Sentiment'] = outputLynch['Sentiment'].replace('1','negative')
outputLynch['Sentiment'] = outputLynch['Sentiment'].replace('3','positive')
outputLynch['Sentiment'] = outputLynch['Sentiment'].replace('2','neutral')


# if os.path.isfile('outputBank.csv'):
#     #Old data
#     oldFrame = pd.read_csv('outputBank.csv')

#     #Concat
#     df_diff = pd.concat([oldFrame, outputBOA],ignore_index=True).drop_duplicates()

#     #Write new rows to csv file
#     df_diff.to_csv('outputBank.csv', header=True, index=False)

# else: # else it exists so append
outputBOA.to_csv('outputBank.csv', header=True, index=False)



# if os.path.isfile('outputLynch.csv'):
#     #Old data
#     oldFrame = pd.read_csv('outputLynch.csv')

#     #Concat
#     df_diff = pd.concat([oldFrame, outputLynch],ignore_index=True).drop_duplicates()

#     #Write new rows to csv file
#     df_diff.to_csv('outputLynch.csv', header=True, index=False)

# else: # else it exists so append

outputLynch.to_csv('outputLynch.csv', header=True, index=False)

# outputBOA.to_csv('outputBank.csv',mode='w',header=True,index=False)

# outputLynch.to_csv('outputLynch.csv',mode='w',header=True,index=False)


# outputted = pd.read_csv('output.csv')
# print(outputted.head())
# # df = df.drop([first_column],axis=1)
# # df.to_csv('output.csv',index=False)





#plot_confusion_matrix(clf, sentiment_test, tweet_test)
#plt.show()
 