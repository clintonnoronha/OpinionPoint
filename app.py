from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import re
import matplotlib
import datetime
import seaborn as sns
import pandas as pd
import wordcloud as wd

nltk.download('words')
stop_words = set(nltk.corpus.words.words())

app = Flask(__name__)

if __name__ == "__main__":
    app.run()

@app.route("/")
def home():
    return render_template('index.html')

matplotlib.use('agg')

nltk.download('vader_lexicon') #required for Sentiment Analysis

# class with main logic
class SentimentAnalysis:
 
    def __init__(self):
        self.tweets = []
        self.tweetContent = []
 
    # This function first connects to the Tweepy API using API keys
    def DownloadData(self, keyword, tweets, fromDate, toDate):

        # passing the keyword, no. of tweets to their respective variables
        kw = keyword
        tweets = int(tweets)

        # passing the fromDate and toDay search to their respective variables
        fromDay = datetime.datetime.strptime(fromDate, '%Y-%m-%d')
        fromDay = fromDay.strftime('%Y-%m-%d')
        toDay = datetime.datetime.strptime(toDate, '%Y-%m-%d')
        toDay = toDay.strftime('%Y-%m-%d')
        

        # fetching tweets through snscrape for search keyword 'kw' from time duration requested
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(kw + ' lang:en since:' +  fromDay + ' until:' + toDay + ' -filter:links -filter:replies').get_items()):
            if i >= int(tweets):
                break
            self.tweets.append(tweet)


        # iterating through tweets fetched and cleaning text in tweetContent (the actual tweets)
        for tweet in self.tweets:
            self.tweetContent.append(self.cleanTxt(tweet.content))

        # initial reaction indicator variables
        compound = 0
        positive = 0
        negative = 0
        neutral = 0

        # Calculating sentiment score for each tweet
        for tweet in self.tweetContent:
            analyzer = SentimentIntensityAnalyzer().polarity_scores(tweet)
            neg = analyzer['neg']
            pos = analyzer['pos']
            comp = analyzer['compound']

            if (neg > pos):
                negative += 1 #increasing negative tweet count by 1
            elif (neg < pos):
                positive += 1 #increasing positive tweet count by 1
            elif (neg == pos):
                neutral += 1 #increasing neutral tweet count by 1

            # calculate sum to get average score later
            compound += comp


        # finding average of how people are reacting
        positive = self.percentage(positive, len(self.tweetContent))
        negative = self.percentage(negative, len(self.tweetContent))
        neutral = self.percentage(neutral, len(self.tweetContent))
 

        # finding average reaction
        compound = compound / len(self.tweetContent)
 
        if (compound <= -0.05):
            htmlcompound = "Negative :("
        elif (compound >= 0.05):
            htmlcompound = "Positive :)"
        else:
            htmlcompound = "Neutral :|"
 
        # call plotPieChart to generate pie chart visual
        self.plotPieChart(positive, negative, neutral, kw)        
        plt.clf()

        #To get all words from all the tweets
        all_words=[]
        for tweet in self.tweetContent:
            words = ''.join([i for i in tweet if not i.isdigit()])
            words = re.findall(r'\w+', words)
            for word in words:
                if word.lower() == keyword.lower() or word.lower() in stop_words or word.lower() == "has": continue
                all_words.append(word.lower())
        
        #Call the plotWordCount to generate bar graph visual
        self.plotWordCount(all_words)
        plt.clf()

        return compound, htmlcompound, positive, negative, neutral, keyword, len(self.tweetContent), fromDate, toDate
 

    # function for cleaning/ preprocessing of tweets
    def cleanTxt(self, text):
        text = re.sub('@[A-Za-z0-9]+', '', text) #Removing @mentions
        text = re.sub('#', '', text) # Removing '#' hash tag
        text = re.sub('RT[\s]+', '', text) # Removing RT
        text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
        return text
 

    # function to calculate percentage formatted upto 2 decimal digits
    def percentage(self, part, whole):
        return float(format(100 * float(part) / float(whole), '0.2f'))
 

    # function which sets and plots the pie chart. The chart is saved in an img file every time the project is run.
    def plotPieChart(self, positive, negative, neutral, query):
        labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]', 'Negative [' + str(negative) + '%]']
        sizes = [positive, neutral, negative]
        colors = ['yellowgreen', 'blue','red']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.style.use('default')
        plt.legend(patches, labels, loc="best")
        plt.title("Sentiment Analysis Result for keyword = "+ query + "")
        plt.axis('equal')
        plt.tight_layout()

        # set path to save pie chart plot
        strFile = r"static\img\plots\plot1.png"

        # Remove previous plot if present
        if os.path.isfile(strFile):
            os.remove(strFile)

        # save the pie chart plot
        plt.savefig(strFile)
 
    # function which sets and plots the pie chart. The chart is saved in an img file every time the project is run.
    def plotWordCount(self, words):
        data = dict()
        for word in words:
            word = word.lower()
            if word in stop_words:
                continue
            data[word] = data.get(word, 0) + 1

        #Sorting the dictionary so we can slice the first 10 values from it
        sorted_words = dict(sorted(data.items(), key = lambda x: x[1], reverse = True))
        sorted_words = {k: sorted_words[k] for k in list(sorted_words)[:20]}
        #Create and generate a pic of bar graph
        keys = list(sorted_words.keys())
        vals = [int(sorted_words[k]) for k in keys]
        sns.set(rc = {'figure.figsize':(20,20)})
        #plt.bar(range(len(sorted_words)), vals, tick_label=keys)
        dataset = {'Words Used': keys,
                'Count': vals} 
        new = pd.DataFrame.from_dict(dataset)
        ax = sns.barplot(
            x="Words Used", 
            y="Count",  
            data = new,
            palette=sns.color_palette("bright")
            )
        #ax = sns.barplot(x=keys,y=vals)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        plt.tight_layout()

        # set path to save pie chart plot
        strFile2 = r"static\img\plots\plot2.png"

        # Remove previous plot if present
        if os.path.isfile(strFile2):
            os.remove(strFile2)

        # save the plot
        plt.savefig(strFile2)

        # WordCloud Graph
        comment_words = ''
        comment_words += " ".join(words)+" "
        wordcloud = wd.WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stop_words,
                min_font_size = 10).generate(comment_words)
                   
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)

        # set path to save pie chart plot
        strFile3 = r"static\img\plots\plot3.png"

        # Remove previous plot if present
        if os.path.isfile(strFile3):
            os.remove(strFile3)

        # save the plot   
        plt.savefig(strFile3)


@app.route('/sentiment_logic', methods=['POST', 'GET'])
def sentiment_logic():

    # Get keyword to search and no. of tweets from html form enter by the user
    keyword = request.form.get('keyword')
    tweets = request.form.get('tweets')
    fromDate = request.form.get('fromDate')
    toDate = request.form.get('toDate')

    # object of SentimentAnalysis Class
    sa = SentimentAnalysis()

    # perform analysis and get all the resultant data
    # then set variables which can be used in the jinja supported html page
    compound, htmlcompound, positive, negative, neutral, keyword1, tweets1, fromDay, toDay = sa.DownloadData(keyword, tweets, fromDate, toDate)

    return render_template('sentiment_analyzer.html', compound=compound, htmlcompound=htmlcompound, positive=positive, 
                            negative=negative, neutral=neutral, keyword=keyword1, tweets=tweets1, fromDate=fromDay, toDate=toDay)


@app.route('/visualize')
def visualize():
    return render_template('PieChart.html')