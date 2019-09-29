import subprocess
import sys
import re
import json
import pandas as pd
import datetime as dt
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key='2b325124183e41499766717b8680c715')

def analyze(text):
    '''returns the polarity score for a chunk of text.'''
    text = re.sub(r'\W|https',' ',str(text))
    text = TextBlob(text)

    return text.sentiment.polarity


def newsUpdate(stock,f, api="rapid"):
    '''Returns a dictionary of articles with their date and polarity'''
    articles = {}
    links = {}

    # load the news articles
    if api == "rapid":
        json_dict = json.loads(f)
        # get the score and link for each article
        try:
            for article in json_dict['result']:

                publish_date = dt.date.fromtimestamp(article['published_at'])

                    # get the content from the article
                content = ''
                for k,v in article.items():
                    if k == 'title':
                        content += v
                    if k == 'summary':
                        content += v
                    if k == 'content':
                        content += v

                # now each article gets a score based on its content
                score = analyze(content)

                articles[publish_date] = score

                # now get the links with {title:url}
                links[article['title']] = article['link']
        except TypeError:
            return None, None

    return articles, links


def get_rapid_news(stock):

    # Get raw JSON file from ruby - news api not functional in pyhton 3.x :(
    file = subprocess.check_output(['ruby', 'stocks.rb', stock])

    articles, links = newsUpdate(stock, file)

    return articles, links

def news_api():
    # /v2/everything
    kwrds = "(renewable energy) OR (green energy) OR (solar energy) OR (solar) OR (environmental tarrifs) OR (alternative energy) OR (solar panel)"
    all_articles = newsapi.get_everything(q=kwrds,language='en',sort_by='relevancy')
    for article in all_articles['articles']:

        publish_date = article['publishedAt']
        content = ""
        for k,v in article.items():
            if k == 'title':
                content += v + " "
            if k == 'description':
                content += v + " "
            if k == "content":
                content += v + " "

        score = analyze(content)
        articles[publish_date] = score

        links[article["title"]] = article["url"]
    return articles, links
