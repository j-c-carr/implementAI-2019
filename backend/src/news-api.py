from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key='2b325124183e41499766717b8680c715')
top_headlines = newsapi.get_top_headlines(sources="bbc-news, bloomberg",language='en')

# /v2/everything
kwrds = "(renewable energy) OR (green energy) OR (solar energy) OR (solar) OR (environmental tarrifs) OR (alternative energy) OR (solar panel)"
all_articles = newsapi.get_everything(q=kwrds,language='en',sort_by='relevancy')

print(all_articles)
