import pandas as pd
from textblob import TextBlob

# Load stock data
stock_data = pd.read_csv('/content/TSLA.csv')  # Update with your dataset path

# Load sentiment data (example: Tweets or news headlines) 
sentiment_data =pd.read_csv('/content/tesla_final.csv') # Update with your dataset path

# Perform sentiment analysis
def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

sentiment_data['Sentiment'] = sentiment_data['title'].apply(get_sentiment_score)

# Aggregate sentiment by date (e.g., average sentiment for each day)
sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'], dayfirst=True)
aggregated_sentiment = sentiment_data.groupby('Date')['Sentiment'].mean().reset_index()

# Merge with stock data
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
merged_data = pd.merge(stock_data, aggregated_sentiment, on='Date', how='left')

# Fill missing sentiment values (if any)
merged_data['Sentiment'].fillna(0, inplace=True)

print(merged_data.head())
