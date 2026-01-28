from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_sentiment_analysis():
    print("Running Sentiment Analysis on Tweets with White Background...")
    os.makedirs('../images', exist_ok=True)
    os.makedirs('../dataset', exist_ok=True)
    os.makedirs('../outputs', exist_ok=True)
    
    tweets = [
        "I love this new product! Excellent experience.",
        "The service was terrible and slow.",
        "It's okay, not the best but works.",
        "Absolutely fantastic performance!",
        "Poor quality, I regret buying this.",
        "Decent for the price.",
        "Highly recommended for all professionals."
    ]
    
    analysis = []
    for t in tweets:
        blob = TextBlob(t)
        sentiment = blob.sentiment.polarity
        label = 'Positive' if sentiment > 0 else ('Negative' if sentiment < 0 else 'Neutral')
        analysis.append([t, sentiment, label])
        
    df = pd.DataFrame(analysis, columns=['Tweet', 'Score', 'Sentiment'])
    df.to_csv('../dataset/sentiment_results.csv', index=False)

    # Plot with White Background
    plt.figure(figsize=(10, 6), facecolor='white')
    df['Sentiment'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c', '#95a5a6'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.savefig('../images/sentiment_chart.png', facecolor='white')
    plt.close()

    with open('../outputs/execution_log.txt', 'w') as f:
        f.write("Project: Twitter Sentiment Analysis\nStatus: Completed\n")

    print("Success: Sentiment analysis complete with white background.")

if __name__ == "__main__":
    run_sentiment_analysis()
