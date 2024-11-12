from flask import Flask, request, jsonify
import feedparser
from transformers import pipeline

app = Flask(__name__)

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    ticker = data.get('ticker', 'AMD')
    keyword = data.get('keyword', 'AMD')

    # Lazy load the pipeline
    pipe = pipeline("text-classification", model="ProsusAI/finbert")

    rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0

    for entry in feed.entries:
        if keyword.lower() not in entry.summary.lower():
            continue
        truncated_summary = entry.summary[:512]
        sentiment = pipe(truncated_summary)[0]
        if sentiment['label'] == 'positive':
            total_score += sentiment['score']
            num_articles += 1
        elif sentiment['label'] == 'negative':
            total_score -= sentiment['score']
            num_articles += 1

    if num_articles > 0:
        final_score = total_score / num_articles
        sentiment_label = "Positive" if final_score > 0.15 else "Negative" if final_score <= -0.15 else "Neutral"
    else:
        sentiment_label = "Neutral"
        final_score = 0

    return jsonify({'final_score': final_score, 'sentiment_label': sentiment_label})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
