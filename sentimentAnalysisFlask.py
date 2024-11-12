from flask import Flask, render_template, request, jsonify
import feedparser
from transformers import pipeline
import json

app = Flask(__name__)

# Initialize the FinBERT pipeline
pipe = pipeline("text-classification", model="ProsusAI/finbert")

def analyze_sentiment(ticker, keyword):
    # Yahoo Finance RSS feed URL
    rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0
    results = []

    for entry in feed.entries:
        if keyword.lower() not in entry.summary.lower():
            continue

        # Truncate the summary to 512 tokens to avoid input size error
        truncated_summary = entry.summary[:512]

        sentiment = pipe(truncated_summary)[0]

        # Collect article details and sentiment
        results.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.summary,
            "sentiment": sentiment["label"],
            "score": sentiment["score"]
        })

        if sentiment['label'] == 'positive':
            total_score += sentiment['score']
            num_articles += 1
        elif sentiment['label'] == 'negative':
            total_score -= sentiment['score']
            num_articles += 1

    # Calculate final sentiment score
    if num_articles > 0:
        final_score = total_score / num_articles
        sentiment_label = "Positive" if final_score > 0.15 else "Negative" if final_score <= -0.15 else "Neutral"
    else:
        sentiment_label = "Neutral"
        final_score = 0

    return {
        "articles": results,
        "final_score": final_score,
        "sentiment_label": sentiment_label
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()  # Get JSON data
    ticker = data.get('ticker')  # Safely access 'ticker'
    keyword = data.get('keyword')  # Safely access 'keyword'

    if not ticker or not keyword:
        return jsonify({"error": "Ticker and keyword are required"}), 400

    result = analyze_sentiment(ticker, keyword)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
