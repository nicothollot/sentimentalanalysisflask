import streamlit as st
import feedparser
from transformers import pipeline

# Title of the Streamlit App
st.title("Sentiment Analysis of Financial News")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AMD):", value="AMD")
keyword = st.sidebar.text_input("Enter Keyword for Filtering:", value="AMD")

# Button to trigger analysis
if st.sidebar.button("Analyze"):
    # Initialize the FinBERT model
    st.text("Loading the sentiment analysis model...")
    pipe = pipeline("text-classification", model="ProsusAI/finbert")

    # Fetch RSS feed
    rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    # Sentiment Analysis
    st.text("Fetching and analyzing articles...")
    total_score = 0
    num_articles = 0
    results = []

    for entry in feed.entries:
        if keyword.lower() not in entry.summary.lower():
            continue

        truncated_summary = entry.summary[:512]
        sentiment = pipe(truncated_summary)[0]

        # Collect Results
        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "sentiment": sentiment["label"],
            "score": sentiment["score"],
            "link": entry.link
        })

        if sentiment["label"] == "positive":
            total_score += sentiment["score"]
            num_articles += 1
        elif sentiment["label"] == "negative":
            total_score -= sentiment["score"]
            num_articles += 1

    # Calculate Overall Sentiment
    if num_articles > 0:
        final_score = total_score / num_articles
        sentiment_label = (
            "Positive" if final_score > 0.15
            else "Negative" if final_score <= -0.15
            else "Neutral"
        )
    else:
        sentiment_label = "Neutral"
        final_score = 0

    # Display Results
    st.subheader(f"Overall Sentiment: {sentiment_label}")
    st.text(f"Sentiment Score: {final_score:.2f}")
    
    st.subheader("Articles Analyzed:")
    for result in results:
        st.write(f"**Title:** {result['title']}")
        st.write(f"**Summary:** {result['summary']}")
        st.write(f"**Sentiment:** {result['sentiment']} (Score: {result['score']:.2f})")
        st.write(f"[Read More]({result['link']})")
        st.write("---")
