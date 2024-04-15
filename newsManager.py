import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
from transformers import pipeline, AutoTokenizer
import torch
from datetime import datetime
import numpy as np


# List of RSS feeds
feeds = [
    'http://www.lemonde.fr/rss/une.xml',
    'https://www.la-croix.com/RSS/UNIVERS_WFRA',
    'https://www.20minutes.fr/feeds/rss-une.xml',
    'https://www.lepoint.fr/24h-infos/rss.xml',
    'https://www.francetvinfo.fr/titres.rss',
    'https://rmc.bfmtv.com/rss/actualites/',
    'https://www.francebleu.fr/rss/a-la-une.xml',
    'https://www.lefigaro.fr/rss/figaro_actualites.xml',
]

sources = {
    'http://www.lemonde.fr/rss/une.xml': "Le Monde",
    'https://www.la-croix.com/RSS/UNIVERS_WFRA': "La Croix",
    'https://www.20minutes.fr/feeds/rss-une.xml': "20 Minutes",
    'https://www.lepoint.fr/24h-infos/rss.xml': "Le Point",
    'https://www.francetvinfo.fr/titres.rss': "France Info",
    'https://rmc.bfmtv.com/rss/actualites/' : "BFM TV",
    'https://www.francebleu.fr/rss/a-la-une.xml': "France Bleu",
    'https://www.lefigaro.fr/rss/figaro_actualites.xml': "Le Figaro"
}

def parse_feed(feed):
    try:
        news_feed = feedparser.parse(feed)
        return {
            "feed": feed,
            "entries": [
                {
                    "title": entry.title,
                    "published_at": entry.published,
                    "link": entry.link,
                    "content": fetch_article_content(entry.link),
                    "category": urlparse(entry.link).path.split('/')[1] if urlparse(entry.link).path.split('/') else ""
                }
                for entry in news_feed.entries
            ]
        }
    except Exception as e:
        print(f"Error parsing {feed}: {e}")
        return {"feed": feed, "entries": None}

def fetch_article_content(url):
    try:
        with requests.get(url) as response:
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            article_content = soup.find('article')
            return article_content.get_text(strip=True) if article_content else ""
    except:
        return ""

def get_articles():

    results = map(parse_feed, feeds)

    articles = []  # Initialize an empty list to store article details

    for result in results:
        if result["entries"]:
            for entry in result["entries"]:
                articles.append({
                    "title": entry["title"],
                    "published_at": entry["published_at"],
                    "source": sources[result["feed"]],
                    "link": entry["link"],
                    "content": entry["content"],
                    "category": entry["category"]
                })

    # Convert list of articles to DataFrame
    df = pd.DataFrame(articles)
    df['date'] = df['published_at'].apply(lambda x: x.split('+')[0])
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x.strip()[:-4], "%a, %d %b %Y %H:%M:%S") if x.endswith(" GMT") else datetime.strptime(x.strip(), "%a, %d %b %Y %H:%M:%S"))
    # Compute sentiment scores for each article content
    df["sentiment_label"] = df["content"].apply(get_sentiment_score)

    # Classify sentiment scores into categories
    conditions = [
        (df["sentiment_label"] == "1 star"),
        (df["sentiment_label"] == "2 stars"),
        (df["sentiment_label"] == "3 stars"),
        (df["sentiment_label"] == "4 stars"),
        (df["sentiment_label"] == "5 stars")
    ]
    choices = ["very negative", "negative", "neutral", "positive", "very positive"]
    df["sentiment_category"] = np.select(conditions, choices)

    return df


# Define a function to compute sentiment scores
def get_sentiment_score(text):

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base-sentiment")
    analyzer = pipeline(
        task='text-classification',
        model="cmarkea/distilcamembert-base-sentiment",
        tokenizer=tokenizer
    )

    # Get the maximum token length supported by the model
    max_length = tokenizer.model_max_length
    result = analyzer(text, truncation=True, max_length=max_length)
        
    return result[0]['label']