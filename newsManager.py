import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd

# List of RSS feeds
feeds = [
    'http://www.lemonde.fr/rss/une.xml',
    'https://www.la-croix.com/RSS/UNIVERS_WFRA',
    'https://www.20minutes.fr/feeds/rss-une.xml',
    'https://www.lepoint.fr/24h-infos/rss.xml',
    'https://www.francetvinfo.fr/titres.rss',
    'https://rmc.bfmtv.com/rss/actualites/'
]

sources = {
    'http://www.lemonde.fr/rss/une.xml': "Le Monde",
    'https://www.la-croix.com/RSS/UNIVERS_WFRA': "La Croix",
    'https://www.20minutes.fr/feeds/rss-une.xml': "20 Minutes",
    'https://www.lepoint.fr/24h-infos/rss.xml': "Le Point",
    'https://www.francetvinfo.fr/titres.rss': "France Info",
    'https://rmc.bfmtv.com/rss/actualites/' : "BFM TV"
}

def parse_feed(feed):

  try:
    NewsFeed = feedparser.parse(feed)  
    entries = []

    for entry in NewsFeed.entries:
        try:
            content = fetch_article_content(entry.link)
        except:
            content =""
        entries.append({"title": entry.title, 
                      "published_at": entry.published,
                      "link": entry.link,
                      "content": content})

    return {"feed": feed, "entries": entries}

  except Exception as e:
    print(f"Error parsing {feed}: {e}")
    return {"feed": feed, "entries": None}
  
def fetch_article_content(url):
  # Send a GET request to the URL
  response = requests.get(url)
  response.raise_for_status()  # Raise an exception for HTTP errors

  # Parse the response content with BeautifulSoup
  soup = BeautifulSoup(response.content, 'html.parser')

  # Extract the article content
  article_content = soup.find('article')
  paragraphs = article_content.find_all('p')

  # Extract text from each paragraph and join them
  article_text = ''.join(para.text for para in paragraphs)

  return article_text

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
                    "content": entry["content"]
                })

    # Convert list of articles to DataFrame
    df = pd.DataFrame(articles)

    return df