from serpapi import GoogleSearch
import pandas as pd, time

API_KEY = "16f475f2fc92388a94f71653d083a892e3b0f873d263757059d063265fa78797"
AUTHOR_ID = "S2tBrxcAAAAJ"

all_articles, start = [], 0
while True:
    params = {
        "engine": "google_scholar_author",
        "author_id": AUTHOR_ID,
        "api_key": API_KEY,
        "hl": "en",
        "num": "100",
        "start": start,
        "sort": "pubdate"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    articles = results.get("articles", [])
    if not articles:
        break
    all_articles.extend(articles)
    start += len(articles)
    time.sleep(1)
df = pd.DataFrame([
    {
        "Title": a.get("title"),
        "Year": a.get("year"),
        "Link": a.get("link"),
        "Venue": a.get("publication"),
        "Authors": a.get("authors"),
        "Citations": a.get("cited_by", {}).get("value"),
        "Abstract": a.get("snippet"),
    }
    for a in all_articles
])
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.sort_values(by="Year", ascending=False)
df.to_excel("data.xlsx", index=False)
print("Scrapping Completed!")