import requests
import random
from datetime import date
import json

API_KEY = "3cee02ac-2d19-40c4-9b0f-e05844940d8d"
BASE_URL = "https://content.guardianapis.com/search"

SECTIONS = ["world", "business", "sport", "culture", "technology", "environment"]
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
ARTICLES_PER_SECTION_PER_YEAR = 10

random.seed(42)  # for reproducible sampling

def get_year_range(year):
    start = date(year, 1, 1).isoformat()
    end = date(year, 12, 31).isoformat()
    return start, end

def fetch_guardian_page(section, from_date, to_date, page=1, page_size=50):
    params = {
        "section": section,
        "from-date": from_date,
        "to-date": to_date,
        "order-by": "newest",
        "page-size": page_size,
        "page": page,
        "show-fields": "headline,byline,bodyText",
        "api-key": API_KEY,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json()["response"]

def fetch_guardian_year_section(section, year, max_pages=3):
    """Fetch up to max_pages * page_size articles for (section, year)."""
    from_date, to_date = get_year_range(year)
    articles = []

    response = fetch_guardian_page(section, from_date, to_date, page=1)
    articles.extend(response["results"])
    total_pages = response["pages"]

    # limit how many pages we actually walk to avoid too many API calls
    for page in range(2, min(total_pages, max_pages) + 1):
        response = fetch_guardian_page(section, from_date, to_date, page=page)
        articles.extend(response["results"])

    return articles

def build_balanced_sample():
    dataset = []

    for section in SECTIONS:
        for year in YEARS:
            print(f"Collecting {section} / {year}...")
            all_results = fetch_guardian_year_section(section, year, max_pages=3)

            # if fewer articles than we want, just take them all
            if len(all_results) <= ARTICLES_PER_SECTION_PER_YEAR:
                chosen = all_results
            else:
                chosen = random.sample(all_results, ARTICLES_PER_SECTION_PER_YEAR)

            # convert to a cleaner record format
            for item in chosen:
                fields = item.get("fields", {})
                record = {
                    "text": fields.get("bodyText"),  # only full article text
                }
                dataset.append(record)

    return dataset

def save_json(data, filename="guardian_articles.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} articles to {filename}")

if __name__ == "__main__":
    data = build_balanced_sample()
    print(f"Collected {len(data)} articles in total.")
    print(data[0])  # print the first article as a sample
    save_json(data)