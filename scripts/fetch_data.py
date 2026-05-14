import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
import ssl
import certifi
import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_query(categories: list[str], start_date: str, end_date: str) -> str:
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    date_query = f"submittedDate:[{start_date} TO {end_date}]"
    return f"({cat_query}) AND {date_query}"


def fetch_papers(query: str, max_results: int = 500) -> list[dict]:
    base_url = "http://export.arxiv.org/api/query?"
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )

    url = base_url + params
    print(f"Request : {url[:80]}...")

    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context) as response:
        xml_data = response.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title", namespace).text.strip()
        abstract = entry.find("atom:summary", namespace).text.strip()
        published = entry.find("atom:published", namespace).text.strip()
        paper_id = entry.find("atom:id", namespace).text.strip()

        papers.append(
            {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "published": published,
            }
        )

    return papers


def save_batch(papers: list[dict], batch_name: str, output_dir: str) -> None:
    output_path = Path(output_dir) / f"{batch_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"Saved : {output_path} ({len(papers)} papers)")


def main():
    config = load_config()
    categories = config["data"]["categories"]
    max_results = config["data"]["papers_per_batch"]
    output_dir = config["paths"]["data_raw"]

    batches = [
        ("batch_01", "20240101", "20240331"),
        ("batch_02", "20240401", "20240630"),
        ("batch_03", "20240701", "20240930"),
        ("batch_04", "20241001", "20241231"),
    ]

    for batch_name, start_date, end_date in batches:
        print(f"\n--- {batch_name} : {start_date} → {end_date} ---")
        query = build_query(categories, start_date, end_date)
        papers = fetch_papers(query, max_results)
        save_batch(papers, batch_name, output_dir)
        time.sleep(3)


if __name__ == "__main__":
    main()
