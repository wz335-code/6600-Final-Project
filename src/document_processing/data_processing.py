import json
import csv 

with open("guardian_articles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("guardian_articles.csv", "w", encoding="utf-8", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=list(data[0].keys()))
    writer.writeheader()
    writer.writerows(data)


