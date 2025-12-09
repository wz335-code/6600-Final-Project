import pandas as pd
import os

os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["FLAN_LOCAL_PATH"] = ""
os.environ["DEEPSEEK_MODEL"] = "deepseek-chat"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"


from langgraph_pipeline import simplify_chunk

# --- Settings ---
CSV_PATH = "guardian_articles_chunked.csv"
TEXT_COLUMN = "text"     # the column containing the text to simplify
LEVEL = "A2"             # CEFR target level

# Load CSV
df = pd.read_csv(CSV_PATH).iloc[100:250]  # limit to rows 100-250

# Create an output column
simplified = []

for text in df[TEXT_COLUMN]:
    text = str(text).strip()
    if not text:
        simplified.append("")
        continue

    out = simplify_chunk(level=LEVEL, chunk_text=text)
    simplified.append(out)

df["simplified_text"] = simplified

# Save new CSV
df.to_csv("my_output_simplified(A2_100_250).csv", index=False)

print("Done! Saved to my_output_simplified(A2_100_250).csv")

