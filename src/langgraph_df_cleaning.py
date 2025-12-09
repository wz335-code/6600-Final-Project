import re
import unicodedata

# Optional: sentence segmentation (only used if sentence_tokenize=True)
import nltk
from nltk.tokenize import sent_tokenize


def clean_text(
    text: str,
    remove_titles: bool = True,
    sentence_tokenize: bool = False,
) -> str:
    """
    Clean CEFR-style reading texts for transformer fine-tuning.

    Steps:
    - Normalize Unicode
    - Normalize quotes and apostrophes
    - Remove "..." artifacts
    - Remove section titles / headings (optional)
    - Merge broken lines into flowing prose
    - Normalize whitespace
    - (Optional) sentence segmentation with NLTK
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1) Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2) Normalize punctuation / quotes
    text = (
        text.replace("’", "'")
            .replace("`", "'")
            .replace("“", '"')
            .replace("”", '"')
    )

    # 3) Remove ellipsis artifacts
    text = text.replace(" ...", " ")
    text = text.replace("...", " ")

    # 4) Split into non-empty stripped lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop blank lines

    def is_title_like(ln: str) -> bool:
        # Heuristic: short all-caps or title-case line with no end punctuation
        if not ln:
            return False
        if any(ch in ln for ch in ".?!"):
            return False
        words = ln.split()
        if len(words) <= 6 and ln.isupper():
            return True
        if len(words) <= 6 and ln.istitle():
            return True
        # Check capitalization ratio
        cap_count = 0
        alpha_words = 0
        for w in words:
            if not w:
                continue
            if w[0].isalpha():
                alpha_words += 1
                if w[0].isupper():
                    cap_count += 1

        if alpha_words == 0:
            return False

        cap_ratio = cap_count / alpha_words

        # If almost all words start with uppercase (e.g., "Schnauzer Dogs")
        if cap_ratio >= 0.75:
            return True
        
        return False

    filtered = []
    for ln in lines:
        if remove_titles and is_title_like(ln):
            # skip global headers like "NEW BOOKS THIS MONTH" or bare titles
            continue
        filtered.append(ln)

    # 5) Merge broken lines:
    #    if previous line doesn't end with punctuation, treat next as continuation.
    merged = []
    for ln in filtered:
        if merged and not merged[-1].rstrip().endswith((".", "?", "!", ":")):
            merged[-1] = merged[-1].rstrip() + ". " + ln.lstrip()
        else:
            merged.append(ln)

    # 6) Join into one block of text
    joined = " ".join(merged)

    # 7) Normalize whitespace
    joined = re.sub(r"\s+", " ", joined).strip()

    # 8) Optional: sentence tokenization (if NLTK available)
    if sentence_tokenize:
        sents = sent_tokenize(joined)
        joined = " ".join(sents)

    return joined