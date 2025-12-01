import requests
import base64
import json
import pandas as pd
import time
import os

USERNAME = ""
PASSWORD = ""
API_BASE = "https://textinspector.com/api/v2"

def create_session():
    auth_str = f"{USERNAME}:{PASSWORD}".encode()
    auth_header = base64.b64encode(auth_str).decode()

    r = requests.get(
        f"{API_BASE}/createsession",
        headers={
            "Authorization": f"Basic {auth_header}",
            "accept": "application/json",
        },
    )
    r.raise_for_status()
    data = r.json()
    print("create_session response:", data)
    return data["sessionid"]

def submit_texts(sessionid, texts, delimiter="#"):
    # IMPORTANT: delimiter is a single char; make sure it doesn't appear in texts
    cleaned = [t.replace(delimiter, " ") for t in texts]
    print(cleaned[:2])  # print first 2 cleaned texts as sample
    payload = {
        "text": delimiter.join(cleaned),
        "split": "1",           # multiple documents
        "delimiter": delimiter,
        "textmode": "Reading",  # or "Writing"/"Listening" if you prefer
    }

    r = requests.post(
        f"{API_BASE}/newanalysis",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "accept": "application/json",
            "Cookie": f"textinspector.session={sessionid}",
        },
    )
    r.raise_for_status()
    data = r.json()
    print("submit_texts response:", data)
    ctxId = data["response"]["ctxId"]
    # In your prints you saw both `documentCount` and in docs `doc_count`
    doc_count = (data["response"].get("documentCount")
        or data["response"].get("doc_count")
        or 1)
    return ctxId, doc_count

def get_cefr_for_all_docs(sessionid, ctxId, doc_count):
    """
    For each doc i in [1, doc_count], fetch:
      - lexical (EVP)
      - diversity
      - readability
      - metrics

    Returns:
      list of dicts, one per document.
    """
    endpoints = {
        "lexical_evp": "lexical",                # CEFR EVP
        "diversity": "diversity",                # VOCD, MTLD
        "metrics": "statistics",                 # general counts
        "scorecard": "scorecard",                # overall scorecard
        "lexical_coca": "taggedlexical-coca"
    }

    results = []

    headers = {
        "accept": "application/json",
        "Cookie": f"textinspector.session={sessionid}",
    }
    
    for i in range(1, doc_count + 1):
        doc_result = {
            "doc_index": i,
            "endpoints_raw": {},   # store raw JSON for each endpoint
            "evp_levels": None,
            "readability": None,
            "metrics": None,
            "diversity": None,
            "scorecard": None,
            "scorecard_overall": None,
            "coca": None
        }

        for key, endpoint in endpoints.items():
            url = f"{API_BASE}/{ctxId}/doc{i}/{endpoint}"
            try:
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                doc_result["endpoints_raw"][key] = r.json()
            except Exception as e:
                # store error instead of crashing
                doc_result["endpoints_raw"][key] = {
                    "error": str(e),
                    "url": url,
                    "status_code": getattr(r, "status_code", None),
                    "text": getattr(r, "text", None),
                }
            time.sleep(0.2)
    
        lexical = doc_result["endpoints_raw"].get("lexical_evp")
        evp_levels = []
        overall_rating = None

        try:
            if lexical and "response" in lexical and "summary" in lexical["response"]:
                summary = lexical["response"]["summary"]

                # per-level EVP breakdown
                for level in ["a1", "a2", "b1", "b2", "c1", "unlisted"]:
                    if level in summary:
                        block = summary[level]
                        evp_levels.append({
                            "code": level,
                            "level": block.get("friendlyName"),
                            "tokens": block["tokens"]["value"],
                            "tokens_percent": block["tokensPercent"]["value"],
                            "types": block["types"]["value"],
                            "types_percent": block["typesPercent"]["value"],
                        })

        except Exception as e:
            overall_rating = {
                "cefrLevel": None,
                "percentage": None,
                "raw": None,
                "error": f"failed to parse lexical_evp: {e}",
            }

        doc_result["evp_levels"] = evp_levels

        diversity = doc_result["endpoints_raw"].get("diversity")
        div_info = None
        try:
            if diversity and "response" in diversity:
                lex_div = diversity["response"].get("lexicalDiversity")
                if lex_div:
                    div_info = {
                        "vocd": lex_div["vocd"]["value"],
                        "mtld": lex_div["mtld"]["value"],
                    }
        except Exception as e:
            div_info = {"error": f"failed to parse diversity: {e}"}

        doc_result["diversity"] = div_info

        metrics_raw = doc_result["endpoints_raw"].get("metrics")
        metrics_info = None
        readability_info = None

        try:
            if metrics_raw and "response" in metrics_raw:
                resp = metrics_raw["response"]
                stats_summary = resp.get("summary", {})  # see "statistics" example in API PDF

                # Core statistics you might care about
                metrics_info = {
                    "avgSentenceLength": stats_summary.get("avgSentenceLength", {}).get("value"),
                    "sentenceCount": stats_summary.get("sentenceCount", {}).get("value"),
                    "tokenCount": stats_summary.get("tokenCount", {}).get("value"),
                    "typeCount": stats_summary.get("typeCount", {}).get("value"),
                    "typeTokenRatio": stats_summary.get("typeTokenRatio", {}).get("value"),
                    "syllableCount": stats_summary.get("syllableCount", {}).get("value"),
                    "averageSyllablesPerWord": stats_summary.get("averageSyllablesPerWord", {}).get("value"),
                    "averageSyllablesPerSentence": stats_summary.get("averageSyllablesPerSentence", {}).get("value"),
                    "syllablesPerHundredWords": stats_summary.get("syllablesPerHundredWords", {}).get("value"),
                    "wordsMoreThanTwoSyllables": stats_summary.get("wordsMoreThanTwoSyllables", {}).get("value"),
                    "wordsMoreThanTwoSyllablesPercentage": stats_summary.get("wordsMoreThanTwoSyllablesPercentage", {}).get("value"),
                }

                # Readability scores live under "readabilityScores" in the same response
                rs = resp.get("readabilityScores", {})
                readability_info = {
                    "fleschReadingEase": rs.get("fleschReadingEase", {}).get("value"),
                    "fleschKincaidGrade": rs.get("fleschKincaidGrade", {}).get("value"),
                    "gunningFogIndex": rs.get("gunningFogIndex", {}).get("value"),
                }
        except Exception as e:
            if metrics_info is None:
                metrics_info = {"error": f"failed to parse metrics/statistics: {e}"}
            if readability_info is None:
                readability_info = {"error": f"failed to parse readability scores: {e}"}

        doc_result["metrics"] = metrics_info
        doc_result["readability"] = readability_info

        scorecard_raw = doc_result["endpoints_raw"].get("scorecard")
        scorecard_overall = None
        scorecard_info = None

        try:
            if scorecard_raw and "response" in scorecard_raw:
                sc_resp = scorecard_raw["response"]

                # 1) overall rating from scorecard
                overall_sc = sc_resp.get("overallRating")
                if overall_sc:
                    scorecard_overall = {
                        "cefrLevel": overall_sc.get("cefrLevel"),
                        "percentage": overall_sc.get("percentage"),
                        "num_metrics_used": overall_sc.get("numberOfMetricsUsed"),
                        "raw": overall_sc,
                    }

                # 2) detailed scoreCard block (per-metric CEFR labels)
                sc_block = sc_resp.get("scoreCard", {})

                # You can pick out some especially interesting sub-metrics:
                lexical_soph = sc_block.get("lexicalSophistication", {})
                lexical_soph_acad = sc_block.get("lexicalSophisticationAcademic", {})
                lexical_div = sc_block.get("lexicalDiversity", {})
                stats_sc = sc_block.get("statistics", {})
                read_sc = sc_block.get("readability", {})

                scorecard_info = {
                    # overall CEFR label for lexical sophistication sub-dimension
                    "lexicalSophistication": {
                        "friendlyName": lexical_soph.get("friendlyName"),
                        "evpTypesA1_cefr": lexical_soph.get("evpPercentOfWordsTypesAtA1Level", {}).get("cefrLevel"),
                        "evpTypesB1_cefr": lexical_soph.get("evpPercentOfWordsTypesAtB1Level", {}).get("cefrLevel"),
                        "evpTokensB2_cefr": lexical_soph.get("evpPercentOfWordsTypesAtB2Level", {}).get("cefrLevel"),
                        "cocaMeanLfcPerType_cefr": lexical_soph.get("cocaMeanLfcPerType", {}).get("cefrLevel")
                    },
                    "lexicalSophisticationAcademic": {
                        "friendlyName": lexical_soph_acad.get("friendlyName"),
                        "awlList1TypesPercent_cefr": lexical_soph_acad.get("awlList1TypesPercent", {}).get("cefrLevel"),
                        "awlList1TokensPercent_cefr": lexical_soph_acad.get("awlList1TokensPercent", {}).get("cefrLevel"),
                        "awlTypes_cefr": lexical_soph_acad.get("academicWordListPercentOfAllAwlWordsTypesInTheText", {}).get("cefrLevel"),
                        "awlTokens_cefr": lexical_soph_acad.get("academicWordListPercentOfAllAwlWordsTokensInTheText", {}).get("cefrLevel")
                    },
                    "lexicalDiversity": {
                        "friendlyName": lexical_div.get("friendlyName"),
                        "mtld_cefr": lexical_div.get("lexicalDiversityMtld", {}).get("cefrLevel"),
                    },
                    "statistics": {
                        "averageWordsPerSentence_cefr": stats_sc.get("averageWordsPerSentence", {}).get("cefrLevel"),
                        "averageSyllablesPerWord_cefr": stats_sc.get("averageSyllablesPerWord", {}).get("cefrLevel"),
                    },
                    "readability": {
                        "friendlyName": read_sc.get("friendlyName"),
                        "fleschEase_cefr": read_sc.get("fleschKincaidReadingEase", {}).get("cefrLevel"),
                        "fleschGrade_cefr": read_sc.get("fleschKincaidReadingGrade", {}).get("cefrLevel"),
                        "gunningFog_cefr": read_sc.get("gunningFog", {}).get("cefrLevel"),
                    },
                }

        except Exception as e:
            if scorecard_overall is None:
                scorecard_overall = {"error": f"failed to parse scorecard overall: {e}"}
            if scorecard_info is None:
                scorecard_info = {"error": f"failed to parse scorecard details: {e}"}

        doc_result["scorecard_overall"] = scorecard_overall
        doc_result["scorecard"] = scorecard_info

        coca_raw = doc_result["endpoints_raw"].get("lexical_coca")
        coca_info = None

        try:
            if coca_raw and "response" in coca_raw:
                summary = coca_raw["response"].get("summary", {})

                # helper to safely get value and cast
                def get_val(key, default=None, cast=float):
                    item = summary.get(key)
                    if not item:
                        return default
                    v = item.get("value")
                    if v is None:
                        return default
                    try:
                        return cast(v)
                    except Exception:
                        return default

                coca_info = {
                    # size / coverage
                    "tokens_total":        get_val("totalTokenItems", default=None, cast=int),
                    "tokens_counted":      get_val("totalCountedTokensItems", default=None, cast=int),
                    "types_total":         get_val("uniqueItemTypes", default=None, cast=int),
                    "types_counted":       get_val("uniqueCountedItemTypes", default=None, cast=int),
                    "elements_incl_homonyms": get_val("elementsIncludingHomonyms", default=None, cast=int),
                    "coverage_percent":    get_val("coverage", default=None, cast=float),

                    # LFC aggregates (frequency-based difficulty)
                    "mean_lfc_per_token":  get_val("meanLFCPerToken"),
                    "mean_lfc_per_type":   get_val("meanLFCPerType"),
                    "mean_lfc_per_100_tokens": get_val("meanLFCPer100Tokens"),
                    "mean_lfc_per_100_types":  get_val("meanLFCPer100Types"),
                    "total_lfc_tokens":    get_val("totalLexicalTokensFrequencyCount"),
                    "total_lfc_types":     get_val("totalLexicalTypesFrequencyCount"),

                    # frequency percentiles (how common median / upper-quantile words are)
                    "p50_token_freq":      get_val("percentileTokens50th"),
                    "p60_token_freq":      get_val("percentileTokens60th"),
                    "p70_token_freq":      get_val("percentileTokens70th"),
                    "p80_token_freq":      get_val("percentileTokens80th"),

                    "p50_type_freq":       get_val("percentileTypes50th"),
                    "p60_type_freq":       get_val("percentileTypes60th"),
                    "p70_type_freq":       get_val("percentileTypes70th"),
                    "p80_type_freq":       get_val("percentileTypes80th"),
                }
        except Exception as e:
            coca_info = {"error": f"failed to parse lexical_coca: {e}"}

        doc_result["coca"] = coca_info

        metrics_resp = doc_result["endpoints_raw"].get("metrics", {})
        try:
            sentences = metrics_resp["response"]["sentences"]
            first_sentence = sentences[0]["sentence"]
        except Exception:
            first_sentence = "(no sentences)"

        print(f"[DEBUG] Doc {i}: first sentence from API:")
        print("   ", first_sentence)
        results.append(doc_result)
        time.sleep(0.2)  # to avoid hitting rate limits
    return results



def main():
    # 4.1 Read texts from CSV
    df = pd.read_csv("B2.csv")   # change file name as needed
    df = df.iloc[40:50] 
    # assumes a column named "text"
    texts = df["B2"].astype(str).tolist()

    # 4.2 Create session
    sessionid = create_session()
    print("Session:", sessionid)

    # 4.3 Submit all texts
    ctxId, doc_count = submit_texts(sessionid, texts)
    print("ctxId:", ctxId, "doc_count:", doc_count)

    # 4.4 Fetch all docs
    docs = get_cefr_for_all_docs(sessionid, ctxId, doc_count)

    # make sure docs are in order doc1, doc2, ...
    docs = sorted(docs, key=lambda d: d["doc_index"])


    output_path = "B2_results.json"

    if os.path.exists(output_path):
        # Load existing list from file
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    # OPTIONAL: renumber doc_index to continue after existing ones
    offset = len(existing)
    for j, doc in enumerate(docs, start=1):
        doc["doc_index"] = offset + j

    # Append new docs
    existing.extend(docs)

    # Save back
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(docs)} new docs. Total now: {len(existing)}.")

if __name__ == "__main__":
    main()