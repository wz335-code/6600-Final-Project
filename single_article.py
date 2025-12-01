import base64
import requests

API_BASE = "https://textinspector.com/api/v2"
USERNAME = ""
PASSWORD = ""

def create_session():
    """Create session and return sessionid."""
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
    return r.json()["sessionid"]


def submit_single_text(sessionid, text, delimiter="#"):
    """Submit ONE text as a document. Returns ctxId."""
    payload = {
        "text": text.replace(delimiter, " "),  # sanitize delimiter
        "split": "1",                          # still required
        "delimiter": delimiter,
        "textmode": "Reading",
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

    resp = data["response"]
    errors = resp.get("errors") or []
    if errors:
        return None, f"API error during newanalysis: {errors}"

    ctxId = resp["ctxId"]
    return ctxId, None


def get_single_cefr(sessionid, ctxId):
    """Retrieve CEFR rating for the only document: doc1."""
    r = requests.get(
        f"{API_BASE}/{ctxId}/doc1/scorecardrating",
        headers={
            "accept": "application/json",
            "Cookie": f"textinspector.session={sessionid}",
        },
    )
    r.raise_for_status()
    data = r.json()

    resp = data.get("response", {})
    overall = resp.get("overallRating")

    if not overall:
        # attempt to extract error message
        error_msg = (
            resp.get("error")
            or data.get("error")
            or data.get("message")
            or "No CEFR rating returned"
        )
        return {"cefr_level": None, "percentage": None, "error": error_msg}

    return {
        "cefr_level": overall["cefrLevel"],
        "percentage": overall["percentage"],

        "error": None,
    }


def evaluate_single_text(text):
    """High-level function: create session → submit text → get CEFR."""
    sessionid = create_session()
    ctxId, error = submit_single_text(sessionid, text)

    if error:
        return {"cefr_level": None, "percentage": None, "error": error}

    return get_single_cefr(sessionid, ctxId)

text = "Celso Amorim joined Brazil’s foreign service almost sixty years ago. He reached the highest job there. But even he cannot remember a year like 2019 in Latin America. “Like this? Never before,” he said. Many countries had big social and political problems that year. The trouble started very early. In Venezuela, large protests began only three weeks after the new year. Many people thought the leader, Nicolás Maduro, would lose power. There were marches, strong fighting, and even an attempt at a military change. But the opposition leader Juan Guaidó did not succeed. Maduro stayed in power.In other countries, problems also began. Puerto Rico, Haiti, Ecuador, and Bolivia all had big protests and violence. In Bolivia, President Evo Morales had to leave his job after the military pushed him out. "
result = evaluate_single_text(text)
print("CEFR evaluation result:", result)