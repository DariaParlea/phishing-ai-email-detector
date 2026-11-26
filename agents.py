"""Agenții LLM (Gemini) pentru cele două sarcini și orchestratorul care îi coordonează.

Fiecare agent primește un email și întoarce un dicționar:
    {"label": 0 sau 1, "score": probabilitatea clasei 1, "explanation": "...", "signals": [...]}

T1: 1 = phishing, 0 = legitim.     T2: 1 = generat de AI, 0 = scris de om.
"""
import hashlib
import json
import os
import re
import time

# ---------------------------------------------------------------------------
# Configurare
# ---------------------------------------------------------------------------
# Modelul folosit de agenți. Verificați în Google AI Studio ce modele are contul vostru
# în planul gratuit; dacă primiți eroarea 404, schimbați numele de aici.
MODEL = "gemini-3.5-flash-lite"
REQUESTS_PER_MINUTE = 10          # sub limita planului gratuit, ca să evităm refuzurile
MAX_EMAIL_CHARS = 3000            # emailurile foarte lungi sunt tăiate, ca să economisim token-uri
CACHE_FILE = "llm_cache.jsonl"    # aici se salvează fiecare răspuns, ca să nu repetăm cererile
MAX_RETRIES = 5                   # de câte ori reîncercăm o cerere refuzată temporar
MAX_FAILED_IN_A_ROW = 3           # după atâtea emailuri eșuate la rând, ne oprim și reluăm mai târziu

_client = None
_last_call = 0.0
_cache = None
_failed_in_a_row = 0
STATS = {"cereri_api": 0, "din_cache": 0, "erori": 0}


class DailyQuotaExceeded(RuntimeError):
    """Limita zilnică a fost atinsă sau serviciul nu răspunde; rularea se poate relua mai târziu."""


def setup(api_key, cache_file=None, model=None, requests_per_minute=None):
    """Pregătește clientul Gemini. Se apelează o singură dată, la început."""
    global _client, CACHE_FILE, MODEL, REQUESTS_PER_MINUTE, _cache, _failed_in_a_row
    from google import genai
    _client = genai.Client(api_key=api_key)
    if cache_file:
        CACHE_FILE = cache_file
    if model:
        MODEL = model
    if requests_per_minute:
        REQUESTS_PER_MINUTE = requests_per_minute
    _cache = None
    _failed_in_a_row = 0


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        _cache[item["key"]] = item
                    except (json.JSONDecodeError, KeyError):
                        pass
    return _cache


def _save_to_cache(item):
    _load_cache()[item["key"]] = item
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _retry_delay(message, default):
    """Dacă serverul spune cât să așteptăm (de ex. "retryDelay": "34s"), folosim acea valoare."""
    m = re.search(r"retry(?:Delay|\s+in)[\"':\s]*([\d.]+)\s*s", message, re.IGNORECASE)
    return min(120.0, float(m.group(1)) + 2) if m else default


def call_llm(prompt, schema, model=None, temperature=0.0, max_retries=None):
    """Trimite un prompt la Gemini și întoarce răspunsul JSON (dict), plus timpul și token-urile.
    Respectă limita de cereri pe minut și reîncearcă la erori temporare."""
    global _last_call, _failed_in_a_row
    model = model or MODEL
    max_retries = max_retries or MAX_RETRIES
    key = hashlib.sha256(json.dumps([model, prompt, schema, temperature]).encode()).hexdigest()
    cached = _load_cache().get(key)
    if cached is not None:
        STATS["din_cache"] += 1
        return cached["result"], cached["meta"]
    if _client is None:
        raise RuntimeError("Apelați întâi agents.setup(api_key).")

    for attempt in range(max_retries):
        wait = 60.0 / REQUESTS_PER_MINUTE + 0.3 - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        t0 = time.perf_counter()
        try:
            response = _client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json",
                        "response_json_schema": schema,
                        "temperature": temperature},
            )
            latency = time.perf_counter() - t0
            STATS["cereri_api"] += 1
            result = _parse_json(response.text)
            usage = getattr(response, "usage_metadata", None)
            meta = {"secunde": round(latency, 3),
                    "tokeni_intrare": getattr(usage, "prompt_token_count", None),
                    "tokeni_iesire": getattr(usage, "candidates_token_count", None),
                    "model": model}
            _save_to_cache({"key": key, "result": result, "meta": meta})
            _failed_in_a_row = 0
            return result, meta
        except Exception as e:  # erori de rețea, limită depășită, server ocupat
            STATS["erori"] += 1
            message = str(e)
            code = getattr(e, "code", None)
            short = message.replace("\n", " ")[:160]
            if code == 429 or "429" in message or "RESOURCE_EXHAUSTED" in message:
                if "PerDay" in message or "per day" in message.lower():
                    raise DailyQuotaExceeded(
                        "Ați atins limita zilnică a planului gratuit. Rezultatele de până acum sunt salvate; "
                        "rulați din nou celula mâine și se continuă de unde a rămas.") from e
                delay = _retry_delay(message, min(60, 15 * (attempt + 1)))
                print(f"  [limita pe minut] aștept {delay:.0f} s ... ({short})")
            elif code in (500, 502, 503, 504) or any(
                    k in message for k in ["UNAVAILABLE", "overloaded", "INTERNAL", "DEADLINE"]) \
                    or "timed out" in message.lower() or "connection" in message.lower():
                delay = min(60, 15 * (attempt + 1))
                print(f"  [serverul Gemini e ocupat] aștept {delay:.0f} s ... ({short})")
            else:
                raise
            time.sleep(delay)

    # Cererea nu a reușit nici după toate reîncercările
    _failed_in_a_row += 1
    if _failed_in_a_row >= MAX_FAILED_IN_A_ROW:
        _failed_in_a_row = 0
        raise DailyQuotaExceeded(
            "Serverul Gemini nu răspunde de mai multe ori la rând (este ocupat sau limita a fost atinsă). "
            "Rezultatele de până acum sunt salvate; rulați din nou celula peste 30-60 de minute "
            "și se continuă de unde a rămas.")
    print("  O cerere nu a reușit; emailul este marcat ca invalid și va fi reîncercat la următoarea rulare.")
    return None, {"secunde": None, "tokeni_intrare": None, "tokeni_iesire": None, "model": model}


def _parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}|\[.*\]", text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Promptul și formatul de răspuns al agenților
# ---------------------------------------------------------------------------
TASKS = {
    "T1": {
        "role": "You are a cybersecurity analyst who specialises in email security.",
        "question": "Decide whether the email below is a PHISHING email (an attempt to steal credentials, money "
                    "or personal data, or to make the reader click a malicious link or open a malicious "
                    "attachment; unsolicited scam or spam advertising also counts as phishing here) or a "
                    "LEGITIMATE email.",
        "labels": ("legitimate", "phishing"),
    },
    "T2": {
        "role": "You are a computational linguist who specialises in detecting machine-generated text.",
        "question": "Decide whether the email below was WRITTEN BY A HUMAN or GENERATED BY AN AI language model. "
                    "Judge only the writing style, not whether the email is malicious: both humans and AI write "
                    "phishing and legitimate emails.",
        "labels": ("human", "ai"),
    },
}

NOTE_ON_TEXT = ("Note: the email text was pre-processed: links were replaced with the word URL and email "
                "addresses with the word EMAIL.")


def response_schema(task):
    neg, pos = TASKS[task]["labels"]
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": [neg, pos]},
            "probability": {"type": "number",
                            "description": f"Probability between 0 and 1 that the email is '{pos}'."},
            "signals": {"type": "array", "items": {"type": "string"},
                        "description": "Up to 4 short concrete clues found in the email."},
            "explanation": {"type": "string",
                            "description": "Two or three sentences a non-expert can understand."},
        },
        "required": ["label", "probability", "signals", "explanation"],
    }


def build_prompt(task, email, examples=None, ml_hint=None):
    """Construiește promptul agentului.
    examples: listă de (text, eticheta 0/1) pentru varianta few-shot.
    ml_hint: (probabilitate, cuvinte importante) de la modelul ML, pentru varianta hibridă."""
    t = TASKS[task]
    neg, pos = t["labels"]
    parts = [t["role"], t["question"], NOTE_ON_TEXT]
    if examples:
        parts.append("Here are some labelled examples:")
        for i, (text, lab) in enumerate(examples, 1):
            parts.append(f"Example {i} (label: {pos if lab == 1 else neg}):\n\"\"\"\n{text[:700]}\n\"\"\"")
    if ml_hint is not None:
        prob, words = ml_hint
        word_list = ", ".join(f"'{w}' ({'+' if c > 0 else '-'})" for w, c in words) or "none"
        parts.append(
            "A machine-learning classifier (TF-IDF + logistic regression, trained on thousands of labelled emails) "
            f"estimated a probability of {prob:.2f} that this email is '{pos}'. The words that influenced it most "
            f"were: {word_list} (+ pushes towards '{pos}', - towards '{neg}'). This classifier can be wrong, "
            "for example when it relies on words that are not really meaningful. Use it as one piece of "
            "evidence and make your own final decision.")
    parts.append(f"Email to analyse:\n\"\"\"\n{email[:MAX_EMAIL_CHARS]}\n\"\"\"")
    parts.append("Think about the concrete clues in the email, then answer in the required JSON format.")
    return "\n\n".join(parts)


def run_agent(task, email, examples=None, ml_hint=None, model=None):
    """Rulează agentul pentru o sarcină și întoarce rezultatul într-un format comun."""
    neg, pos = TASKS[task]["labels"]
    result, meta = call_llm(build_prompt(task, email, examples, ml_hint), response_schema(task), model=model)
    if not isinstance(result, dict) or result.get("label") not in (neg, pos):
        return {"label": None, "score": None, "explanation": None, "signals": [], **meta}
    try:
        prob = float(result.get("probability"))
        prob = min(1.0, max(0.0, prob))
    except (TypeError, ValueError):
        prob = 1.0 if result["label"] == pos else 0.0
    return {"label": int(result["label"] == pos), "score": prob,
            "explanation": result.get("explanation", ""), "signals": result.get("signals", []), **meta}


# ---------------------------------------------------------------------------
# Orchestratorul
# ---------------------------------------------------------------------------
def analyze_email(email, ml_models=None, model=None):
    """Orchestratorul: curăță emailul, îl trimite la cei doi agenți (și, dacă există, la modelele ML,
    ca unealtă) și combină răspunsurile într-un singur raport pentru utilizator.

    ml_models: {"T1": (vectorizer, clf), "T2": (vectorizer, clf)} sau None.
    """
    from preprocessing import clean_text, ml_predict
    text = clean_text(email)
    report = {"text_curatat": text}
    for task in ["T1", "T2"]:
        hint = None
        if ml_models and task in ml_models:
            probs, words = ml_predict(*ml_models[task], [text])
            hint = (float(probs[0]), words[0])
            report[f"{task}_ml_scor"] = hint[0]
        report[task] = run_agent(task, text, ml_hint=hint, model=model)

    phishing, ai = report["T1"]["label"], report["T2"]["label"]
    if phishing is None or ai is None:
        report["verdict"] = "Analiza nu a putut fi finalizată. Încercați din nou."
    elif phishing == 1:
        report["verdict"] = ("PERICULOS: emailul pare a fi phishing" +
                             (" și a fost probabil generat de AI." if ai == 1 else "."))
    else:
        report["verdict"] = ("Emailul pare legitim" +
                             (", dar a fost probabil generat de AI (ceea ce nu înseamnă că este periculos)."
                              if ai == 1 else "."))
    return report
