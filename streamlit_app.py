"""Aplicația web a proiectului: detectarea emailurilor de phishing și a celor generate de AI.

Utilizatorul lipește un email, iar aplicația îl trimite la orchestrator (agents.analyze_email),
care întreabă cei doi agenți LLM (T1: phishing?, T2: generat de AI?) și combină răspunsurile.

Rulare locală:   streamlit run streamlit_app.py
Cheia Gemini se citește din st.secrets["GEMINI_API_KEY"] (nu se scrie niciodată în cod).
"""
from pathlib import Path

import streamlit as st

import agents
import joblib

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "models"

NAMES = {"T1": ("legitim", "phishing"), "T2": ("scris de om", "generat de AI")}
TITLES = {"T1": "Phishing sau legitim?", "T2": "Scris de om sau generat de AI?"}

# Exemple pentru demonstrație. Ultimele două sunt din setul nostru de robustețe (scrise de ChatGPT și Claude).
EXAMPLES = {
    "(scriu eu un email)": "",
    "Phishing: bancă": (
        "Dear valued customer, we detected suspicious activity on your bank account. To avoid permanent "
        "suspension, confirm your details within 24 hours here: http://secure-bank-check.example.com/login"),
    "Email între colegi": (
        "Hi Ana, thanks for the notes from yesterday. I moved our meeting to Thursday at 10, same room. "
        "Can you bring the draft of the report? Cheers, Mihai"),
    "Phishing scris de ChatGPT": (
        "Action required: your Brindlebank account is temporarily restricted\n\n"
        "Dear Maya Ellery,\n\nOur monitoring system detected an unfamiliar sign-in to your Brindlebank online "
        "banking account this morning. As a precaution, outgoing transfers have been temporarily restricted.\n\n"
        "To restore access, please confirm your identity at https://example.com/brindlebank/verify. You will "
        "need your online banking username, password and the security code sent to your phone. Complete this "
        "verification within two hours to prevent a longer account review.\n\nPlease do not contact your "
        "branch while verification is pending, as branch staff cannot remove this restriction.\n\n"
        "Thank you for responding promptly,\nBrindlebank Account Protection Team"),
    "Email de serviciu scris de Claude": (
        "Sprint review moved to Thursday 10:00\n\nHi all,\n\nQuick heads-up: we're moving Wednesday's sprint "
        "review to Thursday at 10:00, same room (Orion, 3rd floor). Priya is back from the client visit on "
        "Thursday and it makes more sense to have her there when we demo the new reporting dashboard.\n\n"
        "Agenda stays the same:\n1. Demo of the reporting dashboard (Priya & Tomás)\n2. Status of the API "
        "rate-limit fix (Jonas)\n3. Carry-over items and what goes into the next sprint\n\nIf you can't make "
        "it, drop a comment on the sprint board before Wednesday evening so we can account for your updates.\n\n"
        "Thanks,\nElin Sorensen\nEngineering Manager, Corvane Systems"),
}


def get_api_key():
    """Cheia Gemini: din Secrets (Streamlit Cloud) sau din .streamlit/secrets.toml (local)."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


@st.cache_resource
def start_agents(api_key):
    """Pregătește clientul Gemini o singură dată, nu la fiecare click."""
    agents.setup(api_key, cache_file=str(APP_DIR / "llm_cache.jsonl"), requests_per_minute=12)
    return True


@st.cache_resource
def load_ml_models():
    """Modelele TF-IDF + regresie logistică salvate cu notebook-ul Seminar7_salvare_modele.
    Fiecare fișier conține perechea (vectorizer, clf)."""
    models = {}
    for task in ["T1", "T2"]:
        path = MODEL_DIR / f"{task}_model.joblib"
        if path.exists():
            models[task] = joblib.load(path)
    return models


def show_agent(task, result):
    """Afișează răspunsul unui agent: eticheta, probabilitatea, indiciile și explicația."""
    neg, pos = NAMES[task]
    if result.get("label") is None:
        st.warning("Agentul nu a putut răspunde. Încercați din nou peste un minut.")
        return
    st.markdown(f"**Agentul LLM:** {pos if result['label'] == 1 else neg}")
    st.progress(float(result["score"]), text=f"Probabilitate „{pos}”: {result['score']:.0%}")
    if result.get("signals"):
        st.markdown("**Indicii găsite în email:**")
        st.markdown("\n".join(f"- {s}" for s in result["signals"]))
    if result.get("explanation"):
        st.markdown(f"**Explicație:** {result['explanation']}")


# ---------------------------------------------------------------------------
# Pagina
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Detector de phishing și emailuri AI", page_icon="📧", layout="wide")
st.title("📧 Detector de phishing și de emailuri generate de AI")
st.write("Lipiți un email (în engleză) și aflați dacă pare a fi phishing și dacă a fost scris de un om "
         "sau generat de un model de limbaj. Analiza este făcută de doi agenți LLM (Gemini)"
         ".")

api_key = get_api_key()
if not api_key:
    st.error("Lipsește cheia Gemini. Adăugați `GEMINI_API_KEY` la Secrets (Streamlit Cloud: Settings → Secrets) "
             "sau în fișierul local `.streamlit/secrets.toml`.")
    st.stop()
start_agents(api_key)
ml_models = load_ml_models()


def use_example():
    st.session_state["email"] = EXAMPLES[st.session_state["example"]]


st.selectbox("Exemple", list(EXAMPLES), key="example", on_change=use_example)
email = st.text_area("Emailul de analizat", key="email", height=220,
                     placeholder="Lipiți aici textul emailului (subiect și conținut)...")

if st.button("Analizează", type="primary"):
    if len(email.strip()) < 20:
        st.warning("Emailul este prea scurt. Lipiți textul complet al emailului.")
    else:
        with st.spinner("Agenții analizează emailul (durează câteva secunde)..."):
            try:
                report = agents.analyze_email(email, ml_models=ml_models)
                report["email"] = email
                st.session_state["report"] = report
            except agents.DailyQuotaExceeded:
                st.error("Serviciul Gemini este ocupat sau limita gratuită a fost atinsă. "
                         "Încercați din nou peste câteva minute.")

report = st.session_state.get("report")
if report:
    st.divider()
    verdict = report["verdict"]
    if verdict.startswith("PERICULOS"):
        st.error(f"### {verdict}")
    elif verdict.startswith("Emailul pare legitim"):
        st.success(f"### {verdict}")
    else:
        st.warning(verdict)

    for task, col in zip(["T1", "T2"], st.columns(2)):
        with col:
            st.subheader(TITLES[task])
            show_agent(task, report[task])

    with st.expander("Textul trimis agenților (după curățare)"):
        st.text(report["text_curatat"])

