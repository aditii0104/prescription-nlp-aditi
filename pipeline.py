"""
Prescription NLP Pipeline
==========================
Hybrid approach: rule-based extraction (regex + lookup tables)
combined with a token-level ML classifier (TF-IDF + Logistic Regression)
for robust medicine-name detection.

Author  : Prescription NLP Assignment
Approach: See README.md for full design rationale.
"""

import re
import json
import string
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# 1.  ABBREVIATION / NORMALISATION TABLES
# ──────────────────────────────────────────────────────────────────────────────

# Form aliases  →  canonical form
FORM_MAP = {
    # Tablet variants
    "tab":       "tablet",  "tab.":    "tablet",  "tabs":    "tablet",
    "tablet":    "tablet",  "tablets": "tablet",  "t/":      "tablet",
    "tb":        "tablet",  "tb.":     "tablet",
    # Capsule variants
    "cap":       "capsule", "cap.":    "capsule", "caps":    "capsule",
    "caps.":     "capsule", "capsule": "capsule", "capsules":"capsule",
    "cp":        "capsule",
    # Injection variants
    "inj":       "injection","inj.":   "injection","injection":"injection",
    # Suspension / syrup
    "susp":      "suspension","susp.": "suspension","suspension":"suspension",
    "syr":       "syrup",   "syr.":   "syrup",    "syrup":   "syrup",
    # Misc
    "drops":     "drops",
}

# Frequency aliases  →  canonical
FREQ_MAP = {
    # Once daily
    "od":           "OD",  "o.d":        "OD",  "o.d.":       "OD",
    "once daily":   "OD",  "once a day":  "OD",  "once/day":   "OD",
    "once dly":     "OD",  "0d":          "OD",  # typo "0D"
    # Twice daily
    "bd":           "BD",  "b.d":        "BD",  "b.d.":       "BD",
    "twice daily":  "BD",  "twice a day": "BD",  "bid":        "BD",
    "every 12 hours":"BD", "every 12 hrs":"BD",  "every12hours":"BD",
    "12 hourly":    "BD",
    # Three times daily
    "tds":          "TDS", "t.d.s":      "TDS", "t.d.s.":     "TDS",
    "tid":          "TDS", "three times daily":"TDS",
    "three times a day":"TDS",
    "every 8 hours":"TDS", "every 8 hrs": "TDS", "every8hours":"TDS",
    "8 hourly":     "TDS",
    # Four times daily
    "qid":          "QID", "q.i.d":      "QID", "4 times daily":"QID",
    "four times daily":"QID",
    "every 6 hours":"QID", "6 hourly":   "QID",
    # SOS / as needed
    "sos":          "SOS", "s.o.s":      "SOS", "s.o.s.":     "SOS",
    "prn":          "SOS", "p.r.n":      "SOS", "if needed":  "SOS",
    "if req":       "SOS", "if nec":     "SOS", "when reqd":  "SOS",
    "when required":"SOS",
    # At bedtime / night
    "hs":           "HS",  "h.s":        "HS",  "h.s.":       "HS",
    "at bedtime":   "HS",  "bedtime":    "HS",  "at hs":      "HS",
    "h/s":          "HS",
    # Meal-related frequencies encoded as patterns
    "1-1-1":        "TDS", "1-0-1":      "BD",  "1-1-0":      "BD",
    "0-0-1":        "OD",  "0-1-0":      "OD",
    # Misc
    "once weekly":  "once weekly",
    "once monthly": "once monthly",
    "once a week":  "once weekly",
}

# Dose instruction aliases  →  canonical notes
FOOD_MAP = {
    "after food":        "after food",
    "after meals":       "after food",
    "af meals":          "after food",
    "af food":           "after food",
    "af fo od":          "after food",   # OCR noise
    "aft food":          "after food",
    "aft meals":         "after food",
    "aft fd":            "after food",
    "aft fod":           "after food",
    "aftr food":         "after food",
    "aftr meals":        "after food",
    "a/f":               "after food",
    "a.f":               "after food",
    "af":                "after food",
    "pc":                "after food",   # post cibum
    "p/c":               "after food",
    "p.c":               "after food",
    "after fod":         "after food",   # typo
    "before food":       "before food",
    "before meals":      "before food",
    "b4 food":           "before food",
    "b4food":            "before food",
    "b/f":               "before food",
    "b4 food":           "before food",
    "b4food":            "before food",
    "bef fd":            "before food",
    "ac":                "before food",  # ante cibum
    "a/c":               "before food",
    "before fod":        "before food",
    "empty stomach":     "empty stomach",
    "empty stmch":       "empty stomach",
    "emty stomach":      "empty stomach",
    "es":                "empty stomach",
    "with food":         "with food",
    "with milk":         "with milk",
    "with meals":        "with food",
    "at bedtime":        "at bedtime",
    "at bedtme":         "at bedtime",
    "morning only":      "morning only",
    "mrng only":         "morning only",
    "in am":             "in the morning",
    "30 min before meals":"30 min before meals",
    "30min before meals": "30 min before meals",
    "30 min bef meals":   "30 min before meals",
    "for nausea":        "for nausea",
    "for pain":          "for pain",
    "for fever":         "for fever",
    "for allergy":       "for allergy",
    "for diarrhoea":     "for diarrhoea",
    "for wheeze":        "for wheeze",
    "for seizures":      "for seizures",
    "for nerve pain":    "for nerve pain",
    "neuropathic":       "for neuropathic pain",
    "plenty of water":   "with plenty of water",
    "prn":               "as needed",
    "if needed":         "as needed",
    "if nec":            "as needed",
    "if nausea":         "for nausea",
}

# Medicine brand/abbrev  →  INN name
MEDICINE_ABBREV = {
    # Common brand abbreviations
    "atorva":       "Atorvastatin",
    "panto":        "Pantoprazole",
    "ome":          "Omeprazole",
    "pcm":          "Paracetamol",
    "paracet":      "Paracetamol",
    "parcetamol":   "Paracetamol",
    "paracitamol":  "Paracetamol",
    "para":         "Paracetamol",
    "metro":        "Metronidazole",
    "metronidaz":   "Metronidazole",
    "azithro":      "Azithromycin",
    "azitromycin":  "Azithromycin",
    "monte":        "Montelukast",
    "montelucast":  "Montelukast",
    "doxy":         "Doxycycline",
    "doxycyclene":  "Doxycycline",
    "doxycyclin":   "Doxycycline",
    "ibu":          "Ibuprofen",
    "ibuprofin":    "Ibuprofen",
    "feso4":        "Ferrous Sulfate",
    "ferrous":      "Ferrous Sulfate",
    "calcarb":      "Calcium Carbonate",
    "cal carb":     "Calcium Carbonate",
    "calcium carb": "Calcium Carbonate",
    "pregabaln":    "Pregabalin",
    "pregabaline":  "Pregabalin",
    "levocetirizne":"Levocetirizine",
    "cetirizne":    "Cetirizine",
    "cetirizine":   "Cetirizine",
    "fexofenadin":  "Fexofenadine",
    "lozartan":     "Losartan",
    "losartan k":   "Losartan",
    "clonazapam":   "Clonazepam",
    "clonazepam":   "Clonazepam",
    "dom":          "Domperidone",
    "domperidne":   "Domperidone",
    "gaba":         "Gabapentin",
    "gabapentiin":  "Gabapentin",
    "glicla":       "Gliclazide",
    "gliclazde":    "Gliclazide",
    "metfromin":    "Metformin",
    "metfornin":    "Metformin",
    "ondan":        "Ondansetron",
    "ondansetrone": "Ondansetron",
    "prednisalone": "Prednisolone",
    "pred":         "Prednisolone",
    "trama":        "Tramadol",
    "ramipril":     "Ramipril",
    "multivit":     "Multivitamin",
    "multivitamin": "Multivitamin",
    "folic acid":   "Folic Acid",
    "amox":         "Amoxicillin",
    "clarithomycin":"Clarithromycin",
    "mefenamic":    "Mefenamic Acid",
    "naproxene":    "Naproxen",
    "napoxen":      "Naproxen",
    "naproxen":     "Naproxen",
    "aspiirin":     "Aspirin",
    "vitamin d3":   "Vitamin D3",
    "vitamin d 3":  "Vitamin D3",
    "panto prazole":"Pantoprazole",
    "azithro mycin":"Azithromycin",
    "cp mefenamic": "Mefenamic Acid",
    "inj panto":    "Pantoprazole",
    "levocetirizne":"Levocetirizine",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Light normalisation:
    - collapse extra whitespace
    - strip trailing punctuation
    - fix common fused-token patterns (e.g. "Ome40mg" → "Ome 40 mg")
    - lowercase for matching, but preserve original for output
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    # Fix "0D" typo for "OD" BEFORE digit-letter splitting (zero mistaken for letter O)
    text = re.sub(r'(?<!\d)0([Dd])(?!\d)', r'O\1', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Insert space between letter and digit boundaries (e.g. "Ome40mg", "BD4")
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    # Normalise slashes used as "per" in frequency (BD/7d → BD for 7d)
    text = re.sub(r'(\b(?:OD|BD|TDS|QID|SOS|HS)\b)\s*/\s*(\d)', r'\1 for \2', text, flags=re.IGNORECASE)
    # Normalise "x2wk", "x 3 days" patterns
    text = re.sub(r'\bx\s*(\d)', r'for \1', text, flags=re.IGNORECASE)
    return text


def lowercase_strip(s: str) -> str:
    return s.lower().strip().strip('.')


# ──────────────────────────────────────────────────────────────────────────────
# 3.  RULE-BASED EXTRACTORS
# ──────────────────────────────────────────────────────────────────────────────

# Strength: e.g. "500 mg", "60000 IU", "100 mg/5 ml", "10 mg/ml", "2.5 mg"
STRENGTH_RE = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|iu|g|ml|units?)'
    r'(?:\s*/\s*(\d+(?:\.\d+)?)\s*(mg|mcg|iu|g|ml))?'
    r'\b',
    re.IGNORECASE
)

# Dosage: explicit "1 tablet", "2 tablets", "10 ml", "1 ampoule", "10 units"
DOSAGE_RE = re.compile(
    r'\b(\d+(?:\.\d+)?)\s+'
    r'(tablets?|capsules?|ampoules?|vials?|drops?|units?|ml|puffs?)\b',
    re.IGNORECASE
)

# Duration: "5 days", "1 week", "2 weeks", "1wk", "2wk", "3d", "10d", "1 month"
DURATION_RE = re.compile(
    r'\bfor\s+(\d+)\s*(days?|weeks?|months?|d|wk|wks|mo)\b'
    r'|'
    r'\b(\d+)\s*(days?|weeks?|months?|d|wk|wks|mo)\b',
    re.IGNORECASE
)

# Frequency - dash pattern "1-1-1", "1-0-1", etc.
DASH_FREQ_RE = re.compile(r'\b([01]-[01]-[01])\b')

# Frequency - "every N hours"
EVERY_N_RE   = re.compile(r'\bevery\s+(\d+)\s*h(?:ou?rs?)?\b', re.IGNORECASE)

# Duration helper for "once weekly / once monthly"
ONCE_PERIOD_RE = re.compile(r'\bonce\s+(weekly|monthly|daily|a\s+week|a\s+month)\b', re.IGNORECASE)


def extract_strength(text: str) -> str:
    """Return first strength match (handles compound like 250 mg/5 ml)."""
    m = STRENGTH_RE.search(text)
    if not m:
        return ""
    val1, unit1, val2, unit2 = m.group(1), m.group(2), m.group(3), m.group(4)
    strength = f"{val1} {unit1.lower()}"
    if val2 and unit2:
        strength += f"/{val2} {unit2.lower()}"
    return strength


def extract_dosage(text: str) -> str:
    """Return explicit dosage (e.g. '1 tablet', '10 ml')."""
    m = DOSAGE_RE.search(text)
    if m:
        qty  = m.group(1)
        form = m.group(2).lower().rstrip('s') + ('s' if float(qty) > 1 else '')
        # canonical singular
        form_singular = re.sub(r's$', '', m.group(2).lower())
        if float(qty) == 1:
            return f"1 {form_singular}"
        else:
            return f"{qty} {form_singular}s"
    return ""


def normalize_duration(val: str, unit: str) -> str:
    """Canonicalise duration unit."""
    unit_map = {
        "d": "days", "day": "days", "days": "days",
        "wk": "weeks", "wks": "weeks", "week": "weeks", "weeks": "weeks",
        "mo": "months", "month": "months", "months": "months",
    }
    canon_unit = unit_map.get(unit.lower(), unit.lower())
    return f"{val} {canon_unit}"


def extract_duration(text: str) -> str:
    """Return duration string."""
    # Check "once weekly / once monthly" first
    om = ONCE_PERIOD_RE.search(text)
    if om:
        period = om.group(1).lower().strip()
        if period in ("weekly", "a week"):
            return "1 week"
        if period in ("monthly", "a month"):
            return "1 month"

    m = DURATION_RE.search(text)
    if not m:
        return ""
    if m.group(1):          # "for N unit"
        return normalize_duration(m.group(1), m.group(2))
    else:                   # "N unit" standalone
        return normalize_duration(m.group(3), m.group(4))


def extract_frequency(text: str) -> str:
    """
    Multi-strategy frequency extraction:
    1. Normalised text → lookup in FREQ_MAP (longest match first)
    2. Dash pattern (1-1-1)
    3. every N hours

    Priority rule: if an explicit dosing frequency (OD/BD/TDS/QID/SOS) is
    present alongside a timing keyword (HS / at bedtime), the explicit
    frequency wins and bedtime is relegated to notes.
    """
    lower = text.lower()

    # Strategy 1: dash pattern
    dm = DASH_FREQ_RE.search(lower)
    if dm:
        key = dm.group(1)
        if key in FREQ_MAP:
            return FREQ_MAP[key]

    # Strategy 2: every N hours
    em = EVERY_N_RE.search(lower)
    if em:
        hours = int(em.group(1))
        hour_map = {6: "QID", 8: "TDS", 12: "BD", 24: "OD"}
        return hour_map.get(hours, f"every {hours} hours")

    # Strategy 3: explicit-first scan
    # First pass: look for non-HS primary frequencies
    PRIMARY_FREQS = {"od","bd","tds","qid","sos","once daily","twice daily",
                     "three times daily","four times daily","once a day",
                     "twice a day","three times a day","1-1-1","1-0-1",
                     "1-1-0","0-0-1","0-1-0","once weekly","once monthly",
                     "every 12 hours","every 8 hours","every 6 hours"}
    sorted_keys = sorted(FREQ_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in PRIMARY_FREQS:
            if re.search(r'\b' + re.escape(key) + r'\b', lower):
                return FREQ_MAP[key]

    # Second pass: allow HS / bedtime as frequency only if no primary found
    for key in sorted_keys:
        if re.search(r'\b' + re.escape(key) + r'\b', lower):
            return FREQ_MAP[key]

    return ""


def extract_notes(text: str, frequency: str, duration: str) -> str:
    """
    Extract administration notes (food instructions, indication, timing).
    Uses longest-match lookup in FOOD_MAP.
    """
    lower = text.lower()
    found = []

    # Longest-match scan
    sorted_keys = sorted(FOOD_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if re.search(r'\b' + re.escape(key) + r'\b', lower):
            canon = FOOD_MAP[key]
            if canon not in found:
                found.append(canon)

    # De-duplicate overlapping notes (e.g. "at bedtime" and "morning only" don't co-occur)
    # Priority: food-relation > timing > indication
    food_notes    = [n for n in found if n in ("after food","before food","empty stomach","with food","with milk")]
    timing_notes  = [n for n in found if n in ("at bedtime","morning only","in the morning","30 min before meals")]
    indicate_notes = [n for n in found if n.startswith("for ") or n in ("as needed","with plenty of water","for nausea","for pain","for fever","for allergy","for diarrhoea","for wheeze","for seizures","for nerve pain","for neuropathic pain")]

    parts = food_notes[:1] + timing_notes[:1] + indicate_notes[:1]
    return "; ".join(parts) if parts else ""


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FORM EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_form(text: str) -> str:
    """
    Extract dosage form.
    Priority:
    1. Leading form keyword (Tab., Cap, Inj, Susp, Syr, ...)
    2. Inline form word (tablet, capsule, ampoule, vial, syrup, suspension)
    """
    lower_stripped = lowercase_strip(text.split()[0]) if text.split() else ""

    # Leading token match
    if lower_stripped in FORM_MAP:
        return FORM_MAP[lower_stripped]

    # Full FORM_MAP scan on full text (longest first)
    lower = text.lower()
    sorted_keys = sorted(FORM_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if re.search(r'\b' + re.escape(key) + r'\b', lower):
            return FORM_MAP[key]

    # Infer from Inj -> injection, Susp -> suspension
    if re.match(r'\binj\b', lower):
        return "injection"
    if re.match(r'\bsusp\b', lower):
        return "suspension"

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MEDICINE NAME EXTRACTION  (ML-assisted)
# ──────────────────────────────────────────────────────────────────────────────
# Design:
#   We treat medicine-name extraction as a token-classification problem.
#   Labels are generated heuristically from a labelled seed set of known names,
#   then a TF-IDF char-ngram + Logistic Regression classifier is trained
#   on those seeds. For unseen tokens the classifier provides a confidence score.
#   Additionally, an abbreviation lookup table covers common brand names.
#
#   Tokens that match known field patterns (strength, form, frequency, duration,
#   food keywords) are blacklisted from medicine name candidates.

BLACKLIST_TOKENS = {
    # Form words
    "tab","tab.","tablet","tablets","cap","cap.","caps","capsule","capsules",
    "inj","inj.","injection","susp","susp.","suspension","syr","syr.","syrup",
    "drops","ampoule","vial","t/","tb","tb.","cp","t",
    # Frequency
    "od","bd","tds","qid","sos","hs","prn","daily","twice","thrice",
    "once","every","hours","hourly","weekly","monthly","bedtime",
    # Duration
    "for","days","day","week","weeks","wk","wks","month","months","d","x",
    # Food instructions
    "after","before","with","food","meals","meal","milk","stomach","empty",
    "morning","night","am","pm","pc","ac","af","a/f","b/f","es","b4",
    # Dosage helper
    "tablet","tablets","capsule","capsules","ampoule","vial","units","unit","ml",
    "1","2","3","4","5","6","7","8","9","10",
    # Misc
    "at","in","only","if","needed","required","reqd","when","and","or","the",
    "min","30","prn","nec","aft","aftr","mrng","of","a",
}

# Known medicine name fragments used as training seeds
KNOWN_MEDICINE_SEEDS = [
    "Amitriptyline","Ranitidine","Loperamide","Vitamin","Losartan","Ciprofloxacin",
    "Ramipril","Montelukast","Amlodipine","Atorvastatin","Atorva","Salbutamol",
    "Zinc","Sulfate","Tramadol","Doxycycline","Metformin","Cetirizine","Glipizide",
    "Clonazepam","Prednisolone","Ferrous","Omeprazole","Sertraline","Gliclazide",
    "Amoxicillin","Naproxen","Cephalexin","Ondansetron","Aspirin","Diclofenac",
    "Pantoprazole","Domperidone","Fexofenadine","Metronidazole","Mefenamic","Acid",
    "Furosemide","Levetiracetam","Atenolol","Levocetirizine","Pregabalin","Gabapentin",
    "Metoprolol","Insulin","Glargine","Sitagliptin","Calcium","Carbonate","Folic",
    "Paracetamol","Ibuprofen","Azithromycin","Clarithromycin","Clindamycin",
    "Doxycyclin","Multivitamin",
    # Abbreviations / brand
    "Panto","Ome","PCM","Paracet","Metro","Azithro","Monte","Doxy","CalCarb",
    "Atorva","Gaba","Glicla","Dom","Ondan","Pred","Trama","Amox","Ibu","FeSO4",
]


def _build_medicine_classifier():
    """
    Build a char-ngram TF-IDF + Logistic Regression classifier.
    Positive class: known medicine name tokens.
    Negative class: blacklist / structural tokens.
    Returns fitted (vectorizer, classifier).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Positive samples: seed medicine names (lowercase)
    pos = [s.lower() for s in KNOWN_MEDICINE_SEEDS]

    # Negative samples: blacklist + numeric patterns + form/freq words
    neg = list(BLACKLIST_TOKENS) + [
        "mg","mcg","iu","ml","g","100","200","250","500","1000",
        "5","10","20","25","40","50","75","80","150","300","325","400",
        "0d","1-1-1","1-0-1","bd","tds","hs","for","with","after","before",
    ]

    X = pos + neg
    y = [1]*len(pos) + [0]*len(neg)

    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=1)
    X_vec = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_vec, y)
    return vec, clf


# Build once at module load
_VEC, _CLF = _build_medicine_classifier()


def _classify_token(token: str) -> float:
    """Return probability [0,1] that token is a medicine-name token."""
    x = _VEC.transform([token.lower()])
    return _CLF.predict_proba(x)[0][1]


def extract_medicine_name(raw_text: str, form: str, strength: str,
                           dosage: str, frequency: str, duration: str) -> str:
    """
    Hybrid medicine-name extractor:
    1. Try abbreviation lookup table.
    2. Remove known non-medicine tokens from the text.
    3. Score remaining tokens with the ML classifier.
    4. Collect highest-scoring contiguous name span.
    """
    text = raw_text.strip()

    # ── Step 1: full-text abbreviation match (longest match first) ──
    lower_text = text.lower()
    for key in sorted(MEDICINE_ABBREV.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(key) + r'\b', lower_text):
            return MEDICINE_ABBREV[key]

    # ── Step 2: build candidate token list ──
    # Remove the form token at the start
    tokens = text.split()
    if tokens and lowercase_strip(tokens[0]) in FORM_MAP:
        tokens = tokens[1:]

    # Build a mask of tokens to skip
    skip_patterns = [
        STRENGTH_RE,
        DOSAGE_RE,
        DURATION_RE,
        re.compile(r'^' + '|^'.join(re.escape(f) for f in list(FREQ_MAP.keys()) + list(FORM_MAP.keys())), re.IGNORECASE),
    ]

    # Values to skip: strength/dosage/duration/freq/form as substrings
    known_skip = set()
    for pat in [strength, dosage, duration, frequency, form]:
        if pat:
            for tok in pat.split():
                known_skip.add(tok.lower())

    candidate_tokens = []
    for tok in tokens:
        clean = tok.strip(string.punctuation)
        lower = clean.lower()
        if not clean:
            continue
        if lower in BLACKLIST_TOKENS:
            continue
        if lower in known_skip:
            continue
        # Skip pure numbers
        if re.match(r'^\d+(\.\d+)?$', clean):
            continue
        # Skip strength / dosage patterns
        if STRENGTH_RE.fullmatch(clean) or DOSAGE_RE.fullmatch(clean):
            continue
        # Skip frequency patterns
        if lower in FREQ_MAP:
            continue
        candidate_tokens.append(clean)

    if not candidate_tokens:
        return ""

    # ── Step 3: Score and select ──
    scored = [(tok, _classify_token(tok)) for tok in candidate_tokens]

    # Collect contiguous high-confidence tokens (threshold=0.4)
    THRESHOLD = 0.40
    name_parts = []
    for tok, score in scored:
        if score >= THRESHOLD:
            name_parts.append(tok)
        elif name_parts:
            break   # stop at first gap after a run

    if not name_parts:
        # Fall back to highest scoring single token
        best = max(scored, key=lambda x: x[1])
        if best[1] > 0.2:
            name_parts = [best[0]]

    if not name_parts:
        return ""

    # ── Step 4: Clean up the name ──
    # Title-case and strip trailing punctuation
    name = " ".join(tok.strip(string.punctuation) for tok in name_parts)
    name = name.title()

    # Remove any trailing strength-like suffix that slipped through
    name = re.sub(r'\s+\d+.*$', '', name).strip()

    return name


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN EXTRACTION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_prescription(raw_text: str) -> dict:
    """
    Full pipeline for one prescription string.
    Returns a structured dict.
    """
    # Normalise
    norm = normalize_text(raw_text)

    form      = extract_form(norm)
    strength  = extract_strength(norm)
    dosage    = extract_dosage(norm)
    frequency = extract_frequency(norm)
    duration  = extract_duration(norm)
    notes     = extract_notes(norm, frequency, duration)
    medicine  = extract_medicine_name(norm, form, strength, dosage, frequency, duration)

    return {
        "raw_text":     raw_text,
        "medicine_name": medicine,
        "form":          form,
        "strength":      strength,
        "dosage":        dosage,
        "frequency":     frequency,
        "duration":      duration,
        "notes":         notes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  BATCH PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str):
    """Read input JSON, run pipeline on every record, write output JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = []
    for i, record in enumerate(records):
        raw = record.get("raw_text", "")
        try:
            result = extract_prescription(raw)
        except Exception as e:
            result = {
                "raw_text":      raw,
                "medicine_name": "",
                "form":          "",
                "strength":      "",
                "dosage":        "",
                "frequency":     "",
                "duration":      "",
                "notes":         f"ERROR: {e}",
            }
        results.append(result)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(records)} records...", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} records written to {output_path}")
    return results


if __name__ == "__main__":
    import sys
    inp  = sys.argv[1] if len(sys.argv) > 1 else "prescription_raw_text_only.json"
    outp = sys.argv[2] if len(sys.argv) > 2 else "output_structured.json"
    print(f"Running pipeline: {inp} → {outp}")
    process_file(inp, outp)
