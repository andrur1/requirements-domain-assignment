import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DB_PATH = "requirements.duckdb"
MODEL_NAME = "all-MiniLM-L6-v2"


# ----------------------------
# TEXT NORMALIZATION
# ----------------------------
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


# ----------------------------
# HEURISTIC BOOST (ALL DOMAINS)
# ----------------------------
def heuristic_boost(text, domain):
    text = normalize_text(text)

    signals = {
        "Software": [
            "software", "algorithm", "logic", "state machine",
            "diagnostic", "logging", "log", "update", "ota",
            "interface", "data", "validation"
        ],
        "Hardware": [
            "hardware", "sensor", "actuator", "ecu", "mcu", "pmic",
            "voltage", "power", "rail", "connector", "electrical",
            "electronic", "component", "plating", "emc", "shielding"
        ],
        "Functional Safety": [
            "asil", "safety", "safe state", "fault", "failure",
            "hazard", "mitigation", "plausibility", "fallback", "crash", "brake"
        ],
        "Cybersecurity": [
            "secure", "security", "authentication", "authenticated",
            "encrypt", "encryption", "cryptographic", "key",
            "secure boot", "access control"
        ],
        "System Engineering": [
            "system", "integration", "cross-domain", "end-to-end",
            "interface", "compatibility", "legacy", "system-level"
        ],
        "Testing & Validation": [
            "test", "testing", "validation", "verification",
            "criteria", "test case", "unit test"
        ]
    }

    keywords = signals.get(domain, [])
    matches = sum(1 for k in keywords if k in text)

    if matches >= 2:
        return 0.12
    elif matches == 1:
        return 0.08
    return 0.0


# ----------------------------
# STRONG SIGNAL DETECTION
# ----------------------------
def has_strong_signal(text, domain):
    text = normalize_text(text)

    strong_signals = {
        "Software": ["algorithm", "software", "logging", "update", "interface"],
        "Hardware": ["connector", "electrical", "component", "emc", "sensor", "voltage"],
        "Functional Safety": ["asil", "safe state", "fault", "hazard", "failure"],
        "Cybersecurity": ["secure boot", "authentication", "encryption", "cryptographic"],
        "System Engineering": ["system integration", "end-to-end", "cross-domain"],
        "Testing & Validation": ["test", "validation", "verification"]
    }

    return any(k in text for k in strong_signals.get(domain, []))


# ----------------------------
# CONFIDENCE CALCULATION
# ----------------------------
def get_confidence(score_top1, score_top2, text, domain):
    score_gap = score_top1 - score_top2
    strong_signal = has_strong_signal(text, domain)

    # very confident: strong semantic + clear separation
    if score_top1 >= 0.65 and score_gap >= 0.12:
        return "High"

    # strong keyword signal
    if strong_signal and score_gap >= 0.05:
        return "High"

    # reasonable cases
    if score_top1 >= 0.50:
        return "Medium"

    return "Low"


# ----------------------------
# RATIONALE GENERATION
# ----------------------------
def generate_rationale(text, domain1, domain2, score_gap):
    text = normalize_text(text)

    domain_reasoning = {
        "Software": {
            "keywords": ["algorithm", "logic", "software", "logging", "log", "update", "ota", "interface", "data", "validation"],
            "explanation": "these terms usually refer to implementation logic, data handling, software updates, logging, or software interfaces"
        },
        "Hardware": {
            "keywords": ["hardware", "sensor", "actuator", "ecu", "mcu", "pmic", "voltage", "power", "rail", "connector", "electrical", "electronic", "component", "plating", "emc", "shielding"],
            "explanation": "these terms refer to physical components, electrical behavior, power infrastructure, connectors, EMC protection, or hardware-level design"
        },
        "Functional Safety": {
            "keywords": ["asil", "safety", "safe state", "fault", "failure", "hazard", "mitigation", "plausibility", "fallback", "crash", "brake"],
            "explanation": "these terms are related to fault handling, hazard mitigation, safe-state behavior, ASIL classification, or safety-critical vehicle behavior"
        },
        "Cybersecurity": {
            "keywords": ["secure", "security", "authentication", "authenticated", "encrypt", "encryption", "cryptographic", "key", "secure boot", "access control"],
            "explanation": "these terms indicate protection against unauthorized access, secure execution, authentication, encryption, or cybersecurity controls"
        },
        "System Engineering": {
            "keywords": ["system", "integration", "cross-domain", "end-to-end", "interface", "compatibility", "legacy", "system-level"],
            "explanation": "these terms suggest system-level coordination, cross-domain behavior, integration constraints, compatibility, or end-to-end functionality"
        },
        "Testing & Validation": {
            "keywords": ["test", "testing", "validation", "verification", "criteria", "test case", "unit test"],
            "explanation": "these terms refer to verification activities, validation criteria, test planning, or test execution"
        }
    }

    matched_keywords = []
    if domain1 in domain_reasoning:
        matched_keywords = [
            keyword for keyword in domain_reasoning[domain1]["keywords"]
            if keyword in text
        ]

    if matched_keywords:
        keyword_text = ", ".join(matched_keywords[:5])
        rationale = (
            f"This requirement is classified under {domain1} because it contains domain-specific indicators "
            f"such as {keyword_text}. In this context, {domain_reasoning[domain1]['explanation']}."
        )
    else:
        rationale = (
            f"The requirement was assigned to {domain1} because its overall wording is semantically closest "
            f"to the {domain1} domain description, even though no explicit domain-specific keyword was dominant."
        )

    if domain2:
        if score_gap < 0.07:
            rationale += (
                f" The secondary candidate is {domain2}, and the similarity gap is small "
                f"({score_gap:.2f}), so the case is marked as ambiguous."
            )
        else:
            rationale += (
                f" The secondary candidate is {domain2}, but the similarity gap "
                f"({score_gap:.2f}) is large enough to keep {domain1} as the primary recommendation."
            )

    return rationale


# ----------------------------
# MAIN
# ----------------------------
def main():
    con = duckdb.connect(DB_PATH)

    requirements_df = con.execute("""
        SELECT requirement_id, full_text
        FROM requirements_unassigned
        WHERE full_text IS NOT NULL
          AND trim(full_text) <> ''
    """).fetchdf()

    domains_df = con.execute("""
        SELECT domain_name, domain_description
        FROM domains_ref
    """).fetchdf()

    model = SentenceTransformer(MODEL_NAME)

    req_texts = requirements_df["full_text"].apply(normalize_text).tolist()
    dom_texts = domains_df["domain_description"].apply(normalize_text).tolist()

    req_embeddings = model.encode(req_texts, normalize_embeddings=True)
    dom_embeddings = model.encode(dom_texts, normalize_embeddings=True)

    sim_matrix = cosine_similarity(req_embeddings, dom_embeddings)

    rows = []

    for i, req_id in enumerate(requirements_df["requirement_id"]):
        text = req_texts[i]
        scores = []

        for j, domain_name in enumerate(domains_df["domain_name"]):
            base = sim_matrix[i, j]
            final = base + heuristic_boost(text, domain_name)

            scores.append((domain_name, final))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        top1 = scores[0]
        top2 = scores[1] if len(scores) > 1 else (None, 0)

        score_gap = top1[1] - top2[1]

        confidence = get_confidence(top1[1], top2[1], text, top1[0])

        ambiguous = score_gap < 0.12 and confidence != "High"

        rationale = generate_rationale(text, top1[0], top2[0], score_gap)

        rows.append({
            "requirement_id": req_id,
            "suggested_domain": top1[0],
            "secondary_domain": top2[0],
            "confidence": confidence,
            "ambiguous": ambiguous,
            "rationale": rationale,
            "score_top1": round(top1[1], 4),
            "score_top2": round(top2[1], 4)
        })

    df = pd.DataFrame(rows)

    con.register("df_view", df)

    con.execute("""
        CREATE OR REPLACE TABLE domain_assignment_output AS
        SELECT * FROM df_view
    """)

    df.to_csv("domain_assignment_output.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()