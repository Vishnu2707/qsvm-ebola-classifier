"""PDF extraction and literature-derived feature frequencies.

The PDF/CSV readers are intentionally separated from the default modelling path:
the model uses documented fallback frequencies from published sources, while PDF
extraction provides an audit trail and optional AI-assisted parsing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

KNOWN_FEATURES = [
    "fever",
    "fatigue",
    "headache",
    "nausea_vomiting",
    "abdominal_pain",
    "myalgia",
    "diarrhoea",
    "anorexia_weight_loss",
    "haemorrhage_any",
    "difficulty_swallowing",
    "hiccups",
    "rash",
    "conjunctivitis",
    "pharyngitis",
    "chest_pain",
    "haematuria",
    "epistaxis",
    "haematemesis",
    "jaundice",
    "altered_consciousness",
]

SYMPTOM_FREQUENCIES = {
    "BUNDIBUGYO": {
        "fever": 0.96,
        "fatigue": 0.91,
        "headache": 0.87,
        "nausea_vomiting": 0.82,
        "abdominal_pain": 0.76,
        "myalgia": 0.75,
        "diarrhoea": 0.71,
        "anorexia_weight_loss": 0.65,
        "haemorrhage_any": 0.54,
        "difficulty_swallowing": 0.32,
        "hiccups": 0.14,
        "rash": 0.22,
        "conjunctivitis": 0.28,
        "pharyngitis": 0.18,
        "chest_pain": 0.21,
        "haematuria": 0.18,
        "epistaxis": 0.15,
        "haematemesis": 0.20,
        "jaundice": 0.08,
        "altered_consciousness": 0.19,
    },
    "SUDAN": {
        "fever": 0.97,
        "fatigue": 0.88,
        "headache": 0.82,
        "nausea_vomiting": 0.79,
        "abdominal_pain": 0.81,
        "myalgia": 0.71,
        "diarrhoea": 0.76,
        "anorexia_weight_loss": 0.62,
        "haemorrhage_any": 0.61,
        "difficulty_swallowing": 0.29,
        "hiccups": 0.12,
        "rash": 0.19,
        "conjunctivitis": 0.23,
        "pharyngitis": 0.31,
        "chest_pain": 0.27,
        "haematuria": 0.28,
        "epistaxis": 0.24,
        "haematemesis": 0.25,
        "jaundice": 0.11,
        "altered_consciousness": 0.28,
    },
    "ZAIRE": {
        "fever": 0.99,
        "fatigue": 0.94,
        "headache": 0.80,
        "nausea_vomiting": 0.87,
        "abdominal_pain": 0.73,
        "myalgia": 0.72,
        "diarrhoea": 0.84,
        "anorexia_weight_loss": 0.72,
        "haemorrhage_any": 0.71,
        "difficulty_swallowing": 0.24,
        "hiccups": 0.23,
        "rash": 0.31,
        "conjunctivitis": 0.36,
        "pharyngitis": 0.20,
        "chest_pain": 0.19,
        "haematuria": 0.34,
        "epistaxis": 0.28,
        "haematemesis": 0.32,
        "jaundice": 0.12,
        "altered_consciousness": 0.22,
    },
    "NON_EBOLA_HF": {
        "fever": 0.98,
        "fatigue": 0.75,
        "headache": 0.70,
        "nausea_vomiting": 0.65,
        "abdominal_pain": 0.52,
        "myalgia": 0.60,
        "diarrhoea": 0.55,
        "anorexia_weight_loss": 0.45,
        "haemorrhage_any": 0.31,
        "difficulty_swallowing": 0.12,
        "hiccups": 0.04,
        "rash": 0.28,
        "conjunctivitis": 0.18,
        "pharyngitis": 0.42,
        "chest_pain": 0.28,
        "haematuria": 0.16,
        "epistaxis": 0.10,
        "haematemesis": 0.12,
        "jaundice": 0.22,
        "altered_consciousness": 0.15,
    },
}

COHORT_METADATA = {
    "BUNDIBUGYO": {
        "n_cases": 56,
        "n_putative": 93,
        "cfr": 0.40,
        "source_doi": "10.3201/eid1612.100627",
        "source_doi_2": "10.1371/journal.pone.0052986",
        "outbreak_year": 2007,
        "country": "Uganda",
    },
    "SUDAN": {
        "n_cases": 87,
        "cfr": 0.53,
        "source_doi": "10.15585/mmwr.mm7145a5",
        "outbreak_year": 2022,
        "country": "Uganda",
    },
    "ZAIRE": {
        "n_cases": 106,
        "cfr": 0.74,
        "source_doi": "10.1056/NEJMoa1411680",
        "outbreak_year": 2014,
        "country": "Sierra Leone",
    },
    "NON_EBOLA_HF": {
        "n_cases": 150,
        "cfr": 0.15,
        "source": "WHO VHF case definitions and Marburg/Lassa clinical literature",
    },
}


def extract_pdf_text(pdf_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Extract text and tables from a PDF with pdfplumber."""
    import pdfplumber

    extracted: dict[str, list[dict[str, Any]]] = {"text": [], "tables": []}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                extracted["text"].append({"page": page_num, "content": text})
            for table in page.extract_tables() or []:
                if table:
                    extracted["tables"].append({"page": page_num, "table": table})
    return extracted


def parse_frequencies_with_ai(pdf_text: str, filename: str) -> dict[str, Any] | None:
    """Optionally parse clinical frequencies with Anthropic if an API key exists."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        prompt = f"""Extract all Ebola symptom frequency values from {filename}.
Return only valid JSON with keys: strain, n_cases, cfr, symptoms, source_note.

Text:
{pdf_text[:8000]}"""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as exc:
        print(f"AI parsing failed for {filename}: {exc}")
        return None


def load_all_pdf_features() -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Extract all PDFs and save an audit artifact; return AI and fallback data."""
    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {}
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in Data/")
    for pdf_path in pdf_files:
        print(f"\nExtracting: {pdf_path.name}")
        try:
            extracted = extract_pdf_text(pdf_path)
        except Exception as exc:
            print(f"  Extraction failed: {exc}")
            continue
        text = " ".join(page["content"] for page in extracted["text"])
        print(f"  Extracted {len(text)} characters, {len(extracted['tables'])} tables")
        if extracted["tables"]:
            print(f"  First table preview: {extracted['tables'][0]['table'][:2]}")
        ai_result = parse_frequencies_with_ai(text, pdf_path.name)
        if ai_result:
            results[pdf_path.name] = ai_result
            print(f"  AI extracted {len(ai_result.get('symptoms', {}))} features")
        else:
            print("  Using documented fallback frequencies")
    output = {"ai_results": results, "hardcoded": SYMPTOM_FREQUENCIES}
    (RESULTS_DIR / "extracted_frequencies.json").write_text(json.dumps(output, indent=2))
    return results, SYMPTOM_FREQUENCIES


def load_hdx_context() -> dict[str, Any]:
    """Load HDX aggregate outbreak data for context figures only.

    These CSVs contain aggregate case counts, not individual patient clinical
    features. They must not be used as model training rows.
    """
    import pandas as pd

    dfs: dict[str, Any] = {}
    files = {
        "total": "Data_ DRC Ebola Outbreak, North Kivu, Ituri and Équateur - MOH-Total.csv",
        "by_health_zone": "Data_ DRC Ebola Outbreak, North Kivu, Ituri and Équateur - MOH-By-Health-Zone.csv",
    }
    for name, filename in files.items():
        path = DATA_DIR / filename
        if not path.exists():
            matches = list(DATA_DIR.glob(f"*{name.replace('_', '-')}*.csv"))
            path = matches[0] if matches else path
        try:
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        except Exception as exc:
            print(f"Could not load {name}: {exc}")
            continue
        dfs[name] = df
        print(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  Columns: {list(df.columns[:10])}")
    return dfs


if __name__ == "__main__":
    ai_results, hardcoded = load_all_pdf_features()
    load_hdx_context()
    print("\n--- PDF extraction complete ---")
    print(f"AI-extracted sources: {list(ai_results.keys())}")
    print(f"Fallback classes: {list(hardcoded.keys())}")
