import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------
# PATH FIX - Add project root to sys.path
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 1_THOUGHTS_CLASSIFIER
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import using full path
from classifier.src.insights.rules_engine import generate_insight


# -------------------------------------------------
# Resolve classifier root for configs/artifacts
# -------------------------------------------------

CLASSIFIER_ROOT = Path(__file__).resolve().parents[2]  # classifier folder


# -------------------------------------------------
# Load config (ONCE)
# -------------------------------------------------

def load_config():
    cfg_path = CLASSIFIER_ROOT / "configs" / "model.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()

ID2LABEL = {int(k): v for k, v in CFG["labels"]["id2label"].items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

MODEL_DIR = CLASSIFIER_ROOT / CFG["training"]["output_dir"] / "final"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

MODEL.to(DEVICE)
MODEL.eval()


# -------------------------------------------------
# Softmax helper
# -------------------------------------------------

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# -------------------------------------------------
# 🔥 RAG-CALLABLE FUNCTION
# -------------------------------------------------

def classify_thought(text: str) -> dict:
    """
    Run model inference on a single thought.

    Returns:
        {
          category: str,
          category_confidence: float,
          suggestion: str,
          model_version: str
        }
    """

    encoded = TOKENIZER(
        text,
        truncation=True,
        padding="max_length",
        max_length=CFG["tokenization"]["max_length"],
        return_tensors="pt",
    )

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = MODEL(**encoded)
        logits = outputs.logits.cpu().numpy()[0]

    probs = softmax(logits)
    pred_id = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label_name = ID2LABEL.get(pred_id, str(pred_id))

    # ---- RULE ENGINE ----
    suggestion = generate_insight(label_name, confidence)

    return {
        "category": label_name,
        "category_confidence": confidence,
        "suggestion": suggestion,
        "model_version": CFG["model"]["name"]
        if "model" in CFG and "name" in CFG["model"]
        else "deberta-v1",
    }


# -------------------------------------------------
# CLI MODE (kept for testing)
# -------------------------------------------------

def main():

    print("\nType a thought (or 'quit' to exit):\n")

    logs = []

    while True:

        text = input("> ").strip()

        if text.lower() in {"quit", "exit"}:
            break

        if not text:
            continue

        result = classify_thought(text)

        # -------------------------------------------------
        # Print result
        # -------------------------------------------------

        print("\nPrediction:")
        print("Label:", result["category"])
        print("Confidence:", round(result["category_confidence"], 4))
        print("Suggestion:", result["suggestion"])

        # -------------------------------------------------
        # Log result
        # -------------------------------------------------

        logs.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "thought_text": text,
                "predicted_label": result["category"],
                "confidence": result["category_confidence"],
                "suggestion": result["suggestion"],
            }
        )

    # -------------------------------------------------
    # Save logs
    # -------------------------------------------------

    if logs:
        out_dir = CLASSIFIER_ROOT / "artifacts" / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(logs)
        path = out_dir / "predictions_log.csv"

        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

        print("\nSaved predictions to:", path)

    print("\nInference session ended.")


if __name__ == "__main__":
    main()
