from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta


# -------------------------------------------------
# Resolve roots
# -------------------------------------------------

# Path to classifier folder (for artifacts)
CLASSIFIER_ROOT = Path(__file__).resolve().parents[2]


# -------------------------------------------------
# Load inference history safely
# -------------------------------------------------

def load_history():
    path = CLASSIFIER_ROOT / "artifacts" / "inference" / "predictions_log.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    return df


# -------------------------------------------------
# Frequency / pattern detection
# -------------------------------------------------

def count_recent(label_name, hours=48):
    df = load_history()

    if df.empty or "predicted_label" not in df.columns:
        return 0

    cutoff = datetime.utcnow() - timedelta(hours=hours)

    recent = df[df["timestamp"] >= cutoff]

    return (recent["predicted_label"] == label_name).sum()


# -------------------------------------------------
# Core rules engine
# -------------------------------------------------

def generate_insight(label, confidence):

    # Low confidence fallback
    if confidence < 0.6:
        return "Low confidence prediction. Consider rephrasing the thought."

    recent_count = count_recent(label)

    if label == "Noise":
        return "Likely random or non-actionable thought. No action suggested."

    if label == "Emotion":
        if recent_count >= 5:
            return (
                "Repeated emotional signals detected recently. "
                "Consider journaling or reflecting on patterns."
            )
        return "Emotional state noted. Take 2 minutes to reflect or write it down."

    if label == "Curiosity":
        return "Interesting question. Add this to a learning or research list."

    if label == "Problem":
        if recent_count >= 3:
            return "Recurring problem detected. Create a concrete action plan to address it."
        return "Problem identified. Turn this into a specific task to fix."

    if label == "Idea":
        if recent_count >= 3:
            return (
                "Multiple ideas appearing recently. "
                "Schedule a focused brainstorming or planning session."
            )
        return "Promising idea. Block 30 minutes to explore or prototype it."

    return "No rule matched."
