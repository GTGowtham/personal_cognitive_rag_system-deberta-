import sys
from pathlib import Path

from transformers import AutoConfig, AutoModelForSequenceClassification


# -------------------------------------------------
# PATH FIX (for consistency, though not required here)
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 1_THOUGHTS_CLASSIFIER
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_model(cfg):
    """
    Build and return a sequence classification model
    using the config dictionary loaded from YAML.
    """

    model_name = cfg["model"]["pretrained_name"]
    num_labels = cfg["model"]["num_labels"]

    print(f"Loading base model: {model_name}")
    print(f"Num labels: {num_labels}")

    # Load base config and override label count
    hf_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=hf_config,
    )

    return model
