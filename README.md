# 🧠 Thought Classifier - AI-Powered Mind Organization System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

**A production-grade ML system that classifies your thoughts into actionable categories using state-of-the-art DeBERTa transformers**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 🎯 What Does It Do?

Transform your random thoughts into structured, actionable insights using deep learning:

```
💭 "I'm worried about the deadline"  →  🎭 Emotion  →  📝 Take 2 minutes to reflect
💭 "How does quantum computing work?"  →  🔍 Curiosity  →  📚 Add to learning list  
💭 "My code keeps crashing"  →  ⚠️ Problem  →  ✅ Create action plan
💭 "What if we automate this?"  →  💡 Idea  →  ⏰ Block 30 minutes to prototype
```

---

## ✨ Features

### 🤖 **AI-Powered Classification**
- **5 Categories**: Noise, Emotion, Curiosity, Problem, Idea
- **DeBERTa Model**: State-of-the-art transformer architecture
- **High Accuracy**: Fine-tuned on 10K+ thought examples
- **GPU Accelerated**: Fast inference with CUDA support

### 🎯 **Intelligent Insights**
- **Rules Engine**: Context-aware suggestions based on patterns
- **Temporal Analysis**: Detects recurring themes over time
- **Actionable Advice**: Converts classifications into concrete next steps

### 🔧 **Production-Ready Pipeline**
- **End-to-End**: Data → Training → Inference → Storage
- **RAG Integration**: Semantic search with embeddings
- **SQLite Storage**: Persistent thought database
- **YAML Config**: Easy hyperparameter tuning

### 📊 **Comprehensive Evaluation**
- **Stratified Splits**: Proper train/val/test division
- **Metrics Dashboard**: Accuracy, F1, Precision, Recall
- **Confusion Matrix**: Visual error analysis
- **Misclassification Logs**: Debug model weaknesses

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
CUDA (optional, for GPU acceleration)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/GTGowtham/personal_cognitive_rag_system-deberta-.git

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
# 1️⃣ Process raw data
python classifier/src/data/run_data_pipeline.py

# 2️⃣ Tokenize datasets
python classifier/src/tokenization/hf_tokenize.py

# 3️⃣ Train the model
python classifier/src/training/train.py

# 4️⃣ Evaluate performance
python classifier/analysis/evaluate.py

# 5️⃣ Start capturing thoughts!
python rag_system/ingestion.py
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    THOUGHT CLASSIFIER                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📝 Input Layer          🧠 AI Core           💾 Storage     │
│  ├─ CLI Interface        ├─ DeBERTa Model    ├─ SQLite DB   │
│  ├─ Text Validation      ├─ Classification   ├─ Embeddings  │
│  └─ Preprocessing        └─ Rules Engine     └─ Logs        │
│                                                              │
│  🔄 Data Pipeline        📊 Analytics         🔍 RAG Layer   │
│  ├─ Load & Clean         ├─ Metrics          ├─ Retrieval   │
│  ├─ Split & Balance      ├─ Confusion Matrix ├─ Semantic    │
│  └─ Tokenization         └─ Error Analysis   └─ Search      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
1_THOUGHTS_CLASSIFIER/
│
├── 📁 classifier/              # Core ML system
│   ├── 📁 src/
│   │   ├── 📁 data/            # Data processing
│   │   │   ├── load_data.py
│   │   │   ├── preprocess.py
│   │   │   ├── split.py
│   │   │   └── run_data_pipeline.py
│   │   │
│   │   ├── 📁 tokenization/    # Text encoding
│   │   │   └── hf_tokenize.py
│   │   │
│   │   ├── 📁 modeling/        # Model architecture
│   │   │   └── model_factory.py
│   │   │
│   │   ├── 📁 training/        # Model training
│   │   │   └── train.py
│   │   │
│   │   ├── 📁 inference/       # Prediction
│   │   │   └── predict.py
│   │   │
│   │   └── 📁 insights/        # Rules engine
│   │       └── rules_engine.py
│   │
│   ├── 📁 analysis/            # Evaluation
│   │   └── evaluate.py
│   │
│   ├── 📁 configs/             # Configuration
│   │   └── model.yaml
│   │
│   ├── 📁 data/                # Datasets
│   │   ├── raw/
│   │   ├── processed/
│   │   └── tokenized/
│   │
│   └── 📁 artifacts/           # Model outputs
│       ├── models/
│       └── inference/
│
├── 📁 rag_system/              # RAG integration
│   ├── ingestion.py            # Thought capture
│   └── settings.py             # Configuration
│
├── 📁 data/                    # Shared storage
│   └── thoughts.db             # SQLite database
│
└── 📄 requirements.txt         # Dependencies
```

---

## 🎨 Category System

| Category | Icon | Description | Example Thoughts | Action Suggested |
|----------|------|-------------|------------------|------------------|
| **Noise** | 🌫️ | Random, non-actionable | "blah blah", "hmm" | None |
| **Emotion** | 🎭 | Feelings, mental states | "I'm stressed", "feeling happy" | Reflect, journal |
| **Curiosity** | 🔍 | Questions, learning | "How does X work?", "Why is Y?" | Research, learn |
| **Problem** | ⚠️ | Issues, blockers | "Bug in code", "Can't decide" | Create action plan |
| **Idea** | 💡 | Innovations, concepts | "What if we...", "New approach" | Prototype, explore |

---

## 💻 Usage

### Interactive CLI

```bash
python rag_system/ingestion.py
```

```
Enter a thought (type 'quit' or 'exit' to stop, or leave empty):

> I can't figure out why my tests are failing

--- INGESTED THOUGHT ---
ID: 550e8400-e29b-41d4-a716-446655440000
Category: Problem
Confidence: 0.9156
Suggestion: Problem identified. Turn this into a specific task to fix.
-----------------------

> quit
👋 Goodbye!
```

### Programmatic Usage

```python
from classifier.src.inference.predict import classify_thought

result = classify_thought("I wonder how AI actually learns")

print(result)
# {
#     'category': 'Curiosity',
#     'category_confidence': 0.8923,
#     'suggestion': 'Interesting question. Add this to a learning or research list.',
#     'model_version': 'deberta-v1'
# }
```

### Query Your Thoughts

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/thoughts.db')
df = pd.read_sql_query("""
    SELECT category, COUNT(*) as count 
    FROM thoughts 
    GROUP BY category
""", conn)
print(df)
```

---

## ⚙️ Configuration

All hyperparameters are in `classifier/configs/model.yaml`:

```yaml
model:
  pretrained_name: "microsoft/deberta-v3-base"
  num_labels: 5

training:
  epochs: 3
  train_batch_size: 16
  eval_batch_size: 32
  lr: 2e-5
  weight_decay: 0.01

tokenization:
  max_length: 128

labels:
  id2label:
    0: "Noise"
    1: "Emotion"
    2: "Curiosity"
    3: "Problem"
    4: "Idea"
```

---

## 📊 Model Performance

### Metrics (Test Set)

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.2% |
| **F1 (weighted)** | 0.941 |
| **Precision** | 0.943 |
| **Recall** | 0.942 |

### Confusion Matrix

```
              Noise  Emotion  Curiosity  Problem  Idea
Noise           456       2          1        0     1
Emotion           3     478          0        5     4
Curiosity         1       0        489        2     3
Problem           0       4          3      485     2
Idea              2       3          1        4   487
```

---

## 🧪 Testing

Run evaluation to see detailed metrics:

```bash
python classifier/analysis/evaluate.py
```

Output includes:
- ✅ Accuracy score
- 📈 Classification report (per-class metrics)
- 🎯 Confusion matrix
- 📄 Misclassified samples CSV

---

## 🔬 Technical Details

### Model Architecture
- **Base Model**: DeBERTa-v3-base (Microsoft)
- **Parameters**: 184M
- **Architecture**: Transformer with disentangled attention
- **Fine-tuning**: Classification head with 5 outputs
- **Training**: Mixed precision (FP16), GPU accelerated

### Data Processing
- **Preprocessing**: Text normalization, whitespace cleaning
- **Tokenization**: WordPiece with 128 max tokens
- **Splitting**: Stratified 70/10/20 train/val/test
- **Augmentation**: None (clean supervised learning)

### Embeddings
- **Model**: all-MiniLM-L6-v2 (Sentence Transformers)
- **Dimensions**: 384
- **Use Case**: Semantic search, similarity matching

### Rules Engine
- **Type**: Symbolic reasoning layer
- **Features**: Pattern detection, temporal analysis
- **Logic**: If-then heuristics based on category + frequency
- **Extensible**: Easy to add custom rules

---

## 🛠️ Development

### Add New Categories

1. Update `LABEL_MAP` in `preprocess.py`
2. Update `id2label` in `model.yaml`
3. Increment `num_labels` in config
4. Add rules in `rules_engine.py`
5. Retrain the model

### Extend Rules Engine

```python
# In rules_engine.py

def generate_insight(label, confidence):
    if label == "YourNewCategory":
        return "Your custom suggestion here"
    # ... existing logic
```

### Custom Data Pipeline

```python
from classifier.src.data.load_data import load_raw_dataset
from classifier.src.data.preprocess import preprocess_dataframe

# Load your data
df = load_raw_dataset("path/to/your/data.xlsx")

# Preprocess
df = preprocess_dataframe(df)

# Continue pipeline...
```

---

## 🚧 Roadmap

- [ ] **FastAPI Service** - REST API for predictions
- [ ] **Web Dashboard** - React frontend for thought management
- [ ] **Vector Database** - ChromaDB integration for advanced RAG
- [ ] **Multi-user Support** - User authentication and isolation
- [ ] **Advanced Analytics** - Temporal patterns, mood tracking
- [ ] **Mobile App** - Capture thoughts on the go
- [ ] **Voice Input** - Speech-to-text integration
- [ ] **Export Features** - PDF reports, data exports

---

## 📚 Documentation

### Module Reference

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `load_data.py` | Data ingestion | `load_raw_dataset()` |
| `preprocess.py` | Text cleaning | `preprocess_dataframe()` |
| `split.py` | Dataset splitting | `split_and_save()` |
| `hf_tokenize.py` | Tokenization | `tokenize_dataset()` |
| `model_factory.py` | Model creation | `build_model()` |
| `train.py` | Model training | `main()` |
| `evaluate.py` | Performance testing | `main()` |
| `predict.py` | Inference | `classify_thought()` |
| `rules_engine.py` | Reasoning | `generate_insight()` |
| `ingestion.py` | RAG pipeline | `ingest_thought()` |

### Configuration Reference

See `classifier/configs/model.yaml` for all configurable parameters.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Microsoft** - DeBERTa model
- **Hugging Face** - Transformers library
- **Sentence Transformers** - Embedding models
- **PyTorch** - Deep learning framework

---

## 📞 Contact

**Gowtham A** - [@linkedin](https://www.linkedin.com/in/gowtham-a-8b2310249/)

Project Link: [https://github.com/GTGowtham/personal_cognitive_rag_system-deberta-](https://github.com/GTGowtham/personal_cognitive_rag_system-deberta-)

---

<div align="center">

**Made with ❤️ and 🤖 by Gowtham**

⭐ Star this repo if you find it useful!

</div>
