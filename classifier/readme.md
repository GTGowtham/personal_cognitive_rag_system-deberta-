# Thought Classifier with DeBERTa --- Full Technical README

## Overview

This project implements a production-style machine learning system that
classifies short human thoughts into five semantic categories: Noise,
Emotion, Curiosity, Problem, and Idea.

It includes: - Data ingestion and validation - Text preprocessing -
Train/validation/test splitting - HuggingFace tokenization - Transformer
fine-tuning - Evaluation metrics - Interactive inference - Rule-based
reasoning engine - YAML-driven configuration

------------------------------------------------------------------------

## Pipeline Flow

Raw Excel Data → Cleaning → Splitting → Tokenization → Training →
Evaluation → Inference → Rules Engine → Logging

------------------------------------------------------------------------

## Directory Structure

src/ data/ tokenization/ modeling/ training/ analysis/ inference/
insights/ configs/ requirements.txt README.md

------------------------------------------------------------------------

## model.yaml Explained

This file controls all hyperparameters and paths.

Sections: - model: pretrained model name and number of labels -
tokenization: max token length - data: column names and storage paths -
training: batch size, epochs, learning rate, output folders - labels:
numeric → string mapping - seed: reproducibility

Each value is read at runtime using PyYAML.

------------------------------------------------------------------------

# Module Documentation

------------------------------------------------------------------------

## load_data.py

Purpose: Load raw Excel file and validate schema.

Imports: - pandas for DataFrame operations - pathlib and os for file
paths

Key Steps: - Resolve project root - Check file existence - Read Excel -
Validate required columns - Ensure dataset not empty

Function: load_raw_dataset(path: str \| Path) → pandas.DataFrame

Raises: - FileNotFoundError - ValueError for schema mismatch

------------------------------------------------------------------------

## preprocess.py

Purpose: Clean text and convert labels to integers.

LABEL_MAP defines supervised target mapping.

Text Cleaning: - cast to string - strip whitespace - normalize multiple
spaces

Label Handling: - verify allowed labels - map to numeric IDs

Returns cleaned DataFrame.

------------------------------------------------------------------------

## split.py

Purpose: Stratified dataset splitting.

Uses sklearn train_test_split.

Process: - split test first - split validation from train - maintain
label proportions

Saves CSV files.

Parameters: df, output_dir, test_size, val_size, random_state

------------------------------------------------------------------------

## run_data_pipeline.py

Purpose: Orchestrate full data pipeline.

Sequentially calls: - load_raw_dataset - preprocess_dataframe -
split_and_save

Acts as reproducible ETL entrypoint.

------------------------------------------------------------------------

## hf_tokenize.py

Purpose: Tokenize datasets using HuggingFace.

Libraries: - datasets - transformers - yaml

Loads config from YAML.

Uses AutoTokenizer.from_pretrained.

Tokenization arguments: - truncation=True - padding="max_length" -
max_length

Saves: - Arrow datasets - tokenizer files

------------------------------------------------------------------------

## model_factory.py

Purpose: Create Transformer classifier.

Uses: - AutoConfig - AutoModelForSequenceClassification

Overrides num_labels to match task.

Encapsulates model creation logic.

------------------------------------------------------------------------

## train.py

Purpose: Fine-tune DeBERTa.

Features: - seed control - dataset loading - metric computation - fp16
mixed precision - Trainer API - checkpointing

TrainingArguments: - output_dir - batch sizes - epochs - learning_rate -
evaluation strategy - save strategy - load_best_model_at_end

Trainer handles backprop, optimization, logging.

------------------------------------------------------------------------

## evaluate.py

Purpose: Run final test evaluation.

Loads trained model.

Uses torch DataLoader.

Computes: - accuracy - precision/recall/F1 - confusion matrix

Writes misclassified samples CSV.

------------------------------------------------------------------------

## predict.py

Purpose: Interactive CLI inference.

Process: - load model/tokenizer - encode text - forward pass - softmax -
id→label mapping - call rules engine - log results

Stores inference logs in CSV.

------------------------------------------------------------------------

## rules_engine.py

Purpose: Add reasoning layer.

Loads inference history.

Counts recent predictions by label.

Implements heuristics: - repeated emotions escalate reflection -
problems trigger action plans - ideas encourage brainstorming

Combines ML output with symbolic logic.

------------------------------------------------------------------------

## Engineering Principles

-   Config-driven
-   Modular design
-   GPU accelerated
-   Reproducible seeds
-   Dataset versioning
-   Logging
-   Error analysis
-   Extendable to RAG

------------------------------------------------------------------------

## Future Work

-   RAG integration
-   Vector databases
-   FastAPI service
-   UI dashboards
-   SQLite logging
-   Knowledge bases
-   Agent routing

------------------------------------------------------------------------

## Running the Project

1.  pip install -r requirements.txt
2.  python -m src.data.run_data_pipeline
3.  python -m src.tokenization.hf_tokenize
4.  python -m src.training.train
5.  python -m src.analysis.evaluate
6.  python -m src.inference.predict

------------------------------------------------------------------------

## Conclusion

This system is a full-stack ML pipeline, not a notebook experiment. It
demonstrates reproducible engineering practices, scalable architecture,
and hybrid AI reasoning suitable for production and RAG expansion.
