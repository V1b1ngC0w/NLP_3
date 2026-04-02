from datasets import Dataset, load_dataset
import os
import gc
import time

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math

from preprocessing_normalisation import preprocess, normalise
from utils import (
    SEED,
    BATCH_SIZE,
    LABELS,
    LSTM_tokenizer,
    transformer_tokenizer,
    tokenize_data,
    calculate_metrics,
    plot_learning_curves,
)
from models import LSTMModel, train_LSTM, train_transformer, get_predictions

from transformers import AutoModelForSequenceClassification

# Try to use the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

def run_tokenizer(
        tokenizer,
        train,
        dev,
        test,
        ) -> tuple:
    
    print("Tokenizing datasets...")

    X_train = tokenize_data(train, tokenizer)
    X_dev = tokenize_data(dev, tokenizer)
    X_test = tokenize_data(test, tokenizer)

    return X_train, X_dev, X_test


def main() -> None:
    bypass = input("Do you want to train transformer models from scratch? Y/n ").lower()

    # download the AG news dataset
    dataset_dict = load_dataset("sh0416/ag_news")
    """
    Dataset downloads the test and train split separately
    It save them in a dictionary where the first element
    is the train dataset, and the second one is the test dataset

    they are both Dataset objects containing:
        features:  label title description
        num_rows: int
    """
    # Preprocessing
    train_set = dataset_dict["train"].to_pandas()
    test_set = dataset_dict["test"].to_pandas()

    # Add the lables back
    train_set["label_text"] = train_set["label"].map(LABELS)
    test_set["label_text"] = test_set["label"].map(LABELS)

    train_set["description"] = train_set["description"].apply(preprocess).apply(normalise)
    train_set["title"] = train_set["title"].apply(preprocess).apply(normalise)
    train_title = train_set["title"]
    train_full = train_set["title"] + train_set["description"] 

    test_set["description"] = test_set["description"].apply(preprocess).apply(normalise)
    test_set["title"] = test_set["title"].apply(preprocess).apply(normalise)
    test_title = test_set["title"]
    test_full = test_set["title"] + " " + test_set["description"]
    test_set["text"] = test_full


    print("Loaded data head (before split):")
    print(train_full.head(5))
    print("TITLE ONLY")
    print(train_set["title"].head(5))

    # splits for normal models
    train, dev = train_test_split(
        train_full,
        test_size=0.16,
        random_state=SEED,
        stratify=train_set["label"]
    )

    # splits for the title only model
    train_title_only, dev_title_only = train_test_split(
        train_title,
        test_size=0.16,
        random_state=SEED,
        stratify=train_set["label"],
    )

    print(f"Split sizes: Train({len(train)}), Dev({len(dev)}), Test({len(test_full)})")

    LSTM_X_train, LSTM_X_dev, LSTM_X_test = run_tokenizer(LSTM_tokenizer, train, dev, test_full)
    tr_X_train, tr_X_dev, tr_X_test = run_tokenizer(transformer_tokenizer, train, dev, test_full)

    # Y labels
    y_train = torch.tensor(train_set.loc[train.index, "label"].values) - 1
    y_dev = torch.tensor(train_set.loc[dev.index, "label"].values) - 1
    y_test = torch.tensor(test_set["label"].values) - 1

    # train the LSTM

    # Batching needed only for LSTM
    train_dataset = TensorDataset(LSTM_X_train['input_ids'], LSTM_X_train['attention_mask'], y_train)
    dev_dataset = TensorDataset(LSTM_X_dev['input_ids'], LSTM_X_dev['attention_mask'], y_dev)
    test_dataset = TensorDataset(LSTM_X_test['input_ids'], LSTM_X_test['attention_mask'], y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(
        vocab_size=LSTM_tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=4,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
        )
    
    trained_model, hist = train_LSTM(model, train_loader, dev_loader)
    plot_learning_curves(hist, "LSTM")
    test_predictions = get_predictions(trained_model, test_loader)
    test_predictions = [p + 1 for p in test_predictions]
    calculate_metrics(test_set, test_predictions, "LSTM", True)
    gc.collect()
    time.sleep(5)

    # train the Transformer

    # Tansformer needs the labels and values in the same dataset
    tr_train_dataset = Dataset.from_dict({
        "input_ids": tr_X_train["input_ids"],
        "attention_mask": tr_X_train["attention_mask"],
        "labels": y_train
    })

    tr_dev_dataset = Dataset.from_dict({
        "input_ids": tr_X_dev["input_ids"],
        "attention_mask": tr_X_dev["attention_mask"],
        "labels": y_dev
    })

    tr_test_dataset = Dataset.from_dict({
        "input_ids": tr_X_test["input_ids"],
        "attention_mask": tr_X_test["attention_mask"],
        "labels": y_test
    })

    print("\n=== Starting Label Noise Sensitivity Experiment ===")
    # Label noise sensitivity
    percentage_sizes = [0.2, 0.4, 0.6, 0.8, 1]
    train_length = len(tr_train_dataset)

    for p in percentage_sizes:

        p_label = int(p * 100)
        MODEL_PATH = f"./saved_roberta_scale_{p_label}"

        print(f"Training with {p*100}% of the dataset")
        #change trainset size
        train_size = math.floor(p*train_length)
        subset_train = tr_train_dataset.shuffle(seed=SEED).select(range(train_size))

        # check if model has already been trained
        if os.path.exists(MODEL_PATH) and bypass == "n":
            print(f"Found saved model at {MODEL_PATH}! Skipping training...")
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
            trainer = train_transformer(model, subset_train, tr_dev_dataset)
        
        else:
            print("No saved model found. Training from scratch...")
            model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=4
            ).to(device)
            trainer = train_transformer(model, subset_train, tr_dev_dataset)
            trainer.train()
            trainer.save_model(MODEL_PATH)
            transformer_tokenizer.save_pretrained(MODEL_PATH)

        prediction_output = trainer.predict(tr_test_dataset)
        logits = prediction_output.predictions
        roberta_preds = np.argmax(logits, axis=1).tolist()
        roberta_preds = [p + 1 for p in roberta_preds]
        calculate_metrics(
            real=test_set,
            pred=roberta_preds,
            model=f"RoBERTa {p*100}% of the dataset",
            print_err= (p==1)
        )

        del trainer
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)

    print("\n=== Starting Title-Only Ablation Experiment ===")

    # Tokenize the title-only datasets
    tr_X_train_title = tokenize_data(train_title_only, transformer_tokenizer)
    tr_X_dev_title = tokenize_data(dev_title_only, transformer_tokenizer)
    tr_X_test_title = tokenize_data(test_title, transformer_tokenizer)

    # Tansformer needs the labels and values in the same dataset
    title_train_dataset = Dataset.from_dict({
        "input_ids": tr_X_train_title["input_ids"],
        "attention_mask": tr_X_train_title["attention_mask"],
        "labels": y_train
    })
    title_dev_dataset = Dataset.from_dict({
        "input_ids": tr_X_dev_title["input_ids"],
        "attention_mask": tr_X_dev_title["attention_mask"],
        "labels": y_dev
    })
    title_test_dataset = Dataset.from_dict({
        "input_ids": tr_X_test_title["input_ids"],
        "attention_mask": tr_X_test_title["attention_mask"],
        "labels": y_test
    })

    TITLE_MODEL_PATH = "./saved_roberta_title_only"

    if os.path.exists(TITLE_MODEL_PATH) and bypass == "n":
        print(f"Found saved title-only model at {TITLE_MODEL_PATH}! Skipping training...")
        title_model = AutoModelForSequenceClassification.from_pretrained(TITLE_MODEL_PATH).to(device)
        title_trainer = train_transformer(title_model, title_train_dataset, title_dev_dataset)

    else:
        print("No saved model found. Training from scratch...")
        title_model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", 
            num_labels=4
        ).to(device)
        title_trainer = train_transformer(title_model, title_train_dataset, title_dev_dataset)
        title_trainer.train()
        title_trainer.save_model(TITLE_MODEL_PATH)
        transformer_tokenizer.save_pretrained(TITLE_MODEL_PATH)

    title_prediction_output = title_trainer.predict(title_test_dataset)
    title_logits = title_prediction_output.predictions
    title_preds = np.argmax(title_logits, axis=1).tolist()
    title_preds = [p + 1 for p in title_preds]
    calculate_metrics(real=test_set, pred=title_preds, model="RoBERTa (Title Only)")

    del title_trainer
    del title_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
