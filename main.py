from datasets import Dataset, load_dataset

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

    X_train = tokenize_data(train["text"], tokenizer)
    X_dev = tokenize_data(dev["text"], tokenizer)
    X_test = tokenize_data(test["text"], tokenizer)

    return X_train, X_dev, X_test


def main() -> None:

    # download the AG news dataset
    dataset_dict = load_dataset("SetFit/ag_news")
    """
    Dataset downloads the test and train split separately
    It save them in a dictionary where the first element
    is the train dataset, and the second one is the test dataset

    they are both Dataset objects containing:
        features: text label label_text
        num_rows: int
    """
    # Preprocessing
    train_full = dataset_dict["train"].to_pandas()
    test = dataset_dict["test"].to_pandas()
    train_full["text"] = train_full["text"].apply(preprocess).apply(normalise)
    test["text"] = test["text"].apply(preprocess).apply(normalise)

    print("Loaded data head (before split):")
    print(train_full.head(5))

    # split the training dataset into train and validation
    train, dev = train_test_split(
        train_full,
        test_size=0.16,
        random_state=SEED,
        stratify=train_full["label"]
    )

    print(f"Split sizes: Train({len(train)}), Dev({len(dev)}), Test({len(test)})")

    LSTM_X_train, LSTM_X_dev, LSTM_X_test = run_tokenizer(LSTM_tokenizer, train, dev, test)
    tr_X_train, tr_X_dev, tr_X_test = run_tokenizer(transformer_tokenizer, train, dev, test)

    # Y labels
    y_train = torch.tensor(train["label"].values)
    y_dev = torch.tensor(dev["label"].values)
    y_test = torch.tensor(test["label"].values)

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
    calculate_metrics(test, test_predictions, "LSTM")


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

    model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=4
            ).to(device)
    
    # Label noise sensitivity
    percentage_sizes = [0.2, 0.4, 0.6, 0.8, 1]
    train_length = len(tr_train_dataset)

    for p in percentage_sizes:
        print(f"Training with {p*100}% of the dataset")

        #change trainset size
        train_size = math.floor(p*train_length)
        subset_train = tr_train_dataset.shuffle(seed=SEED).select(range(train_size))

        trainer = train_transformer(model, subset_train, tr_dev_dataset)
        trainer.train()

        prediction_output = trainer.predict(tr_test_dataset)
        logits = prediction_output.predictions
        roberta_preds = np.argmax(logits, axis=1).tolist()
    
        calculate_metrics(real=test, pred=roberta_preds, model=f"RoBERTa {p*100}% of the dataset")

        del trainer
        torch.cuda.empty_cache()
    
    


if __name__ == "__main__":
    main()
