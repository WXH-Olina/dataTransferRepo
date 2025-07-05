"""
For testing and training the model.
4th attempt: Use nonlinear classifier + dropout layer.
"""

import os
import shutil
from zipfile import ZipFile
import requests
import io

import numpy as np
import pandas as pd
import torch

import torch.nn as nn
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Resize,
    GaussianBlur,
    RandomAdjustSharpness,
    RandomEqualize,
    ToTensor,
)
from transformers import TrainingArguments, Trainer
from transformers import ConvNextV2ForImageClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification

from sklearn.metrics import balanced_accuracy_score
from typing import Dict

CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
FULL_DATASET_DIR = "./data/DetectChineseCharacters"
SAVE_DIR = "./models/model4"
# Just colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

class model_ReLU(nn.Module):
    def __init__(
        self,
        num_labels: int,
        id2label: Dict,
        label2id: Dict,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
        checkpoint=None,
    ) -> None:
        super().__init__()
        self.CHECKPOINT = (
            "facebook/convnextv2-tiny-1k-224" if checkpoint is None else checkpoint
        )
        self.loss_fn = loss_fn
        self.convNeXt = ConvNextV2ForImageClassification.from_pretrained(
            CHECKPOINT,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            output_hidden_states=True,
        )
        # Instead of add topping layers on ConvNextV2Model, try to modify the classifier layer.
        self.convNeXt.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Softmax(dim=-1),
            nn.Linear(in_features=768, out_features=num_labels),
        )

    def forward(self, pixel_values, labels=None):
        logits = self.convNeXt(pixel_values).logits

        if labels is None:
            return logits
        else:
            predictedLabels = logits
            loss = self.loss_fn(
                predictedLabels, labels
            )  # BCEWithLogitsLoss expect raw logits
            return {"loss": loss, "logits": logits}


def img_transforms(examples):
    examples["pixel_values"] = [
        transforms(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"balanced_accuracy": balanced_accuracy_score(labels, preds)}


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


if __name__ == "main":
    # Download data
    try:
        os.mkdir("./data")
        print("Data folder created.")
    except FileExistsError as e:
        print("Data folder exists.")

    if os.path.exits("./data/DetectChineseCharacters"):
        print("Dataset folder exists. Jump to model training!")
    else:
        data = requests.get(
            "https://github.com/WXH-Olina/dataTransfer/raw/refs/heads/main/DetectChineseCharacters.zip"
        )
        z = ZipFile(io.BytesIO(data.content))
        z.extractall("./data")

    # Create dataset
    full_dataset = load_dataset(
        "imagefolder", data_dir="/content/data/DetectChineseCharacters", split="train"
    )
    full_dataset = full_dataset.train_test_split(test_size=0.3)

    labels = full_dataset["train"].features["label"].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    # Create preprocessor
    img_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    SIZE = (
        img_processor.size["shortest_edge"]
        if "shortest_edge" in img_processor.size
        else (img_processor.size["height"], img_processor.size["width"])
    )

    transforms = Compose([
        Resize(size=SIZE, antialias=True),
        GaussianBlur(kernel_size=(1, 5)),
        RandomAdjustSharpness(sharpness_factor=2),
        RandomEqualize(),
        ToTensor(),
    ])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The model will be trained on {DEVICE}.")

    full_dataset = full_dataset.with_transform(img_transforms)
    full_dataset = full_dataset  # datasets use GPU as long as it's avaible. So it doesn't provide `to(device)` attribute here.
    model4 = model_ReLU(len(labels), id2label, label2id).to(DEVICE)
    STRATEGY = "epoch"
    OUTPUT_DIR = "./convnextv2-tiny-4"  # Output for 4th attempt

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy=STRATEGY,
        save_strategy=STRATEGY,
        logging_steps=10,
        remove_unused_columns=False,
        learning_rate=1e-5,  # Use 1e-5 instead of 0.001
        per_device_train_batch_size=16,  # 8 is too slow, 32 is too coarse
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=25,
        warmup_ratio=0.1,
        metric_for_best_model="balanced_accuracy",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model4,  # training model4
        args=training_args,
        data_collator=collate_fn,
        train_dataset=full_dataset["train"],
        eval_dataset=full_dataset["test"],
        processing_class=img_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(f"{GREEN}Success!{RESET} Model Training completed.")

    # Prepare save_path
    try:
        os.mkdir("./models")
        print("./models folder created.")
    except FileExistsError:
        print("./models folder exists.")
    save_path = SAVE_DIR
    if os.path.exists(SAVE_DIR):
        option = input(f"{SAVE_DIR} exists, do you want to overwrite it? yes|NO")
        if option=="yes":
            shutil.rmtree(SAVE_DIR)
            os.mkdir(SAVE_DIR)
        else:
            save_path = input("Suggest a new path to save model: ")
            while True:
                try:
                    os.mkdir(save_path)
                    break
                except FileExistsError:
                    save_path = input(f"{save_path} exists. Suggest a new one: ")
    
    trainer.save_model(save_path)
    print(f"{GREEN}Success!{RESET} Model saved to {save_path}.")
    
