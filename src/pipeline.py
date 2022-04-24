import numpy as np
import torch
from datasets import ClassLabel, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

import util
from MTL.model import MultiTaskModel
from MTL.train import MultiTaskTrainer
from preprocess import preprocess


def prepare_datasets(cx_datasets, **kwargs):
    task_datasets = {}
    for split in ["train", "validation", "test"]:
        task_datasets[split] = {}
        for cx in cx_datasets:
            if cx == "with_past" or cx == "with_future":
                for task, (ds, _) in cx_datasets[cx].items():
                    if task not in task_datasets[split]:
                        task_datasets[split][task] = ds[split]
                    else:
                        ds_to_concat = [task_datasets[split][task], ds[split]]
                        task_datasets[split][task] = concatenate_datasets(ds_to_concat)

    train_dataset = task_datasets["train"]
    eval_dataset = task_datasets["validation"]
    test_dataset = task_datasets["test"]

    eval_dataset["task"] = kwargs["evaluation"]
    test_dataset["task"] = kwargs["evaluation"]

    return train_dataset, eval_dataset, test_dataset


def pipeline(**kwargs):
    data_files = {"train": "/train_sent_emo.csv",
                  "validation": "/dev_sent_emo.csv",
                  "test": "/test_sent_emo.csv"}
    labels = {
        "Speaker": ClassLabel(
            num_classes=7,
            names=["Chandler", "Joey", "Monica", "Rachel", "Ross", "Phoebe", "Others"]),
        "Emotion": ClassLabel(
            num_classes=7,
            names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]),
        "Sentiment": ClassLabel(
            num_classes=3,
            names=["positive", "neutral", "negative"])
    }
    checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    cx_datasets = preprocess(tokenizer, labels, **kwargs)
    train_dataset, eval_dataset, test_dataset = prepare_datasets(
        cx_datasets, **kwargs)

    task_models = {
        task: AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=label.num_classes)
        for task, (_, label) in cx_datasets["no_context"].items()
    }
    multi_task_model = MultiTaskModel.from_task_models(task_models)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    if not kwargs["do_train"]:
        multi_task_model.load_state_dict(torch.load(kwargs["model_file"], map_location=torch.device(device)))

    multi_task_model.to(device)

    def compute_metrics(eval_preds):
        metric = load_metric("f1")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")

    training_args = TrainingArguments(
        output_dir=kwargs["train_dir"],
        seed=kwargs["seed"],
        overwrite_output_dir=True,
        label_names=["labels"],
        learning_rate=kwargs["learning_rate"],
        num_train_epochs=kwargs["epoch"],
        per_device_train_batch_size=kwargs["batch_size"],
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        load_best_model_at_end=True
    )

    trainer = MultiTaskTrainer(
        multi_task_model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if kwargs["do_train"]:
        trainer.train()

    f1 = trainer.predict(test_dataset).metrics['test_f1']
    print("Weighted F1:", f1)


if __name__ == "__main__":
    kwargs = vars(util.get_args())
    pipeline(**kwargs)
