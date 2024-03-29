import numpy as np
import torch
import json
from datasets import ClassLabel, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import os

import util
from MTL.model import MultiTaskModel
from MTL.train import MultiTaskTrainer
from preprocess import preprocess


def prepare_datasets(datasets, **kwargs):
    for dataset_name, cx_datasets in datasets.items():
        task_dataset = {}
        for split in ["train", "validation", "test"]:
            task_dataset[split] = {}
            for cx in cx_datasets:
                if cx == "with_past" and kwargs["num_past_utterances"] == 0:
                    continue
                elif cx == "with_future" and kwargs["num_future_utterances"] == 0:
                    continue
                elif cx == "no_context" and kwargs["num_past_utterances"] + kwargs["num_future_utterances"] > 0:
                    continue
                else:
                    for task, (ds, _) in cx_datasets[cx].items():
                        if split == "train" and task not in kwargs["train_task"]:
                            continue
                        if task not in task_dataset[split]:
                            task_dataset[split][task] = ds[split]
                        else:
                            ds_to_concat = [task_dataset[split][task], ds[split]]
                            task_dataset[split][task] = concatenate_datasets(ds_to_concat)

        train_dataset = task_dataset["train"]
        eval_dataset = task_dataset["validation"]
        test_dataset = task_dataset["test"]

        datasets[dataset_name] = {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}

    train_dataset = {dataset_name + "_" + task: datasets[dataset_name]["train"][task]
                     for dataset_name in datasets if dataset_name in kwargs["train_dataset"]
                     for task in datasets[dataset_name]["train"] if task in kwargs["train_task"]}

    eval_dataset_task = kwargs["eval_dataset"] + "_" + kwargs["eval_task"]
    eval_dataset = {eval_dataset_task: datasets[kwargs["eval_dataset"]]["validation"][kwargs["eval_task"]]}
    eval_dataset["task"] = eval_dataset_task
    test_dataset = {eval_dataset_task: datasets[kwargs["eval_dataset"]]["test"][kwargs["eval_task"]]}
    test_dataset["task"] = eval_dataset_task

    return train_dataset, eval_dataset, test_dataset


def evaluation(dataset_name, dir_name, trainer, dataset_labels, tokenizer):
    pred = trainer.predict(dataset_name)
    f1 = pred.metrics['test_f1']
    print("Weighted F1:", f1)

    score_filename = kwargs["result_dir"] + dir_name + "D4_scores.out"
    os.makedirs(os.path.dirname(score_filename), exist_ok=True)

    f = open(score_filename, "a+")
    f.write(json.dumps(kwargs))
    f.write("\nWeighted F1: {}\n".format(f1))
    f.close()

    pred_labels = dataset_labels[kwargs["eval_dataset"]][kwargs["eval_task"]].int2str(pred.predictions.argmax(axis=-1))
    true_labels = dataset_labels[kwargs["eval_dataset"]][kwargs["eval_task"]].int2str(pred.label_ids)

    prediction_filename = kwargs["output_dir"] + dir_name + "predictions.out"
    os.makedirs(os.path.dirname(prediction_filename), exist_ok=True)

    inputs = tokenizer.batch_decode(dataset_name[kwargs["eval_dataset"] + "_" + kwargs["eval_task"]]["input_ids"])
    f = open(prediction_filename, "w")
    f.write("Input\tPredicted\tTrue\n")
    f.write("\n".join(["\t".join([input, pred_label, true_label])
                       for input, pred_label, true_label
                       in zip(inputs, pred_labels, true_labels)]))
    f.close()


def pipeline(**kwargs):
    dataset_labels = {
        "MELD":
            {
                "Speaker": ClassLabel(
                    num_classes=7,
                    names=["Chandler", "Joey", "Monica", "Rachel", "Ross", "Phoebe", "Others"]),
                "Emotion": ClassLabel(
                    num_classes=7,
                    names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]),
                "Sentiment": ClassLabel(
                    num_classes=3,
                    names=["positive", "neutral", "negative"])
            },
        "EmoryNLP":
            {
                "Speaker": ClassLabel(
                    num_classes=7,
                    names=["Chandler", "Joey", "Monica", "Rachel", "Ross", "Phoebe", "Others"]),
                "Emotion": ClassLabel(
                    num_classes=7,
                    names=["Sad", "Mad", "Scared", "Powerful", "Peaceful", "Joyful", "Neutral"])
            },
        "MPDD":
            {
                "Emotion": ClassLabel(
                    num_classes=7,
                    names=["angry", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])
            },
    }

    tokenizer = AutoTokenizer.from_pretrained(kwargs["checkpoint"], max_length=1024, padding="max_length",
                                              truncation=True)
    datasets = preprocess(tokenizer, dataset_labels, **kwargs)
    train_dataset, eval_dataset, test_dataset = prepare_datasets(datasets, **kwargs)

    tasks = {task: dataset_labels[task.split("_")[0]][task.split("_")[1]] for task in train_dataset.keys()}
    task_models = {
        task: AutoModelForSequenceClassification.from_pretrained(
            kwargs["checkpoint"],
            num_labels=label.num_classes,
            max_length=1024)
        for task, label in tasks.items()
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1"
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

    evaluation(eval_dataset, "/devtest/", trainer, dataset_labels, tokenizer)
    evaluation(test_dataset, "/evaltest/", trainer, dataset_labels, tokenizer)


if __name__ == "__main__":
    kwargs = vars(util.get_args())
    pipeline(**kwargs)
