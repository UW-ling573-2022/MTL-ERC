import torch
import numpy as np
from datasets import ClassLabel, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

from preprocess import preprocess
from MTL.model import MultiTaskModel
from MTL.train import MultiTaskTrainer

def prepare_datasets(cx_datasets, task_to_evaluate, task_to_test):
    task_datasets = {}
    for split in ["train", "validation", "test"]:
        task_datasets[split] = {}
        for cx in cx_datasets:
            if cx == "with_past":
                for task, (ds, _) in cx_datasets[cx].items():
                    if task not in task_datasets[split]:
                        task_datasets[split][task] = ds[split]
                    else:
                        ds_to_concat = [task_datasets[split][task], ds[split]]
                        task_datasets[split][task] = concatenate_datasets(ds_to_concat)

    train_dataset = task_datasets["train"]
    eval_dataset = task_datasets["validation"]
    test_dataset = task_datasets["test"]

    eval_dataset["Task to Evaluate"] = task_to_evaluate
    test_dataset["Task to Test"] = task_to_test
    
    return train_dataset, eval_dataset, test_dataset

def pipeline(do_train, saved_model_path=None):
    
    data_files = {"train": "/train_sent_emo.csv", 
                "validation": "/dev_sent_emo.csv", 
                "test": "/test_sent_emo.csv"}
    labels = {
        "Speaker": ClassLabel(
            num_classes = 7, 
            names = ["Chandler", "Joey", "Monica", "Rachel", "Ross", "Phoebe", "Others"]),
        "Emotion": ClassLabel(
            num_classes = 7,
            names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]),
        "Sentiment": ClassLabel(
            num_classes = 3, 
            names=["positive", "neutral", "negative"])
    }
    
    checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    cx_datasets = preprocess(tokenizer, data_files, labels)
    train_dataset, eval_dataset, test_dataset = prepare_datasets(
        cx_datasets, task_to_evaluate="Emotion", task_to_test="Emotion")

    task_models = {
        task: AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=label.num_classes)
        for task, (_, label) in cx_datasets["no_context"].items()
    }
    multi_task_model = MultiTaskModel.from_task_models(task_models)

    if saved_model_path is not None:
        multi_task_model.load_state_dict(torch.load(saved_model_path))
    
    def compute_metrics(eval_preds):
        metric = load_metric("f1")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")
    
    training_args = TrainingArguments(
        output_dir="../outputs/multi_task_model",
        overwrite_output_dir=True,
        label_names=["labels"],
        learning_rate=1e-5,
        num_train_epochs=10,
        per_device_train_batch_size=32,  
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
    
    if do_train:
        trainer.train()
        
    f1 = trainer.predict(test_dataset).metrics['test_f1']
    print("Weighted F1:", f1)

if __name__ == "__main__":
   pipeline(do_train=False, saved_model_path="../outputs/multi_task_model/best_model.pth")