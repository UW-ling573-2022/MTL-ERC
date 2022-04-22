import torch
import numpy as np
from datasets import ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

from preprocess import preprocess
from MTL.model import MultiTaskModel
from MTL.train import MultitaskTrainer

def pipeline(do_train, saved_model_path=None):
    data_files = {"train": "data/train_sent_emo.csv", 
                  "validation": "data/dev_sent_emo.csv", 
                  "test": "data/test_sent_emo.csv"}
    checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    labels = {
        "speaker": ClassLabel(num_classes = 7, names = ["Chandler", "Joey", "Monica", "Rachel", "Ross", "Phoebe", "Others"]),
        "emotion": ClassLabel(num_classes = 7,names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])
    }
    
    multi_task_datasets = preprocess(tokenizer, data_files, labels)
    train_dataset = {
        task: datasets["train"]
        for task, datasets in multi_task_datasets.items()
    }
    eval_dataset = {
        task: datasets["validation"]
        for task, datasets in multi_task_datasets.items()
    } 
    test_dataset = {
         task: datasets["test"]
        for task, datasets in multi_task_datasets.items()
    }
    eval_dataset["Task to Evaluate"] = "emotion"
    test_dataset["Task to Test"] = "emotion"
    
    single_task_models = {
        task: AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=label.num_classes)
        for task, label in labels.items()
    }
    multi_task_model = MultiTaskModel.from_single_task_models(single_task_models)
    
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
        per_device_train_batch_size=8,  
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        load_best_model_at_end=True
    )
    
    trainer = MultitaskTrainer(
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
   pipeline()