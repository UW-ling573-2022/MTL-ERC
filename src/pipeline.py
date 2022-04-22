import numpy as np
import torch
from datasets import ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

from MTL.model import MultiTaskModel
from MTL.train import MultitaskTrainer
from load_data import TextDataset


def pipeline(do_train, saved_model_path=None):
    train_dataset = {
        "emotion": TextDataset(
            dataset="MELD",
            split="train",
            field="emotion",
            speaker_mode=True,
            num_past_utterances=1,
            num_future_utterances=1,
        ),
        "speaker": TextDataset(
            dataset="MELD",
            split="train",
            field="speaker",
            speaker_mode=True,
            num_past_utterances=1,
            num_future_utterances=1,
        )
    }
    eval_dataset = {
        "emotion": TextDataset(
            dataset="MELD",
            split="dev",
            field="emotion",
            speaker_mode=True,
            num_past_utterances=1,
            num_future_utterances=1,
        ),
        "speaker": TextDataset(
            dataset="MELD",
            split="dev",
            field="speaker",
            speaker_mode=True,
            num_past_utterances=1,
            num_future_utterances=1,
        )
    }

    checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    labels = {
        "speaker": ClassLabel(num_classes=7,
                              names=["Chandler", "Joey", "Monica", "Rachel", "Ross", "Phoebe", "Others"]),
        "emotion": ClassLabel(num_classes=7,
                              names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])
    }

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

    test_dataset = {
        "emotion": TextDataset(
            dataset="MELD",
            split="test",
            field="emotion",
            speaker_mode=True,
            num_past_utterances=1,
            num_future_utterances=1,
        ),
        "speaker": TextDataset(
            dataset="MELD",
            split="test",
            field="speaker",
            speaker_mode=True,
            num_past_utterances=1,
            num_future_utterances=1,
        )
    }

    f1 = trainer.predict(test_dataset).metrics['test_f1']
    print("Weighted F1:", f1)


if __name__ == "__main__":
    pipeline(True)
