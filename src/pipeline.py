import numpy as np
import torch
from datasets import ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

import util
from MTL.model import MultiTaskModel
from MTL.train import MultitaskTrainer
from load_data import TextDataset


def pipeline(**args):
    train_dataset = {
        "emotion": TextDataset(
            dataset="MELD",
            split="train",
            field="emotion",
            seed=args["seed"],
            directory=args["data_dir"],
            num_past_utterances=args["num_past_utterances"],
            num_future_utterances=args["num_future_utterances"],
        ),
        "speaker": TextDataset(
            dataset="MELD",
            split="train",
            field="speaker",
            seed=args["seed"],
            directory=args["data_dir"],
            num_past_utterances=args["num_past_utterances"],
            num_future_utterances=args["num_future_utterances"],
        )
    }
    eval_dataset = {"emotion": TextDataset(
        dataset="MELD",
        split="dev",
        field="emotion",
        seed=args["seed"],
        directory=args["data_dir"],
        num_past_utterances=args["num_past_utterances"],
        num_future_utterances=args["num_future_utterances"],
    ), "speaker": TextDataset(
        dataset="MELD",
        split="dev",
        field="speaker",
        seed=args["seed"],
        directory=args["data_dir"],
        num_past_utterances=args["num_past_utterances"],
        num_future_utterances=args["num_future_utterances"],
    ), "task": args["evaluation"]}
    test_dataset = {
        "emotion": TextDataset(
            dataset="MELD",
            split="test",
            field="emotion",
            seed=args["seed"],
            directory=args["data_dir"],
            num_past_utterances=args["num_past_utterances"],
            num_future_utterances=args["num_future_utterances"],
        ),
        "speaker": TextDataset(
            dataset="MELD",
            split="test",
            field="speaker",
            seed=args["seed"],
            directory=args["data_dir"],
            num_past_utterances=args["num_past_utterances"],
            num_future_utterances=args["num_future_utterances"],
        ), "task": args["evaluation"]
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    if not args["do_train"]:
        multi_task_model.load_state_dict(torch.load(args["model_file"], map_location=torch.device(device)))

    multi_task_model.to(device)

    def compute_metrics(eval_preds):
        metric = load_metric("f1")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")

    training_args = TrainingArguments(
        output_dir=args["train_dir"],
        seed=args["seed"],
        overwrite_output_dir=True,
        label_names=["labels"],
        learning_rate=args["learning_rate"],
        num_train_epochs=args["epoch"],
        per_device_train_batch_size=args["batch_size"],
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

    if args["do_train"]:
        trainer.train()

    f1 = trainer.predict(test_dataset).metrics['test_f1']
    print("Weighted F1:", f1)


if __name__ == "__main__":
    args = vars(util.get_args())
    pipeline(**args)
