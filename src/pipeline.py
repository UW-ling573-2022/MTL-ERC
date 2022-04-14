from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from preprocess import Preprocessor

def pipeline():
    data_files = {"train": "data/train_sent_emo.csv", 
                  "validation": "data/dev_sent_emo.csv", 
                  "test": "data/test_sent_emo.csv"}
    checkpoint = "roberta-base"
    preprocessor = Preprocessor(checkpoint, data_files)
    preprocessor.preprocess()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=7)
    training_args = TrainingArguments(
        output_dir="../results",
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch"
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=preprocessor.datasets["train"],
        eval_dataset=preprocessor.datasets["validation"],
        tokenizer=preprocessor.tokenizer,
    )
    trainer.train()
    predictions = trainer.predict(preprocessor["test"])  

if __name__ == "__main__":
   pipeline()