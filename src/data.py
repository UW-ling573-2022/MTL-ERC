from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

def preprocess(checkpoint, data_files):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    raw_datasets = load_dataset("csv", data_files=data_files)
    speaker = ClassLabel(num_classes = 7, names = ["Chandler", "Joey", "Monica", "Rachel", "Ross", "Phoebe", "Others"])
    emotion = ClassLabel(num_classes = 7,names=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])
    
    def preprocess_func(example):
        tokenized = tokenizer(example["Utterance"])
        tokenized["speaker"] = speaker.str2int(example["Speaker"]) \
            if example["Speaker"] in speaker.names else speaker.str2int("Others")
        tokenized["emotion"] = emotion.str2int(example["Emotion"])
        return tokenized
    datasets = raw_datasets.map(preprocess_func)
    datasets = datasets.cast_column("speaker", speaker)
    datasets = datasets.cast_column("emotion", emotion)
    
    cols_to_keep = ["input_ids", "attention_mask", "speaker", "emotion"]
    cols_to_remove = [c for c in datasets["train"].column_names if c not in cols_to_keep]
    datasets = datasets.remove_columns(cols_to_remove)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(datasets["validation"], batch_size=8, collate_fn=data_collator)
    test_dataloader = DataLoader(datasets["test"], batch_size=8, collate_fn=data_collator)
    return train_dataloader, eval_dataloader, test_dataloader