from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess(tokenizer, data_files, labels):
    raw_datasets = load_dataset("csv", data_files=data_files)
    speaker = labels["speaker"]
    emotion = labels["emotion"]
    
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
    speaker_datasets = datasets.remove_columns(["emotion"])
    emotion_datasets = datasets.remove_columns(["speaker"])
    
    speaker_datasets = speaker_datasets.rename_column("speaker", "labels")
    emotion_datasets = emotion_datasets.rename_column("emotion", "labels")
    
    return {"speaker": speaker_datasets, "emotion": emotion_datasets}