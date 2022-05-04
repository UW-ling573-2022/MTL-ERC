from datasets import load_dataset


def preprocess(tokenizer, labels, **kwargs):

    meld_files = {
        "train": kwargs["data_dir"] + "MELD/train_sent_emo.csv", 
        "validation": kwargs["data_dir"] + "MELD/dev_sent_emo.csv",
        "test": kwargs["data_dir"] + "MELD/test_sent_emo.csv"
    }
    
    emorynlp_files = {
        "train": kwargs["data_dir"] + "EMORYNLP/train.csv",
        "validation": kwargs["data_dir"] + "EMORYNLP/dev.csv",
        "test": kwargs["data_dir"] + "EMORYNLP/test.csv"
    }
    
    datasets = {"MELD": load_dataset("csv", data_files=meld_files),
                "EmoryNLP": load_dataset("csv", data_files=emorynlp_files)}
    
    def encode_label(example, labels):
        for task, label in labels.items():
            if task == "Speaker":
                example[task] = label.str2int(example[task]) \
                    if example[task] in label.names else label.str2int("Others")
            else:
                example[task] = label.str2int(example[task])
        return example

    for name, dataset in datasets.items():
        datasets[name] = dataset.map(lambda e: encode_label(e, labels[name]))

    def add_context(example, idx, dataset, labels):
        example["Past"] = ""
        example["Future"] = ""

        if example["Utterance_ID"] != 0:
            i = 1
            while idx - i >= 0:
                past = dataset[idx - i]
                past_speaker = labels["Speaker"].int2str(past["Speaker"])
                past_utterance = past["Utterance"]
                if kwargs["speaker_in_context"]:
                    example["Past"] = past_speaker + ":" + past_utterance + " " + example["Past"]
                else:
                    example["Past"] = past_utterance + " " + example["Past"]
                if past["Utterance_ID"] == 0 or i >= kwargs["num_past_utterances"]:
                    break
                i += 1

        if idx + 1 < len(dataset) and dataset[idx + 1]["Utterance_ID"] != 0:
            i = 1
            while idx + i < len(dataset):
                future = dataset[idx + i]
                future_speaker = labels["Speaker"].int2str(future["Speaker"])
                future_utterance = future["Utterance"]
                if kwargs["speaker_in_context"]:
                    example["Future"] += " " + future_speaker + ":" + future_utterance
                else:
                    example["Future"] += " " + future_utterance
                i += 1
                if idx + i < len(dataset) and dataset[idx + i]["Utterance_ID"] == 0 \
                    or i >= kwargs["num_future_utterances"]:
                    break

        return example

    for name, dataset in datasets.items():
        for split, ds in dataset.items():
            dataset[split] = ds.map(lambda e, i: add_context(e, i, ds, labels[name]), with_indices=True)

    def tokenize(example, add_past, add_future):
        if add_past:
            return tokenizer(example["Past"], example["Utterance"])
        elif add_future:
            return tokenizer(example["Utterance"], example["Future"])
        else:
            return tokenizer(example["Utterance"])
        
    for name, dataset in datasets.items():
        cx_datasets = {}
        cx_datasets["with_past"] = dataset.map(
            lambda e: tokenize(e, add_past=True, add_future=False), batched=True)
        cx_datasets["with_future"] = dataset.map(
            lambda e: tokenize(e, add_past=False, add_future=True), batched=True)
        cx_datasets["no_context"] = dataset.map(
            lambda e: tokenize(e, add_past=False, add_future=False), batched=True)

        tasks = list(labels[name].keys())
        for cx in cx_datasets:
            cols_to_keep = ["input_ids", "attention_mask"] + tasks
            cols_to_remove = [c for c in cx_datasets[cx]["train"].column_names if c not in cols_to_keep]
            cx_datasets[cx] = cx_datasets[cx].remove_columns(cols_to_remove)
            task_datasets = {}
            for task in tasks:
                label = labels[name][task]
                ds = cx_datasets[cx]
                ds = ds.cast_column(task, label)
                ds = ds.remove_columns([t for t in tasks if t != task])
                ds = ds.rename_column(task, "labels")
                ds.set_format()
                task_datasets[task] = (ds, label)
            cx_datasets[cx] = task_datasets
        datasets[name] = cx_datasets

    return datasets