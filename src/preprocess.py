from datasets import load_dataset


def preprocess(tokenizer, labels, **kwargs):

    data_files = {"train": kwargs["data_dir"] + "MELD/train_sent_emo.csv",
                  "validation": kwargs["data_dir"] + "MELD/dev_sent_emo.csv",
                  "test": kwargs["data_dir"] + "MELD/test_sent_emo.csv"}
    raw_datasets = load_dataset("csv", data_files=data_files)

    def encode_label(example):
        for task, label in labels.items():
            if task == "Speaker":
                example[task] = label.str2int(example[task]) \
                    if example[task] in label.names else label.str2int("Others")
            else:
                example[task] = label.str2int(example[task])
        return example

    raw_datasets = raw_datasets.map(encode_label)

    def add_context(example, idx, dataset):
        example["Past"] = ""
        example["Future"] = ""

        if example["Utterance_ID"] != 0:
            i = 1
            while idx - i >= 0:
                past = dataset[idx - i]
                past_speaker = labels["Speaker"].int2str(past["Speaker"])
                past_utterance = past["Utterance"]
                example["Past"] += past_speaker + ":" + past_utterance
                if past["Utterance_ID"] == 0 or i == kwargs["num_past_utterances"]:
                    break
                i += 1

        if idx + 1 < len(dataset) and dataset[idx + 1]["Utterance_ID"] != 0:
            i = 1
            while idx + i < len(dataset):
                future = dataset[idx + i]
                future_speaker = labels["Speaker"].int2str(future["Speaker"])
                future_utterance = future["Utterance"]
                example["Future"] += future_speaker + ":" + future_utterance
                i += 1
                if future["Utterance_ID"] == 0 or i == kwargs["num_future_utterances"] - 1:
                    break

        return example

    for split, dataset in raw_datasets.items():
        raw_datasets[split] = dataset.map(lambda e, i: add_context(e, i, dataset), with_indices=True)

    def tokenize(example, add_past, add_future):
        if add_past:
            return tokenizer(example["Past"], example["Utterance"])
        elif add_future:
            return tokenizer(example["Utterance"], example["Future"])
        else:
            return tokenizer(example["Utterance"])

    cx_datasets = {}
    cx_datasets["with_past"] = raw_datasets.map(
        lambda e: tokenize(e, add_past=True, add_future=False), batched=True)
    cx_datasets["with_future"] = raw_datasets.map(
        lambda e: tokenize(e, add_past=False, add_future=True), batched=True)
    cx_datasets["no_context"] = raw_datasets.map(
        lambda e: tokenize(e, add_past=False, add_future=False), batched=True)

    tasks = list(labels.keys())
    for cx in cx_datasets:
        cols_to_keep = ["input_ids", "attention_mask"] + tasks
        cols_to_remove = [c for c in cx_datasets[cx]["train"].column_names if c not in cols_to_keep]
        cx_datasets[cx] = cx_datasets[cx].remove_columns(cols_to_remove)
        task_datasets = {}
        for task in tasks:
            label = labels[task]
            ds = cx_datasets[cx]
            ds = ds.cast_column(task, label)
            ds = ds.remove_columns([t for t in tasks if t != task])
            ds = ds.rename_column(task, "labels")
            ds.set_format()
            task_datasets[task] = (ds, label)
        cx_datasets[cx] = task_datasets

    return cx_datasets
