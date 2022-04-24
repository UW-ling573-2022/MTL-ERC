"""
MIT License

Copyright (c) 2020 CLTL Leolani
Copyright (c) 2021 Junyin Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import util

MELD_SPEAKER = ["Chandler", "Joey", "Monica", "Rachel", "Ross", "Phoebe"]


def set_seed(seed: int) -> None:
    """Set random seed to a fixed value.

    Set everything to be deterministic
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_emotion2id(dataset: str) -> Tuple[dict, dict]:
    """Get a dict that converts string class to number."""

    if dataset == "MELD":
        # MELD has 7 classes
        emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
        id2emotion = {val: key for key, val in emotion2id.items()}
    return emotion2id, id2emotion


def get_speaker2id(dataset: str) -> Tuple[dict, dict]:
    """Get a dict that converts string class to number."""

    if dataset == "MELD":
        # MELD has 7 classes
        speakers = [
            "CHANDLER",
            "JOEY",
            "MONICA",
            "RACHEL",
            "ROSS",
            "PHOEBE",
            "OTHERS",
        ]
        speaker2id = {speaker: idx for idx, speaker in enumerate(speakers)}
        id2speaker = {val: key for key, val in speaker2id.items()}
    return speaker2id, id2speaker


class TextDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset="MELD",
            split="train",
            field="emotion",
            num_past_utterances=0,
            num_future_utterances=0,
            model_checkpoint="roberta-base",
            directory=util.get_root_dir() + "data/",
            up_to=False,
            seed=0
    ):
        """Initialize emotion recognition in conversation text modality dataset class."""

        self.dataset = dataset
        self.directory = directory
        self.split = split
        self.field = field
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        self.model_checkpoint = model_checkpoint
        self.emotion2id, self.id2emotion = get_emotion2id(self.dataset)
        self.speaker2id, self.id2speaker = get_speaker2id(self.dataset)
        self.up_to = up_to
        self.seed = seed

        self._load_from_raw()
        self._string2tokens()

    def _load_from_raw(self):
        if self.dataset in ["MELD"]:
            # Get the full file path
            raw_path = os.path.join(self.directory, self.dataset, self.split + "_sent_emo.csv")
            # Load csv to pandas Dataframe
            raw_data = pd.read_csv(raw_path)
            # Load each field
            self.dialog_group = {}
            self.emotion = {}
            self.speaker_emotion = {}
            for _, row in raw_data.iterrows():
                utterance_id = '{}_{}_{}_{}'.format(row['Dialogue_ID'],
                                                    row['Utterance_ID'],
                                                    row['Season'],
                                                    row['Episode'])
                dialog_id = '{}_{}_{}'.format(row['Dialogue_ID'], row['Season'], row['Episode'])
                if dialog_id not in self.dialog_group:
                    self.dialog_group[dialog_id] = []
                self.dialog_group[dialog_id].append(utterance_id)
                self.emotion[utterance_id] = row['Emotion']
                utterance = row["Utterance"]
                speaker = "OTHERS"
                if row["Speaker"] in MELD_SPEAKER:
                    speaker = row["Speaker"].upper()

                self.speaker_emotion[utterance_id] = {"utterance": speaker + ": " + utterance,
                                                      "clean": utterance,
                                                      "emotion": row['Emotion'],
                                                      "speaker": speaker,
                                                      }
        else:
            raise ValueError(f"{self.dataset} is not MELD")


    def _create_input(
            self, diaids, num_past_utterances, num_future_utterances
    ):
        """Create an input which will be an input to RoBERTa."""

        args = {
            "diaids": diaids,
            "num_past_utterances": num_past_utterances,
            "num_future_utterances": num_future_utterances,
        }

        logging.debug(f"arguments given: {args}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        num_truncated = 0

        inputs = []
        for diaid in tqdm(diaids):
            ues = [
                self.speaker_emotion[uttid]
                for uttid in self.dialog_group[diaid]
            ]

            num_tokens = [len(tokenizer(ue["utterance"])["input_ids"]) for ue in ues]

            for idx, ue in enumerate(ues):
                if ue["emotion"] not in list(self.emotion2id.keys()):
                    continue

                emotion = self.emotion2id[ue["emotion"]]
                speaker = self.speaker2id[ue["speaker"]]

                indexes = [idx]
                indexes_past = [
                    i for i in range(idx - 1, idx - num_past_utterances - 1, -1)
                ]
                indexes_future = [
                    i for i in range(idx + 1, idx + num_future_utterances + 1, 1)
                ]

                offset = 0
                if len(indexes_past) < len(indexes_future):
                    for _ in range(len(indexes_future) - len(indexes_past)):
                        indexes_past.append(None)
                elif len(indexes_past) > len(indexes_future):
                    for _ in range(len(indexes_past) - len(indexes_future)):
                        indexes_future.append(None)

                for i, j in zip(indexes_past, indexes_future):
                    if i is not None and i >= 0:
                        indexes.insert(0, i)
                        offset += 1
                        if (
                                sum([num_tokens[idx_] for idx_ in indexes])
                                > max_model_input_size
                        ):
                            del indexes[0]
                            offset -= 1
                            num_truncated += 1
                            break
                    if j is not None and j < len(ues):
                        indexes.append(j)
                        if (
                                sum([num_tokens[idx_] for idx_ in indexes])
                                > max_model_input_size
                        ):
                            del indexes[-1]
                            num_truncated += 1
                            break

                utterances = [ues[idx_]["utterance"] for idx_ in indexes]
                clean = [ues[idx_]["clean"] for idx_ in indexes]

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(utterances) == 1
                    final_utterance = clean[0]

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + clean[-1]
                    else:
                        final_utterance = (
                                " ".join(utterances[:-1]) + "</s></s>" + clean[-1]
                        )

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = clean[0] + "</s></s>"
                    else:
                        final_utterance = (
                                clean[0] + "</s></s>" + " ".join(utterances[1:])
                        )

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + clean[0] + "</s></s>"
                    else:
                        final_utterance = (
                                " ".join(utterances[:offset])
                                + "</s></s>"
                                + clean[offset]
                                + "</s></s>"
                                + " ".join(utterances[offset + 1:])
                        )
                else:
                    raise ValueError

                input_ids_attention_mask = tokenizer(final_utterance)
                input_ids = input_ids_attention_mask["input_ids"]
                attention_mask = input_ids_attention_mask["attention_mask"]

                label = emotion
                if self.field == "speaker":
                    label = speaker

                input_ = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": label,
                }

                inputs.append(input_)

        logging.info(f"number of truncated utterances: {num_truncated}")
        return inputs

    def _string2tokens(self):
        """Convert string to (BPE) tokens."""
        logging.info(f"converting utterances into tokens ...")

        diaids = sorted(list(self.dialog_group.keys()))

        set_seed(self.seed)
        random.shuffle(diaids)

        if self.up_to:
            logging.info(f"Using only the first {self.up_to} dialogues ...")
            diaids = diaids[: self.up_to]

        logging.info(f"creating input utterance data ... ")
        self.inputs_ = self._create_input(
            diaids=diaids,
            num_past_utterances=self.num_past_utterances,
            num_future_utterances=self.num_future_utterances,
        )

    def __getitem__(self, index):
        return self.inputs_[index]

    def __len__(self):
        return len(self.inputs_)


if __name__ == "__main__":
    ds_train = TextDataset(
        dataset="MELD",
        split="dev",
        num_past_utterances=1,
        num_future_utterances=1,
    )
