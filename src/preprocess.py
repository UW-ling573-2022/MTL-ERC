from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

class Preprocessor:
    def __init__(self, checkpoint, data_files):
        self.datasets = load_dataset("csv", data_files=data_files)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.labels = ClassLabel(num_classes = 7,names=["anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear"])
        
    def __encode_labels(self, label_col):
        encode_func  = lambda x: {"labels": self.labels.str2int(x[label_col])}
        self.datasets = self.datasets.map(encode_func, batched=True)
        self.datasets = self.datasets.cast_column("labels", self.labels)
    
    def __tokenize(self):
        tokenize_func = lambda x: self.tokenizer(x["Speaker"], x["Utterance"])
        self.datasets = self.datasets.map(tokenize_func, batched=True)
        
        
    def __clean_cols(self):
        cols_to_keep = ["input_ids", "attention_mask", "labels"]
        cols_to_remove = [c for c in self.datasets["train"].column_names if c not in cols_to_keep]
        self.datasets = self.datasets.remove_columns(cols_to_remove)
    
    def preprocess(self):
        self.__encode_labels("Emotion")
        self.__tokenize()
        self.__clean_cols()