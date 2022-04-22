import torch
from load_data import TextDataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from model import ERCModel
from train import train


def pipeline():
    checkpoint = "roberta-base"
    ds_train = TextDataset(
        dataset="MELD",
        split="train",
        speaker_mode=True,
        num_past_utterances=1,
        num_future_utterances=1,
    )
    ds_eval = TextDataset(
        dataset="MELD",
        split="dev",
        speaker_mode=True,
        num_past_utterances=1,
        num_future_utterances=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(ds_train, shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(ds_eval, batch_size=8, collate_fn=data_collator)
    model = ERCModel(checkpoint, 7, 7)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, epochs=5, steps_per_epoch=len(ds_train))
    train(model,
          train_dataloader,
          eval_dataloader,
          epochs=5,
          loss_fn=loss_fn,
          optimizer=optimizer,
          scheduler=scheduler,
          )


if __name__ == "__main__":
    pipeline()
