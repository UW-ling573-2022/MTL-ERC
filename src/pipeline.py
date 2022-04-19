import torch
from data import preprocess
from model import ERCModel
from train import train

def pipeline():
    data_files = {"train": "data/train_sent_emo.csv", 
                  "validation": "data/dev_sent_emo.csv", 
                  "test": "data/test_sent_emo.csv"}
    checkpoint = "roberta-base"
    train_dataloader, eval_dataloader, test_dataloader = preprocess(checkpoint, data_files)
    model = ERCModel(checkpoint, 7, 7)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, epochs=5, steps_per_epoch=len(train_dataloader))
    train(model, 
          train_dataloader, 
          eval_dataloader,
          epochs=5,
          loss_fn=loss_fn,
          optimizer=optimizer,
          scheduler=scheduler,
          device="cuda")

if __name__ == "__main__":
   pipeline()