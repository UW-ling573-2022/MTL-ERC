import torch
from tqdm.auto import tqdm


def criterion(outputs, labels, loss_fn):
    losses = 0
    for key in outputs:
        losses += loss_fn(outputs[key], labels[key])
    return losses


def train(model, 
          train_dataloader, 
          eval_dataloader, 
          epochs, 
          loss_fn, 
          optimizer, 
          scheduler):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    num_training_steps = epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(1, epochs + 1):
        train_speaker_correct = 0
        train_emotion_correct = 0
        eval_speaker_correct = 0
        eval_emotion_correct = 0
        
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, speaker_ids, emotion_ids = batch.values()
            labels = {"speaker": speaker_ids, "emotion": emotion_ids}
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels, loss_fn)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            train_speaker_correct += torch.sum(torch.argmax(outputs["speaker"], dim=1) == labels["speaker"]).item()
            train_emotion_correct += torch.sum(torch.argmax(outputs["emotion"], dim=1) == labels["emotion"]).item()
        
        train_speaker_acc = train_speaker_correct / len(train_dataloader.dataset)
        train_emotion_acc = train_emotion_correct / len(train_dataloader.dataset)
        
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, speaker_ids, emotion_ids = batch.values()
            labels = {"speaker": speaker_ids, "emotion": emotion_ids}
            outputs = model(input_ids, attention_mask)
            
            eval_speaker_correct += torch.sum(torch.argmax(outputs["speaker"], dim=1) == labels["speaker"]).item()
            eval_emotion_correct += torch.sum(torch.argmax(outputs["emotion"], dim=1) == labels["emotion"]).item()
            
        eval_speaker_acc = eval_speaker_correct / len(eval_dataloader.dataset)
        eval_emotion_acc = eval_emotion_correct / len(eval_dataloader.dataset)
        
        progress_bar.set_description(f"Epoch {epoch} | Train Speaker Acc: {train_speaker_acc:.3f} | Train Emotion Acc: {train_emotion_acc:.3f} | Eval Speaker Acc: {eval_speaker_acc:.3f} | Eval Emotion Acc: {eval_emotion_acc:.3f}")
    