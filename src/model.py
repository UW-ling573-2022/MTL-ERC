import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ERCModel(nn.Module):
    def __init__(self, checkpoint, num_speakers, num_emotions):
        super(ERCModel, self).__init__()
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.base_model = AutoModel.from_pretrained(checkpoint)
        self.speaker = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, num_speakers),
        )
        self.emotion = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, num_emotions),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        speaker_logits = self.speaker(outputs.last_hidden_state[:, 0, :])
        emotion_logits = self.emotion(outputs.last_hidden_state[:, 0, :])
        return {"speaker": speaker_logits, "emotion": emotion_logits}