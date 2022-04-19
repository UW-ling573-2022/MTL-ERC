import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ERCModel(nn.Module):
    def __init__(self, checkpoint, num_speakers, num_emotions):
        super(ERCModel, self).__init__()
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.base_model = AutoModel.from_pretrained(checkpoint)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.hidden_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
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
        hidden_layer_output = self.hidden_layer(outputs.last_hidden_state)[:, 0, :]
        speaker_logits = self.speaker(hidden_layer_output)
        emotion_logits = self.emotion(hidden_layer_output)
        return {"speaker": speaker_logits, "emotion": emotion_logits}