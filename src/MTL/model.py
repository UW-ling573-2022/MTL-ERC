import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class MultiTaskModel(PreTrainedModel):
    def __init__(self, encoder, single_task_models):
        super(MultiTaskModel, self).__init__(PretrainedConfig())
        self.encoder = encoder
        self.single_task_models = nn.ModuleDict(single_task_models)
        
    @classmethod
    def from_single_task_models(cls, single_task_models):
        shared_encoder = None
        for model in single_task_models.values():
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
        return cls(shared_encoder, single_task_models)
                  
    @staticmethod
    def get_encoder_attr_name(model):
        model_name = model.__class__.__name__
        if model_name.startswith('Bert'):
            return 'bert'
        elif model_name.startswith('Roberta'):
            return 'roberta'
        elif model_name.startswith('Albert'):
            return 'albert'
        else:
            raise ValueError('Unsupported model: {}'.format(model_name))
        
    def forward(self, task, input_ids, attention_mask, **kwargs):
        model = self.single_task_models[task]
        return model(input_ids, attention_mask, **kwargs)