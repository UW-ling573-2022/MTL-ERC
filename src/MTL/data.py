import numpy as np
from torch.utils.data import DataLoader

class SingleTaskDataLoader:
    def __init__(self, task, **kwargs):
        self.task = task
        self.data_loader = DataLoader(**kwargs)
        self.batch_size = self.data_loader.batch_size
        self.dataset = self.data_loader.dataset
        
    def __len__(self) -> int:
        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            batch["task"] = self.task
            yield batch
    
class MultiTaskDataLoader:
    def __init__(self, task_data_loaders):
        self.task_data_loaders = task_data_loaders
        self.dataset = [None] * sum([len(dl.dataset) for dl in task_data_loaders.values()])
        
    def __len__(self) -> int:
        return sum([len(dl) for dl in self.task_data_loaders.values()])
    
    def __iter__(self):
        task_choices = []
        for task, dl in self.task_data_loaders.items():
            task_choices.extend([task] * len(dl))
        task_choices = np.array(task_choices)
        np.random.shuffle(task_choices)
        for task in task_choices:
            yield next(iter(self.task_data_loaders[task]))