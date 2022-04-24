import torch
import datasets
from transformers import Trainer
from transformers import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard
from MTL.data import SingleTaskDataLoader, MultiTaskDataLoader

class MultiTaskTrainer(Trainer):

    def get_single_task_dataloader(self, task, dataset, description):
        if description == "training" and self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        elif description == "evaluation" and dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)

        if isinstance(dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                dataset = IterableDatasetShard(
                    dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return SingleTaskDataLoader(
                task,
                dataset=dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        if description == "training":
            self.train_dataset, dataset = dataset, self.train_dataset
            sampler = self._get_train_sampler()
            self.train_dataset, dataset = dataset, self.train_dataset
            batch_size = self.args.train_batch_size
        else:
            sampler = self._get_eval_sampler(dataset)
            batch_size = self.args.eval_batch_size
            
        return SingleTaskDataLoader(
            task,
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def get_train_dataloader(self):
        return MultiTaskDataLoader({
            task: self.get_single_task_dataloader(task, dataset, description="training")
            for task, dataset in self.train_dataset.items()
        })
        
    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        task_to_eval = eval_dataset["task"]
        return self.get_single_task_dataloader(task_to_eval, eval_dataset[task_to_eval], description="evaluation")
        
    
    def get_test_dataloader(self, test_dataset):
        task_to_test = test_dataset["task"]
        return self.get_single_task_dataloader(task_to_test, test_dataset[task_to_test], description="test")