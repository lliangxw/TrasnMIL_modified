import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer


# import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Get the number of GPUs
gpu_count = torch.cuda.device_count()
print(f"Number of GPUs: {gpu_count}")

# List GPU details
for i in range(gpu_count):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='MAIA_CCA_vs_TM/TransMIL.yaml',type=str)
    parser.add_argument('--gpus', default = [2])
    parser.add_argument('--fold', default = 0)
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    trainer = Trainer(
        # amp_backend='apex',
        accelerator='gpu',
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        devices='auto',
        # gpus=cfg.General.gpus,
        # amp_level=cfg.General.amp_level,  
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
        # gpus=1
    )

    #---->train or test
    if cfg.General.server == 'train':
        # new_model = model.load_from_checkpoint(checkpoint_path='/home/local-admin/Documents/projects/TransMIL/logs/MAIA_ALL/TransMIL/fold0/last.ckpt', cfg=cfg) #### Added
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    #---->main
    main(cfg)
 
