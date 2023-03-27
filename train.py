import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from msmtu.datamodules import LULCDataModule, LULCDataModuleOptim
from msmtu.pl_modules import LitBRITS

import wandb
import os

from pprint import pprint

from msmtu.utils import save_predictions

# ------------
# Set up default hyperparameters
# ------------
default_config = {
    'model': {
        'rnn_hid_size': 100,
        'impute_weight': 0.25,
        'label_weight': 0.75,
        'data_dim': 7,
        'seq_len': 12,
        'lr': 3e-3,
        'wd': 0,
        'optimizer': 'adam',
        'scheduler': 'cosine'
    },
    'data': {
        'data_dir': '/mnt/homeGPU/jrodriguez/lifewatch/data/modis_500_3/',#'/mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/soft_data/Training/training_3_05_N1',
        # 'val_dir': '/mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/soft_data/SN/sn_3_05_N1',
        'batch_size': 2048,
        # 'train_val_test_split': (1.0, 0.0, 0.0),
        'data_dim': 7,
        'test_imp': False
    },
    'trainer': {
        'max_epochs': 200,
        'fast_dev_run': 2,
        # 'overfit_batches': 2,
        # 'detect_anomaly': True,
        'accumulate_grad_batches': 1,
        # 'track_grad_norm': 2,
        # 'val_check_interval': 0.25,
        'gradient_clip_val': 0.95
    }
}


def train(project: str, run_name: str, config: dict):
    print('Project:')
    print(project)

    print('run_name:')
    print(run_name)

    print('Run config:')
    print(default_config)

    # ------------
    # Create Data Module
    # ------------

    data_module = LULCDataModuleOptim(**config['data'])
    _, idx_to_class = data_module.get_dicts()
    print(idx_to_class)

    # ------------
    # Create Model
    # ------------
    ckpt_path = None  # "/mnt/homeGPU/jrodriguez/lifewatch/BRITS_lightning/models/dutiful-forest-5/b61e7s8p/checkpoints/epoch_49_step_89500.ckpt"

    # idx_to_class variable needs to be computed at runtime
    model = LitBRITS(**config['model'], idx_to_class=idx_to_class)

    """if ckpt_path is not None:
        print(model.optimizers())
        model.optimizers()[0].param_groups[0]['capturable'] = True"""

    # ------------
    # Log Metrics using Wandb
    # ------------

    if config['trainer']['fast_dev_run']:
        logger = False
    else:
        logger = WandbLogger(project=project, name=run_name, save_dir='./logger', log_model=True)
        logger.watch(model, log="gradients", log_freq=200)  # log gradients
        print(f'wandb.run.dir: {wandb.run.dir}')

    # ------------
    # Training
    # ------------
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/rmse_epoch",
        mode="min",
        dirpath=None,
        filename="brits-epoch={epoch}-val_rmse_epoch={val/rmse_epoch:.3f}",
        auto_insert_metric_name=False,
        save_last=True,  # Save the last checkpoint in case we want to continue training later
    )

    # Create trainer
    trainer = pl.Trainer(logger=logger, **config['trainer'], accelerator='auto', devices=[1],
                         callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)  # Train the model

    # ------------
    # Test model if test set is provided
    # ------------
    if 'test_labels_path' in list(config['data'].keys()) and config['data']['test_labels_path'] is not None:
        print('TESTING')
        trainer.test(model, datamodule=data_module, verbose=True, ckpt_path='best')  # Test with the best model

        computed_cm = model.test_cm.compute()

        pprint('Test confusion matrix')
        pprint(computed_cm)

        csv_path = os.path.join(wandb.run.dir, 'predictions.csv')

        print(f'Saving predictions to {csv_path}')
        predictions = trainer.predict(model, datamodule=data_module, ckpt_path='best')

        save_predictions(predictions, csv_path, idx_to_class)


if __name__ == '__main__':
    # ------------
    # Set project and entity for wandb logging
    # ------------
    path_prefix = 'N2'
    sets_prefix = 'N2_agg'
    project = f'BRITS_LULC_SIPNA {path_prefix}'
    name = 'scratch soft mse 1213 topo'

    # ceama_path = /home/jrodriguez/data/SIPNA/13
    # ngpu_path = /mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/Andalucia/datasets
    default_config['data']['data_dir'] = '/home/jrodriguez/data/SIPNA/13/all_1213'
    default_config['data'][
        'train_labels_path'] = f'/home/jrodriguez/data/SIPNA/13/labels/{path_prefix}/{path_prefix}_1213/training_df_{sets_prefix}.csv'
    default_config['data'][
        'val_labels_path'] = f'/home/jrodriguez/data/SIPNA/13/labels/{path_prefix}/{path_prefix}_1213/val_df_{sets_prefix}.csv'
    default_config['data'][
        'test_labels_path'] = f'/home/jrodriguez/data/SIPNA/13/labels/{path_prefix}/{path_prefix}_1213/test_df_{sets_prefix}.csv'
    default_config['data'][
        'ancillary_path'] = '/home/jrodriguez/data/SIPNA/13/ancillary/topo_climatic_imputed.csv'

    """default_config['data']['data_dir'] = '/mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/Andalucia/datasets/all_1213'
    default_config['data']['train_labels_path'] = f'/mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/Andalucia/datasets' \
                                                  f'/labels/{path_prefix}/{path_prefix}_1213/training_df_{sets_prefix}.csv'
    default_config['data']['val_labels_path'] = f'/mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/Andalucia/datasets' \
                                                f'/labels/{path_prefix}/{path_prefix}_1213/val_df_{sets_prefix}.csv'
    default_config['data']['test_labels_path'] = f'/mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/Andalucia/datasets' \
                                                 f'/labels/{path_prefix}/{path_prefix}_1213/test_df_{sets_prefix}.csv'
    default_config['data']['ancillary_path'] = f'/mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/Andalucia/datasets/ancillary/topo_climatic_imputed.csv'
"""

    default_config['model']['rnn_hid_size'] = 100

    default_config['trainer']['fast_dev_run'] = True
    default_config['trainer']['max_epochs'] = 200


    train(project=project, run_name=name, config=default_config)
