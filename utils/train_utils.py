from datetime import datetime

import os
import shutil
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils.lightning_callbacks import FavorAttentionProjectionRedrawCallback


def prepare_logging_directory(config, experiment_name, features_config=None):
    # create logging directory
    print(config['logging']['root_path'], config['logging']['name'], experiment_name)
    log_path = os.path.join(config['logging']['root_path'], config['logging']['name'], experiment_name)
    print('log path', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    # also save features config from features directory
    if 'features_dir' in config['data']:
        shutil.copyfile(
            src=os.path.join(config['data']['root_path'], config['data']['features_dir'], 'config.yaml'),
            dst=os.path.join(log_path, 'features_config.yaml')
        )
    if features_config is not None:
        with open(os.path.join(log_path, 'features_config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(features_config))
    else:
        #create features config to access descriptor_dim for both training types
        features_config = OmegaConf.load(os.path.join(log_path, 'features_config.yaml'))

    '''
        automatically set superglue: descriptor_dim, superglue: positional_encoding: output_size, 
        and superglue:attention_gnn: embed_dim to be the same as descriptor dim in features config
        detailing which network to use for feature extraction in the case of realtime extraction
        and which network was used in the case of pre-extraction. These values will be saved
        In the main config file used in the inference step.
    '''
    descriptor_dim = features_config['descriptor_dim']
    config['superglue']['descriptor_dim'] = descriptor_dim
    config['superglue']['positional_encoding']['output_size'] = descriptor_dim
    config['superglue']['attention_gnn']['embed_dim'] = descriptor_dim
    
    # write main config
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(config))

    return log_path


def get_training_callbacks(config, log_path, experiment_name):
    # MatchingModule callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename='superglue-{step:d}',
        every_n_epochs=1,
        save_top_k=-1
    )
    lr_monitor = LearningRateMonitor()

    callbacks = [checkpoint_callback, lr_monitor]

    # optional favor attention callback
    attention_type = config['superglue']['attention_gnn']['attention']
    if attention_type.startswith('favor'):
        callbacks.append(FavorAttentionProjectionRedrawCallback(
            redraw_every_n_steps=config['train'].get('favor_redraw_projection_every_n_steps', 1000)))

    return callbacks


def get_training_loggers(config, log_path, experiment_name):
    tb_logger = TensorBoardLogger(config['logging']['root_path'], config['logging']['name'], version=experiment_name)
    wandb_logger = WandbLogger(name=experiment_name, project="superglue")
    if os.environ.get('LOCAL_RANK', 0) == 0:
        wandb_logger.experiment.config.update(OmegaConf.to_container(config))

    return [tb_logger, wandb_logger]
