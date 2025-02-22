import shutup

shutup.please()
import os
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
# from pytorch_lightning.strategies import DataParallelStrategy

from data.megadepth_datamodule import MegaDepthPairsDataModuleFeatures
from models.matching_module import MatchingTrainingModule
from utils.train_utils import get_training_loggers, get_training_callbacks, prepare_logging_directory


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--config', type=str, help='path to config file', default='config/config_cached.yaml')
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load('config/config_cached.yaml')  # base config
    if args.config != 'config/config_cached.yaml':
        add_conf = OmegaConf.load(args.config)
        config = OmegaConf.merge(config, add_conf)

    pl.seed_everything(int(os.environ.get('LOCAL_RANK', 0)))

    # moved assignment of features_config before experiment_name creation to facilitate correcting error inputting
    # the entire path to the extracted features being used as part of the experiment name
    features_config = OmegaConf.load(os.path.join(config['data']['root_path'],
                                               config['data']['features_dir'], 'config.yaml'))

    # Prepare directory for logs and checkpoints
    # Use features_config['name'] rather than config['data']['features_dir'] to prevent entire path
    # from being inputted as part of the experiment name, resulting in logs being saved in an unexpected location
    if os.environ.get('LOCAL_RANK', 0) == 0:
        experiment_name = '{}_cache__attn_{}__laf_{}__{}'.format(
            features_config['name'],
            config['superglue']['attention_gnn']['attention'],
            config['superglue']['laf_to_sideinfo_method'],
            str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        )
        log_path = prepare_logging_directory(config, experiment_name)
    else:
        experiment_name, log_path = '', ''

    # Init Lightning Data Module
    data_config = config['data']
    dm = MegaDepthPairsDataModuleFeatures(
        root_path=data_config['root_path'],
        train_list_path=data_config['train_list_path'],
        val_list_path=data_config['val_list_path'],
        test_list_path=data_config['test_list_path'],
        batch_size=data_config['batch_size_per_gpu'],
        num_workers=data_config['dataloader_workers_per_gpu'],
        target_size=data_config['target_size'],
        features_dir=data_config['features_dir'],
        num_keypoints=data_config['max_keypoints'],
        val_max_pairs_per_scene=data_config['val_max_pairs_per_scene'],
        balanced_train=data_config.get('balanced_train', False),
        train_pairs_overlap=data_config.get('train_pairs_overlap')
    )


    # Init model
    model = MatchingTrainingModule(
        train_config={**config['train'], **config['inference'], **config['evaluation']},
        features_config=features_config,
        superglue_config=config['superglue'],
    )

    # Set callbacks and loggers
    callbacks = get_training_callbacks(config, log_path, experiment_name)
    loggers = get_training_loggers(config, log_path, experiment_name)

    # Init distributed trainer
    # Replace accelerator="ddp" and plugins=DDPPlugin with strategy=DDPStrategy
    # due to recent pytorch lightning updates removing certain plugins/accelerators 
    # and replacing them with the better descriptor of strategy.
    trainer = pl.Trainer(
        gpus=config['gpus'],
        max_epochs=config['train']['epochs'],
        # accelerator="ddp",
        gradient_clip_val=config['train']['grad_clip'],
        log_every_n_steps=config['logging']['train_logs_steps'],
        limit_train_batches=config['train']['steps_per_epoch'],
        num_sanity_val_steps=5,
        callbacks=callbacks,
        logger=loggers,
        # strategy=DataParallelStrategy(),
        strategy=DDPStrategy(find_unused_parameters=False),
        # plugins=DDPPlugin(find_unused_parameters=False),
        precision=config['train'].get('precision', 32),
    )
    # If loaded from checkpoint - validate
    if config.get('checkpoint') is not None:
        trainer.validate(model, datamodule=dm, ckpt_path=config.get('checkpoint'))
    trainer.fit(model, datamodule=dm, ckpt_path=config.get('checkpoint'))


if __name__ == '__main__':
    main()
