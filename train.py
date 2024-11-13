import torch
from torch import optim
import lightning as L
from lightningmodule import IMNER
from data_module import ImnerDataModule
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger,WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint
class Hypernum:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
audiodatadir = "your_aishell1_data_dir"
hypernum = Hypernum(data_dir = "data/CNERTA",
                    model_dir = "MmspeechModelWeight/damo/ofa_mmspeech_pretrain_large_zh",
                    audio_dir = audiodatadir,             
                    wandb_project_name = "IMNER",
                    wandbdescription = None,
                    wandbwritepredtext = False,
                    uespretrainedmmspeech = True,
                    train_task = 'IMNER',#'TNER','MNER','SNER','IMNER'
                    seed = 42,
                    batch_size = 12,
                    epoch_num = 50,
                    warmup_rate = 0.1,
                    check_val_every_n_epoch = 1,
                    learning_rate = 2e-5,
                    log_every_n_steps = 50,
                    trainsamplerate = 1.0,
                    precision = '16-mixed',
                    deterministic = False,
                    gpu_device = [1],
                    generate_hyper = dict(num_beams=3, 
                                          num_return_sequences = 1,
                                          max_new_tokens = 50,
                                          output_scores = True
                                          )
                    )
seed_everything(hypernum.seed, workers=True)
#model and datamodule
model = IMNER(hypernum)

datamodule = ImnerDataModule(hypernum)

#logger
tblogger = TensorBoardLogger("lightning_logs", name="dev_test_model",default_hp_metric=False)
wandb_logger = WandbLogger(name = hypernum.wandbdescription,project=hypernum.wandb_project_name)

#callback
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath="./checkpoint", 
                                      filename='imner-{epoch:02d}-{f1maxmonitor:.2f}',
                                      save_top_k=1, 
                                      monitor="f1maxmonitor",
                                      mode = 'max',
                                      save_weights_only = True
                                      )

trainer = L.Trainer( max_epochs=hypernum.epoch_num,
                    logger = wandb_logger,
                    callbacks=[lr_monitor,checkpoint_callback],
                    log_every_n_steps=hypernum.log_every_n_steps,
                    enable_checkpointing=True,
                    devices=hypernum.gpu_device,
                    check_val_every_n_epoch=hypernum.check_val_every_n_epoch,
                    precision = hypernum.precision,
                    deterministic = hypernum.deterministic,
                    #gradient_clip_val=0.5,
                    #gradient_clip_algorithm = "norm"
                    )
trainer.fit(model=model, datamodule=datamodule)