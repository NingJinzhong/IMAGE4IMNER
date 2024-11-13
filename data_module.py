import os
import re
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import random
import lightning as L
from torch.utils.data import Dataset, DataLoader, TensorDataset
from preprocessor import IMNERPreprocessor
from modeling_mmspeech import MMSpeechModel
from modelscope.utils.constant import ModeKeys
from modelscope.utils.config import Config
from modelscope.preprocessors import Preprocessor

from collate import collate_fn
from functools import partial


class IMNERDataset(Dataset):
    def __init__(self,data_dir,model_dir,audio_dir,hasaudio = True,hasphone = True,ap_sample = False,trainsamplerate = 1.0):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.audio_dir = audio_dir
        self.hasaudio = hasaudio
        self.hasphone = hasphone
        self.ap_sample = ap_sample
        self.all_data = self.readdatafile(trainsamplerate)
        self.cfg = Config.from_file(os.path.join(self.model_dir,"configuration.json"))
        self.preprocessor = IMNERPreprocessor(cfg = self.cfg,
                                              model_dir = self.model_dir,
                                              mode = ModeKeys.TRAIN,
                                              hasaudio = self.hasaudio,
                                              hasphone = self.hasphone,
                                              ap_sample = self.ap_sample
                                              )
        self.audio_ind2dir_dic = self.get_audio_dir_index(self.audio_dir)
    def __getitem__(self, index):
        audiofile,text = self.all_data[index]
        audiofiledir = self.audio_ind2dir_dic[audiofile]
        result = self.preprocessor({'wav':audiofiledir,'text':text})
        return result
    def __len__(self):
        return len(self.all_data)
    def readdatafile(self,trainsamplerate):
        data_item_list = []
        with open(self.data_dir,'r',encoding='utf-8') as f:
            for data_item in f.readlines():
                data_item_list.append((data_item.strip().split(' ')[0],data_item.strip().split(' ')[1]))
        data_item_list = self.sample_from_list(data_item_list,trainsamplerate)
        return data_item_list
    def sample_from_list(self,data_item_list, trainsamplerate):
        if trainsamplerate<=1.0:
            sample_size = int(len(data_item_list) * trainsamplerate)
        else:
            sample_size = trainsamplerate
        return random.sample(data_item_list, sample_size)
    def get_audio_dir_index(self,audio_dir):
        index2dirdic = {}
        for curDir, dirs, files in os.walk(audio_dir):
            for file in files:
                if not file.endswith('.tar.gz'):
                    index2dirdic[file.split('.')[0]] = os.path.join(curDir,file)
        return index2dirdic
    
class ImnerDataModule(L.LightningDataModule):
    def __init__(self,hypernum) -> None:
        super().__init__()
        self.data_dir = hypernum.data_dir
        self.model_dir = hypernum.model_dir
        self.audio_dir = hypernum.audio_dir
        self.batch_size = hypernum.batch_size
        self.train_task = hypernum.train_task
        self.trainsamplerate = hypernum.trainsamplerate
    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.train_task == 'IMNER':
                self.train_dataset = IMNERDataset(os.path.join(self.data_dir,'train.txt'),
                                                self.model_dir,
                                                self.audio_dir,
                                                ap_sample = True,
                                                trainsamplerate = self.trainsamplerate
                                                )
            if self.train_task == 'TNER':
                self.train_dataset = IMNERDataset(os.path.join(self.data_dir,'train.txt'),
                                                self.model_dir,
                                                self.audio_dir,
                                                hasaudio= False,
                                                hasphone=True,
                                                trainsamplerate = self.trainsamplerate
                                                )
            if self.train_task == 'MNER':
                self.train_dataset = IMNERDataset(os.path.join(self.data_dir,'train.txt'),
                                                self.model_dir,
                                                self.audio_dir,
                                                hasaudio= True,
                                                hasphone=True,
                                                trainsamplerate = self.trainsamplerate
                                                )
            if self.train_task == 'SNER':
                self.train_dataset = IMNERDataset(os.path.join(self.data_dir,'train.txt'),
                                                self.model_dir,
                                                self.audio_dir,
                                                hasaudio= True,
                                                hasphone=False,
                                                trainsamplerate = self.trainsamplerate
                                                )

        if stage == "test":
            self.test_dataset = IMNERDataset(os.path.join(self.data_dir,'test.txt'),
                                              self.model_dir,
                                              self.audio_dir
                                              )
        if stage == "predict":
            self.predict_dataset = IMNERDataset(os.path.join(self.data_dir,'test.txt'),
                                              self.model_dir,
                                              self.audio_dir
                                              )
        self.dev_dataset_a = IMNERDataset(os.path.join(self.data_dir,'test.txt'),
                                            self.model_dir,
                                            self.audio_dir,
                                            hasaudio= True,
                                            hasphone=False
                                            )
        self.dev_dataset_p = IMNERDataset(os.path.join(self.data_dir,'test.txt'),
                                            self.model_dir,
                                            self.audio_dir,
                                            hasaudio= False,
                                            hasphone=True
                                            )
        self.dev_dataset_ap = IMNERDataset(os.path.join(self.data_dir,'test.txt'),
                                            self.model_dir,
                                            self.audio_dir,
                                            hasaudio= True,
                                            hasphone=True
                                            )
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        my_collate = partial(collate_fn,
                                pad_idx = self.train_dataset.preprocessor.tokenizer.pad_token_id,
                                eos_idx = self.train_dataset.preprocessor.tokenizer.bos_token_id)
        traindataloader = DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=8,collate_fn=my_collate)
        
        return traindataloader
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        my_collate = partial(collate_fn,
                                pad_idx = self.dev_dataset_ap.preprocessor.tokenizer.pad_token_id,
                                eos_idx = self.dev_dataset_ap.preprocessor.tokenizer.bos_token_id)
        devdataloader_a = DataLoader(self.dev_dataset_a,
                                     batch_size=self.batch_size,
                                     num_workers=8,
                                     collate_fn=my_collate)
        devdataloader_p = DataLoader(self.dev_dataset_p,
                                     batch_size=self.batch_size,
                                     num_workers=8,
                                     collate_fn=my_collate)
        devdataloader_ap = DataLoader(self.dev_dataset_ap,
                                     batch_size=self.batch_size,
                                     num_workers=8,
                                     collate_fn=my_collate)
        return {'a':devdataloader_a,'p':devdataloader_p,'ap':devdataloader_ap}

    
if __name__ == "__main__":
    pass
