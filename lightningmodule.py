from typing import Any
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger,WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from modeling_mmspeech import MMSpeechModel
from modelscope.models.multi_modal.ofa.configuration_mmspeech import MMSpeechConfig

class IMNER(L.LightningModule):
    def __init__(self, hypernum) -> None:
        super().__init__()
        self.hypernum = hypernum
        self.model_dir = hypernum.model_dir
        if hypernum.uespretrainedmmspeech:
            print('使用预训练的mmspeech')
            self.model = MMSpeechModel.from_pretrained(self.model_dir)
        else:
            print('使用随机初始化的mmspeech')
            self.modelconfig = MMSpeechConfig.from_pretrained(self.model_dir)
            self.model = MMSpeechModel(self.modelconfig)
        
            
        self.learning_rate = hypernum.learning_rate
        #标签label区
        self.symbal2nertype = {
            '<':'ORG-S',
            '>':'ORG-E',
            '(':'LOC-S',
            ')':'LOC-E',
            '[':'PER-S',
            ']':'PER-E'
        }

        self.nertype2symbal = dict(zip(self.symbal2nertype.values(),self.symbal2nertype.keys()))
        self.nertypes = ['ORG','LOC','PER']


        #全局变量区

        #用于评估和测试的全局变量
        self.gold_num = dict(a = 0,p = 0,ap = 0)
        self.pred_num = dict(a = 0,p = 0,ap = 0)
        self.pred_true = dict(a = 0,p = 0,ap = 0)


        #用于记录最大的f值
        self.ap_f1max = 0
        self.F14maxf1= dict(a = 0,p = 0,ap = 0)
        self.P4maxf1= dict(a = 0,p = 0,ap = 0)
        self.R4maxf1= dict(a = 0,p = 0,ap = 0)


        #用于记录预测结果的全局变量
        self.predtext_for_wandb_write = dict(a = [],p = [],ap = [])
        self.labeltext_for_wandb_write = dict(a = [],p = [],ap = [])

        self.save_hyperparameters()
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        stepping_batches = self.trainer.estimated_stepping_batches
        num_warmup_steps = int((self.hparams.hypernum.warmup_rate)*stepping_batches)
        scheduler = get_constant_schedule_with_warmup(optimizer = optimizer,num_warmup_steps = num_warmup_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
            "scheduler":scheduler,
            "interval": "step",
            "frequency": 1,
            }
        }
    def on_train_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.config.update(self.hparams.hypernum.__dict__)

    def training_step(self, batch,batch_idx) -> STEP_OUTPUT:
        model_input = batch['net_input']
        model_out = self.model(**model_input)
        logits = model_out.logits
        targets = batch['target']
        target_pad_id = self.trainer.datamodule.train_dataset.preprocessor.tokenizer.pad_token_id
        loss_fct = CrossEntropyLoss(ignore_index=target_pad_id)
        loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("train_loss", loss,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx,dataloader_idx=0) -> STEP_OUTPUT:
        devdataloaderkeys = list(self.trainer.val_dataloaders.keys())
        devdataloaderkey = devdataloaderkeys[dataloader_idx]

        batch['net_input'].pop('decoder_input_ids')
        gen_outputs = self.model.generate(**batch['net_input'],**self.hypernum.generate_hyper)
        target_tokenizer = self.trainer.datamodule.dev_dataset_ap.preprocessor.tokenizer
        self.phone_tokenizer = self.trainer.datamodule.dev_dataset_ap.preprocessor.text2phone_tokenizer
        
        #gen_text_batch每条样本包含多个候选得分
        gen_text_batch = target_tokenizer.batch_decode(gen_outputs, 
                                                       skip_special_tokens=True,
                                                       clean_up_tokenization_spaces = True)
        gen_text_batch = [gen_text_item.replace(' ','') for gen_text_item in gen_text_batch]
        labels_text = batch['labels']

        #仅用于测试！！
        #gen_text_batch = labels_text
        if devdataloaderkey == 'a':
            only_audio = True
        else:
            only_audio = False
        batch_entities_labelwithspan = self.batchtext2entitywithspan(labels_text,islabel = True,only_audio = only_audio)
        batch_entities_predwithspan = self.batchtext2entitywithspan(gen_text_batch,only_audio = only_audio)

        wrongflag = self.compute_metricwithspan_step_update(batch_entities_predwithspan,
                                                batch_entities_labelwithspan,
                                                devdataloaderkey
                                                )
        # if wrongflag:
        #     for l,p in zip(labels_text,gen_text_batch):
        #         print(devdataloaderkey+':')
        #         print('标签:{}'.format(l))
        #         print('预测值:{}'.format(p))

        wandb_write_temp = ['']*(self.hypernum.generate_hyper["num_return_sequences"])
        for pred_id,pred_text_item in enumerate(gen_text_batch):
            id_in_sample = pred_id%(self.hypernum.generate_hyper["num_return_sequences"])
            id_smaple = int(pred_id/self.hypernum.generate_hyper["num_return_sequences"])
            wandb_write_temp[id_in_sample]=pred_text_item
            if id_in_sample == (self.hypernum.generate_hyper["num_return_sequences"]-1):
                self.predtext_for_wandb_write[devdataloaderkey].append(wandb_write_temp)
                self.labeltext_for_wandb_write[devdataloaderkey].append(labels_text[id_smaple])
                wandb_write_temp = ['']*(self.hypernum.generate_hyper["num_return_sequences"])
    def on_validation_epoch_end(self):
        F1 = dict(a = 0,p = 0,ap = 0)
        P = dict(a = 0,p = 0,ap = 0)
        R = dict(a = 0,p = 0,ap = 0)


        for key in self.trainer.val_dataloaders.keys():
            P[key] = self.pred_true[key]/(self.pred_num[key]+1e-10)
            R[key] = self.pred_true[key]/(self.gold_num[key]+1e-10)
            F1[key] = 2*P[key]*R[key]/(P[key]+R[key]+1e-10)

        if F1['ap']>self.ap_f1max:

            self.ap_f1max = max(self.ap_f1max,F1['ap'])
            self.F14maxf1 = F1
            self.P4maxf1 = P
            self.R4maxf1 = R
            if isinstance(self.logger, WandbLogger) and self.hypernum.wandbwritepredtext:
                columns = []
                for key in self.trainer.val_dataloaders.keys():
                    columns = [key+'-'+"F1值"]+[key+'-'+'第{}个候选答案'.format(i) for i in range(self.hypernum.generate_hyper["num_return_sequences"])]+columns
                columns = ['标签']+columns

                writedata = []
                for ind in range(len(self.predtext_for_wandb_write['a'])):
                    writedata_temp = []
                    for key in self.predtext_for_wandb_write.keys():
                        writedata_temp = [F1[key]]+self.predtext_for_wandb_write[key][ind]+writedata_temp
                    writedata_temp = [self.labeltext_for_wandb_write[key][ind]]+writedata_temp
                    writedata.append(writedata_temp)

                self.logger.log_text(key = "epoch:{},".format(self.trainer.current_epoch),
                                columns=columns, 
                                data=writedata
                                )

        ner_metric_dic = dict(f1_a = F1['a'],f1_p = F1['p'],f1_ap = F1['ap'],
                              f1max_a = self.F14maxf1['a'],f1max_p = self.F14maxf1['p'],f1max_ap = self.F14maxf1['ap'],
                              P4maxf1_a = self.P4maxf1['a'],P4maxf1_p = self.P4maxf1['p'],P4maxf1_ap = self.P4maxf1['ap'],
                              R4maxf1_a = self.R4maxf1['a'],R4maxf1_p = self.R4maxf1['p'],R4maxf1_ap = self.R4maxf1['ap'],
                              f1maxmonitor = self.ap_f1max
                              )
        self.log_dict(ner_metric_dic,logger=True,on_epoch = True)
        #self.logger.log_metrics(ner_metric_dic)
        self.gold_num = dict(a = 0,p = 0,ap = 0)
        self.pred_num = dict(a = 0,p = 0,ap = 0)
        self.pred_true = dict(a = 0,p = 0,ap = 0)
        self.predtext_for_wandb_write = dict(a = [],p = [],ap = [])
        self.labeltext_for_wandb_write = dict(a = [],p = [],ap = [])

    def batchtext2entitywithspan(self,batch_text,islabel = False,only_audio = False):
        batch_entities = []
        text_item_entyties_temp = []
        if islabel:
            for text_item in batch_text:
                text_item_entyties = self.text2entitywithspan(text_item,only_audio= only_audio)
                batch_entities.append(text_item_entyties)
        else:
            for text_id,text_item in enumerate(batch_text):
                text_item_entyties = self.text2entitywithspan(text_item,only_audio = only_audio)
                for entity in text_item_entyties:
                    text_item_entyties_temp.append(entity)
                if (text_id+1)%self.hypernum.generate_hyper["num_return_sequences"]==0:
                    text_item_entyties_temp_vote = []
                    for entity_list in text_item_entyties_temp:
                        entity_list_count = self.count_specific_sublist(text_item_entyties_temp,entity_list)
                        if entity_list_count > (self.hypernum.generate_hyper["num_return_sequences"]/2) and (entity_list not in text_item_entyties_temp_vote):
                            text_item_entyties_temp_vote.append(entity_list)
                    batch_entities.append(text_item_entyties_temp_vote)
                    text_item_entyties_temp = []
        return batch_entities
    def count_specific_sublist(self,nested_list, sublist):
        count = 0
        for item in nested_list:
            if item == sublist:
                count += 1
        return count
    def text2entitywithspan(self,text,only_audio = False):
        entities_list_for = []
        entities_list_rev = []
        entity_temp = [0,0,0,0]
        rawtext_list = []
        entity_point_dic_temp={}
        str_ind = 0
        for str in text:
            if str in self.symbal2nertype:
                if entity_point_dic_temp.get(self.symbal2nertype[str]):
                    entity_point_dic_temp[self.symbal2nertype[str]].append(str_ind)
                else:
                    entity_point_dic_temp[self.symbal2nertype[str]] = []
                    entity_point_dic_temp[self.symbal2nertype[str]].append(str_ind)
            else:
                rawtext_list.append(str)
                str_ind+=1
        for ner_type in self.nertypes:
            s_symbal = ner_type+'-S'
            e_symbal = ner_type+'-E'
            if entity_point_dic_temp.get(s_symbal):
                #正向就近匹配解码实体
                for str_ind_s in range(len(rawtext_list)):
                    if str_ind_s in entity_point_dic_temp[s_symbal]:
                        entity_temp[0] = str_ind_s
                        if str_ind_s<len(rawtext_list):
                            for str_ind_e in range(str_ind_s+1,len(rawtext_list)):
                                if entity_point_dic_temp.get(e_symbal):#预测出来的实体，有些边界符号不完整
                                    if  str_ind_e in entity_point_dic_temp[e_symbal]:
                                        entity_temp[1] = str_ind_e
                                        entity_temp[2] = ner_type
                                        if only_audio:
                                            #entity_temp[3] = ''.join(rawtext_list[entity_temp[0]:entity_temp[1]])
                                            entity_temp[3] = self.phone_tokenizer.trans(''.join(rawtext_list[entity_temp[0]:entity_temp[1]]))[0]
                                            entity_temp[0] = 0
                                            entity_temp[1] = 0
                                            #entity_temp[2] = 0
                                            
                                        entities_list_for.append(entity_temp)
                                        entity_temp = [0,0,0,0]
                                        break
                #反向解码实体
                for str_ind_e in range(len(rawtext_list)-1,-1,-1):
                    if entity_point_dic_temp.get(e_symbal):#预测出来的实体，有些边界符号不完整
                        if str_ind_e in entity_point_dic_temp[e_symbal]:
                            entity_temp[1] = str_ind_e
                            if str_ind_e>0:
                                for str_ind_s in range(str_ind_e-1,-1,-1):
                                    if entity_point_dic_temp.get(s_symbal):
                                        if str_ind_s in entity_point_dic_temp[s_symbal]:
                                            entity_temp[0] = str_ind_s
                                            entity_temp[2] = ner_type
                                            if only_audio:
                                                #entity_temp[3] = ''.join(rawtext_list[entity_temp[0]:entity_temp[1]])
                                                entity_temp[3] = self.phone_tokenizer.trans(''.join(rawtext_list[entity_temp[0]:entity_temp[1]]))[0]
                                                entity_temp[0] = 0
                                                entity_temp[1] = 0
                                                #entity_temp[2] = 0
                                                
                                            entities_list_rev.append(entity_temp)
                                            entity_temp = [0,0,0,0]
                                            break
        # 将内部列表转换为元组，以便能够被添加到集合中
        set_for = {tuple(item) for item in entities_list_for}
        set_rev = {tuple(item) for item in entities_list_rev}

        merged_list = [list(item) for item in set_for.union(set_rev)]#返回的实体列表里最好不要有重复实体。
        return merged_list
    def compute_metricwithspan_step_update(self,pred_batch,label_batch,dataloaderkey):
        gold_num_batch = 0
        pred_num_batch = 0
        pred_true_num_batch = 0
        wrong_flag = False

        for pred_item,label_item in zip(pred_batch,label_batch):
            pred_true_count_temp = 0

            gold_num_item = len(label_item)
            pred_num_item = len(pred_item)

            for pred_item_e in pred_item:
                if pred_item_e in label_item:
                    pred_true_count_temp+=1
                else:
                    wrong_flag = True
            pred_true_num_item = pred_true_count_temp

            gold_num_batch+=gold_num_item
            pred_num_batch+=pred_num_item
            pred_true_num_batch+=pred_true_num_item
        
                #用于评估和测试的全局变量
        self.gold_num[dataloaderkey] +=gold_num_batch
        self.pred_num[dataloaderkey] +=pred_num_batch
        self.pred_true[dataloaderkey] +=pred_true_num_batch
        return wrong_flag