import torch
from transformers import RobertaConfig, RobertaModel
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
import time


    

class RoBertaMTSep(nn.Module):
    def __init__(self, model_type, primitive_class, reason_class, loss_ratio, dropout):
        super().__init__()
        self.primitive_class = primitive_class
        self.reason_class = reason_class
        self.roberta = RobertaModel.from_pretrained(model_type)
        self.config = RobertaConfig.from_pretrained(model_type)
        self.loss_ratio = loss_ratio

        # self.trans_fn_candidate = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        
        self.veridical_layer = nn.Linear(self.config.hidden_size, self.reason_class)
        self.natural_layer = nn.Linear(self.config.hidden_size, self.reason_class)
        self.reason_layer = nn.Linear(self.config.hidden_size, self.reason_class) 

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, 
                ver_input_ids, ver_attention_mask, ver_token_type_ids,
                nat_input_ids, nat_attention_mask, nat_token_type_ids,
                ver_labels=None, nat_labels=None,
                reduce=True, primitive_labels=None, reason_labels=None, stage="s1", order="y", mode="train"):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        inputs_rep = outputs[1] # [batch, dim]
        batch_size = inputs_rep.size()[0]

        ver_outputs = self.roberta(input_ids=ver_input_ids, attention_mask=ver_attention_mask)
        ver_inputs_rep = ver_outputs[1] # [batch, dim]
        nat_outputs = self.roberta(input_ids=nat_input_ids, attention_mask=nat_attention_mask)
        nat_inputs_rep = nat_outputs[1] # [batch, dim]

        # inputs_rep_trans = self.trans_fn_candidate()
        veridical_logits = self.veridical_layer(ver_inputs_rep)
        natural_logits = self.natural_layer(nat_inputs_rep) # [batch, 3]
        reason_logits = self.reason_layer(inputs_rep)   # [batch, 3]

        # composed logits
        composition_logits = veridical_logits.repeat_interleave(3, dim=1) + natural_logits.repeat(1, 3)
        
        self.loss_fn = nn.CrossEntropyLoss(reduce=reduce)
        if mode == "train":
            loss_reason = self.loss_fn(reason_logits, reason_labels)
            if stage == "s1":
                if order == "y":
                    loss_ver = self.loss_fn(veridical_logits, ver_labels)
                    loss = loss_reason + loss_ver
                else:
                    loss_nat = self.loss_fn(natural_logits, nat_labels)
                    loss = loss_reason + loss_nat
            else:
                if order == "y":
                    loss_nat = self.loss_fn(natural_logits, nat_labels)
                    loss = loss_reason + loss_nat
                else:
                    loss_ver = self.loss_fn(veridical_logits, ver_labels)
                    loss = loss_reason + loss_ver

            return loss, veridical_logits, natural_logits, composition_logits, reason_logits
        else:
            return 0, veridical_logits, natural_logits, composition_logits, reason_logits
         



