from torch import nn
from torch import optim
import torch
import numpy as np
from transformers.optimization import AdamW
from continual.er import ExperienceReplay
import copy
torch.autograd.set_detect_anomaly(True)

class KD(ExperienceReplay):
    def __init__(self, model, optimizer, args):
        super().__init__(model, optimizer, args)
        self.model = model
        self.prev_model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.args = args

        # memory
        self.kd_k = args.kd_k
        self.init_memory()

    def train_(self, inputs, batch, task_follow=False, updata_cache=False):
        """
        :param inputs: inputs to the model
        :param batch: batch of original data
        """

        memory_input = self.sample_mem_batch(self.args.device, self.kd_k)

        # standard training
        #loss, primitive_logits, reason_logits, reason_label_logits = self.model(**inputs)
        loss, veridical_logits, natural_logits, primitive_logits, reason_logits  = self.model(**inputs)


        # previous model for memory
        #if memory_input is not None:
        if task_follow:
            with torch.no_grad():
                self.prev_model.load_state_dict(self.cache)
                self.prev_model.cuda()
                before_memory_loss, before_veridical_logits, before_natural_logits, _, \
                before_reason_logits = self.prev_model(**memory_input)


            #with torch.no_grad():
            after_memory_loss, after_veridical_logits, after_natural_logits, _, \
            after_reason_logits = self.model(**memory_input)

            #kd_primitive_loss = nn.functional.mse_loss(before_memory_primitive_logits,
                                                       #after_memory_primitive_logits)
            kd_ver_loss = nn.functional.mse_loss(before_veridical_logits, after_veridical_logits)
            kd_nat_loss = nn.functional.mse_loss(before_natural_logits, after_natural_logits)
            kd_reason_loss = nn.functional.mse_loss(before_reason_logits,
                                                    after_reason_logits)
            #kd_loss = kd_primitive_loss + kd_reason_loss
            kd_loss = kd_ver_loss + kd_nat_loss + kd_reason_loss
            #print("kd_primitive_loss=", kd_primitive_loss)
            #print("kd_reason_loss=", kd_reason_loss)
            loss += self.args.kd_lambda * kd_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.model.zero_grad()

        if updata_cache:
            self.cache = copy.deepcopy(self.model.state_dict())

        # check accuracy
        ver_labels = batch[11].detach().cpu().numpy()
        ver_predictions = torch.argmax(veridical_logits, dim=1).detach().cpu().numpy()
        ver_acc = np.equal(ver_predictions, ver_labels)
        ver_acc = np.sum(ver_acc) / len(ver_acc)

        nat_labels = batch[12].detach().cpu().numpy()
        nat_predictions = torch.argmax(natural_logits, dim=1).detach().cpu().numpy()
        nat_acc = np.equal(nat_predictions, nat_labels)
        nat_acc = np.sum(nat_acc) / len(nat_acc)

        primitive_labels = batch[3].detach().cpu().numpy()
        primitive_predictions = torch.argmax(primitive_logits, dim=1).detach().cpu().numpy()
        primitive_acc = np.equal(primitive_predictions, primitive_labels)
        primitive_acc = np.sum(primitive_acc) / len(primitive_acc)

        reason_labels = batch[4].detach().cpu().numpy()
        reason_predictions = torch.argmax(reason_logits, dim=1).detach().cpu().numpy()
        reason_acc = np.equal(reason_predictions, reason_labels)
        reason_acc = np.sum(reason_acc) / len(reason_acc)

        # update memory
        for index in range(self.args.batch_size):
            self.update_mem(inputs, index)


        return loss, ver_acc, nat_acc, primitive_acc, reason_acc
