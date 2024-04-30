from torch import nn
from torch import optim
import torch
import numpy as np


class NaiveWrapper(nn.Module):
    def __init__(self, model, optimizer, args, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.args = args

        self.clip_grad = True


    def train_(self, inputs, batch):
        """
        :param inputs: inputs to the model
        :param batch: batch of original data
        """
        loss, veridical_logits, natural_logits, primitive_logits, reason_logits  = self.model(**inputs)

        # update gradient
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.model.zero_grad()

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

        return loss, ver_acc, nat_acc, primitive_acc, reason_acc
