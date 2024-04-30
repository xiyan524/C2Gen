from torch import nn
from torch import optim
import torch
import numpy as np
from transformers.optimization import AdamW
from continual.er import ExperienceReplay

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


class AGEM(ExperienceReplay):
    def __init__(self, model, optimizer, args):
        super().__init__(model, optimizer, args)
        self.model = model
        self.optimizer = optimizer
        self.args = args

        # memory
        self.agem_k = args.agem_k
        self.grad_dims = []
        self.violation_count = 0
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

        self.init_memory()

    def store_grad(self, pp, grads, grad_dims):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
        """
        # store the gradients
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(param.grad.data.view(-1))
            cnt += 1

    def compute_grad(self, inputs):
        self.zero_grad()
        loss, _, _, _, _ = self.model(**inputs)
        loss.backward()
        grads = torch.Tensor(sum(self.grad_dims)).to(self.args.device)
        self.store_grad(self.parameters, grads, self.grad_dims)
        return grads

    def fix_grad(self, mem_grads):
        # check whether the current gradient interfere with the average gradients
        grads = torch.Tensor(sum(self.grad_dims)).to(mem_grads.device)
        self.store_grad(self.parameters, grads, self.grad_dims)
        dotp = torch.dot(grads, mem_grads)
        if dotp < 0:
            # project the grads back to the mem_grads
            # g_new = g - g^Tg_{ref} / g_{ref}^Tg_{ref} * g_{ref}
            new_grad = grads - (torch.dot(grads, mem_grads) / (torch.dot(mem_grads, mem_grads) + 1e-12)) * mem_grads
            overwrite_grad(self.parameters, new_grad, self.grad_dims)
            return 1
        else:
            return 0

    def train_(self, inputs, batch):
        """
        :param inputs: inputs to the model
        :param batch: batch of original data
        """
        memory_input = self.sample_mem_batch(self.args.device, self.agem_k)
        if memory_input is not None:
            # calculate gradients on the memory batch
            mem_grads = self.compute_grad(memory_input)

        # standard training
        self.optimizer.zero_grad()
        #loss, primitive_logits, reason_logits, reason_label_logits = self.model(**inputs)
        loss, veridical_logits, natural_logits, primitive_logits, reason_logits = self.model(**inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.max_grad_norm)
        if memory_input is not None:
            violated = self.fix_grad(mem_grads)
            self.violation_count += violated
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

        # update memory
        for index in range(self.args.batch_size):
            self.update_mem(inputs, index)

        return loss, ver_acc, nat_acc, primitive_acc, reason_acc
