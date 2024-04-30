from torch import nn
from torch import optim
import torch
import numpy as np
from continual.naive import NaiveWrapper
import copy
from transformers.optimization import AdamW
import math
import random

class ExperienceReplay(NaiveWrapper):
    def __init__(self, model, optimizer, args):
        super().__init__(model, optimizer, args)
        self.model = model
        self.optimizer = optimizer
        self.args = args

        # memory
        self.mem_limit = args.memory_size
        self.mem_bs = args.batch_size
        self.mem_occupied = {}
        self.init_memory()

    def init_memory(self):
        memory_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))

        memory_ver_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_ver_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_ver_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        
        memory_nat_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_nat_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_nat_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        
        memory_ver_labels = torch.tensor([-1] * self.mem_limit)
        memory_nat_labels = torch.tensor([-1] * self.mem_limit)
        memory_pair_labels = torch.tensor([-1] * self.mem_limit)
        memory_reason_labels = torch.tensor([-1] * self.mem_limit)

        self.memory = {'inputs': memory_inputs, 'attention_masks': memory_attention_masks, 'segments': memory_segments, \
        'ver_inputs': memory_ver_inputs, 'ver_attention_masks': memory_ver_attention_masks, 'ver_segments': memory_ver_segments, \
        'nat_inputs': memory_nat_inputs, 'nat_attention_masks': memory_nat_attention_masks, 'nat_segments': memory_nat_segments,  \
        'ver_labels': memory_ver_labels, 'nat_labels': memory_nat_labels, \
        'pair_labels': memory_pair_labels, 'reason_labels': memory_reason_labels}
        self.example_seen = 0

    def get_random(self, seed=1):
        random_state = None
        for i in range(seed):
            if random_state is None:
                random_state = np.random.RandomState(self.example_seen + self.args.seed)
            else:
                random_state = np.random.RandomState(random_state.randint(0, int(1e5)))
        return random_state

    def sample_mem_batch(self, device, k=None, seed=1):
        random_state = self.get_random(seed)
        if k is None:
            k = self.mem_bs
        # reservoir
        n_max = min(self.mem_limit, self.example_seen)
        available_indices = [_ for _ in range(n_max)]

        if not available_indices:
            return None
        elif len(available_indices) < k:
            indices = np.arange(n_max)
        else:
            indices = random_state.choice(available_indices, k, replace=False)

        indices = torch.tensor(indices).to(device)
        inputs = {"input_ids": self.memory['inputs'][indices].to(self.args.device),
                  "attention_mask": self.memory['attention_masks'][indices].to(self.args.device),
                  "token_type_ids": self.memory['segments'][indices].to(self.args.device),
                  "ver_input_ids": self.memory['ver_inputs'][indices].to(self.args.device),
                  "ver_attention_mask": self.memory['ver_attention_masks'][indices].to(self.args.device),
                  "ver_token_type_ids": self.memory['ver_segments'][indices].to(self.args.device),
                  "nat_input_ids": self.memory['nat_inputs'][indices].to(self.args.device),
                  "nat_attention_mask": self.memory['nat_attention_masks'][indices].to(self.args.device),
                  "nat_token_type_ids": self.memory['nat_segments'][indices].to(self.args.device),
                  "primitive_labels": self.memory['pair_labels'][indices].to(self.args.device),
                  "reason_labels": self.memory['reason_labels'][indices].to(self.args.device),
                  "ver_labels": self.memory['ver_labels'][indices].to(self.args.device),
                  "nat_labels": self.memory['nat_labels'][indices].to(self.args.device),
                  "mode": "train"}
        return inputs

    def update_mem(self, inputs, index):
        if self.example_seen < self.mem_limit:
            self.memory['inputs'][self.example_seen] = inputs['input_ids'][index]
            self.memory['attention_masks'][self.example_seen] = inputs['attention_mask'][index]
            self.memory['segments'][self.example_seen] = inputs['token_type_ids'][index]
            self.memory['ver_inputs'][self.example_seen] = inputs['ver_input_ids'][index]
            self.memory['ver_attention_masks'][self.example_seen] = inputs['ver_attention_mask'][index]
            self.memory['ver_segments'][self.example_seen] = inputs['ver_token_type_ids'][index]
            self.memory['nat_inputs'][self.example_seen] = inputs['nat_input_ids'][index]
            self.memory['nat_attention_masks'][self.example_seen] = inputs['nat_attention_mask'][index]
            self.memory['nat_segments'][self.example_seen] = inputs['nat_token_type_ids'][index]
            self.memory['pair_labels'][self.example_seen] = inputs['primitive_labels'][index]
            self.memory['reason_labels'][self.example_seen] = inputs['reason_labels'][index]
            self.memory['ver_labels'][self.example_seen] = inputs['ver_labels'][index]
            self.memory['nat_labels'][self.example_seen] = inputs['nat_labels'][index]
        else:
            rand_num = np.random.RandomState(self.example_seen + self.args.seed).randint(0, self.example_seen)
            if rand_num < self.mem_limit:
                self.memory['inputs'][rand_num] = inputs['input_ids'][index]
                self.memory['attention_masks'][rand_num] = inputs['attention_mask'][index]
                self.memory['segments'][rand_num] = inputs['token_type_ids'][index]
                self.memory['ver_inputs'][rand_num] = inputs['ver_input_ids'][index]
                self.memory['ver_attention_masks'][rand_num] = inputs['ver_attention_mask'][index]
                self.memory['ver_segments'][rand_num] = inputs['ver_token_type_ids'][index]
                self.memory['nat_inputs'][rand_num] = inputs['nat_input_ids'][index]
                self.memory['nat_attention_masks'][rand_num] = inputs['nat_attention_mask'][index]
                self.memory['nat_segments'][rand_num] = inputs['nat_token_type_ids'][index]
                self.memory['pair_labels'][rand_num] = inputs['primitive_labels'][index]
                self.memory['reason_labels'][rand_num] = inputs['reason_labels'][index]
                self.memory['ver_labels'][rand_num] = inputs['ver_labels'][index]
                self.memory['nat_labels'][rand_num] = inputs['nat_labels'][index]

        self.example_seen += 1

    def train_(self, inputs, batch):
        """
        :param inputs: inputs to the model
        :param batch: batch of original data
        """
        # standard training
        #loss, primitive_logits, reason_logits, _ = self.model(**inputs)
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

        #### memory training
        memory_input = self.sample_mem_batch(self.args.device)
        if memory_input is not None:
            mem_loss, _, _, _, _ = self.model(**memory_input)

            mem_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.model.zero_grad()

            #print("memory_loss:", mem_loss)
            loss = (loss + mem_loss) / 2

        # update memory
        for index in range(self.args.batch_size):
            self.update_mem(inputs, index)


        return loss, ver_acc, nat_acc, primitive_acc, reason_acc


class ExperienceReplayBuffer(NaiveWrapper):
    """Buffer ring"""
    def __init__(self, model, train_data, task_num, optimizer, args):
        super().__init__(model, optimizer, args)
        self.model = model
        self.train_data = train_data
        self.task_num = task_num
        self.optimizer = optimizer
        self.args = args

        # memory
        self.mem_limit = args.memory_size
        self.mem_bs = args.batch_size
        self.init_memory()

    def init_memory(self):
        self.example_seen = 0

        memory_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_ver_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_ver_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_ver_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_nat_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_nat_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_nat_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))

        memory_ver_labels = torch.tensor([-1] * self.mem_limit)
        memory_nat_labels = torch.tensor([-1] * self.mem_limit)
        memory_pair_labels = torch.tensor([-1] * self.mem_limit)
        memory_reason_labels = torch.tensor([-1] * self.mem_limit)

        self.each_task_memory_size = self.mem_limit // self.task_num
        self.each_task_data_num = len(self.train_data) // self.task_num
        for task_id in range(self.task_num):
            # sample data for each task
            random_state = self.get_random(self.args.seed+task_id)
            available_indices = np.arange(task_id * self.each_task_data_num, (task_id + 1) * self.each_task_data_num)
            indices = random_state.choice(available_indices, self.each_task_memory_size, replace=False)
            indices = torch.tensor(indices).to(self.args.device)

            start_index = task_id * self.each_task_memory_size
            end_index = (task_id + 1) * self.each_task_memory_size
            train_data_inputs = torch.stack(self.train_data.inputs, dim=0).to(self.args.device)
            train_data_attention_masks = torch.stack(self.train_data.attention_masks, dim=0).to(self.args.device)
            train_data_segments = torch.stack(self.train_data.segments, dim=0).to(self.args.device)
            
            train_data_ver_inputs = torch.stack(self.train_data.ver_inputs, dim=0).to(self.args.device)
            train_data_ver_attention_masks = torch.stack(self.train_data.ver_attention_masks, dim=0).to(self.args.device)
            train_data_ver_segments = torch.stack(self.train_data.ver_segments, dim=0).to(self.args.device)
            
            train_data_nat_inputs = torch.stack(self.train_data.nat_inputs, dim=0).to(self.args.device)
            train_data_nat_attention_masks = torch.stack(self.train_data.nat_attention_masks, dim=0).to(self.args.device)
            train_data_nat_segments = torch.stack(self.train_data.nat_segments, dim=0).to(self.args.device)
            
            train_data_ver_labels = torch.tensor(self.train_data.ver_labels).to(self.args.device)
            train_data_nat_labels = torch.tensor(self.train_data.nat_labels).to(self.args.device)
            train_data_pair_labels = torch.tensor(self.train_data.pair_labels).to(self.args.device)
            train_data_reason_labels = torch.tensor(self.train_data.reason_labels).to(self.args.device)

            memory_inputs[start_index: end_index] = train_data_inputs[indices]
            memory_attention_masks[start_index: end_index] = train_data_attention_masks[indices]
            memory_segments[start_index: end_index] = train_data_segments[indices]

            memory_ver_inputs[start_index: end_index] = train_data_ver_inputs[indices]
            memory_ver_attention_masks[start_index: end_index] = train_data_ver_attention_masks[indices]
            memory_ver_segments[start_index: end_index] = train_data_ver_segments[indices]

            memory_nat_inputs[start_index: end_index] = train_data_nat_inputs[indices]
            memory_nat_attention_masks[start_index: end_index] = train_data_nat_attention_masks[indices]
            memory_nat_segments[start_index: end_index] = train_data_nat_segments[indices]

            memory_ver_labels[start_index: end_index] = train_data_ver_labels[indices]
            memory_nat_labels[start_index: end_index] = train_data_nat_labels[indices]
            memory_pair_labels[start_index: end_index] = train_data_pair_labels[indices]
            memory_reason_labels[start_index: end_index] = train_data_reason_labels[indices]


        self.memory = {'inputs': memory_inputs, 'attention_masks': memory_attention_masks, 'segments': memory_segments, \
        'ver_inputs': memory_ver_inputs, 'ver_attention_masks': memory_ver_attention_masks, 'ver_segments': memory_ver_segments, \
        'nat_inputs': memory_nat_inputs, 'nat_attention_masks': memory_nat_attention_masks, 'nat_segments': memory_nat_segments,  \
        'ver_labels': memory_ver_labels, 'nat_labels': memory_nat_labels, \
        'pair_labels': memory_pair_labels, 'reason_labels': memory_reason_labels}


    def get_random(self, seed=1):
        random_state = None
        for i in range(seed):
            if random_state is None:
                random_state = np.random.RandomState(self.example_seen + self.args.seed)
            else:
                random_state = np.random.RandomState(random_state.randint(0, int(1e5)))
        return random_state

    def sample_mem_batch(self, device, k=None, seed=1):
        random_state = self.get_random(seed)
        if k is None:
            k = self.mem_bs

        current_task = int(self.example_seen // self.each_task_data_num)
        if current_task == 0:
            return None
        else:
            available_indices = np.arange(current_task * self.each_task_memory_size)

        indices = random_state.choice(available_indices, k, replace=False)
        indices = torch.tensor(indices).to(device)
        inputs = {"input_ids": self.memory['inputs'][indices].to(self.args.device),
                  "attention_mask": self.memory['attention_masks'][indices].to(self.args.device),
                  "token_type_ids": self.memory['segments'][indices].to(self.args.device),
                  "ver_input_ids": self.memory['ver_inputs'][indices].to(self.args.device),
                  "ver_attention_mask": self.memory['ver_attention_masks'][indices].to(self.args.device),
                  "ver_token_type_ids": self.memory['ver_segments'][indices].to(self.args.device),
                  "nat_input_ids": self.memory['nat_inputs'][indices].to(self.args.device),
                  "nat_attention_mask": self.memory['nat_attention_masks'][indices].to(self.args.device),
                  "nat_token_type_ids": self.memory['nat_segments'][indices].to(self.args.device),
                  "primitive_labels": self.memory['pair_labels'][indices].to(self.args.device),
                  "reason_labels": self.memory['reason_labels'][indices].to(self.args.device),
                  "ver_labels": self.memory['ver_labels'][indices].to(self.args.device),
                  "nat_labels": self.memory['nat_labels'][indices].to(self.args.device),
                  "mode": "train"}
        return inputs

    def train_(self, inputs, batch):
        """
        :param inputs: inputs to the model
        :param batch: batch of original data
        """
        # standard training
        #loss, primitive_logits, reason_logits, reason_label_logits = self.model(**inputs)
        loss, veridical_logits, natural_logits, primitive_logits, reason_logits  = self.model(**inputs)

        # update gradient
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.max_grad_norm)
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

        #### memory training
        memory_input = self.sample_mem_batch(self.args.device)
        if memory_input is not None:
            mem_loss, _, _, _, _ = self.model(**memory_input)
            mem_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.model.zero_grad()


            loss = (loss + mem_loss) / 2

        self.example_seen += len(batch[0])

        return loss, ver_acc, nat_acc, primitive_acc, reason_acc


class ExperienceReplayMIR(NaiveWrapper):
    """Maximally Interfered Sampling"""
    def __init__(self, model, optimizer, args):
        super().__init__(model, optimizer, args)
        self.model = model
        self.optimizer = optimizer
        self.args = args

        # memory
        self.mem_limit = args.memory_size
        self.mem_bs = args.batch_size
        self.mir_k = args.mir_k
        self.init_memory()

    def init_memory(self):
        memory_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))

        memory_ver_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_ver_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_ver_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        
        memory_nat_inputs = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_nat_attention_masks = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        memory_nat_segments = torch.tensor(np.zeros([self.mem_limit, self.args.max_input_num], dtype=np.int64))
        
        memory_ver_labels = torch.tensor([-1] * self.mem_limit)
        memory_nat_labels = torch.tensor([-1] * self.mem_limit)
        memory_pair_labels = torch.tensor([-1] * self.mem_limit)
        memory_reason_labels = torch.tensor([-1] * self.mem_limit)

        self.memory = {'inputs': memory_inputs, 'attention_masks': memory_attention_masks, 'segments': memory_segments, \
        'ver_inputs': memory_ver_inputs, 'ver_attention_masks': memory_ver_attention_masks, 'ver_segments': memory_ver_segments, \
        'nat_inputs': memory_nat_inputs, 'nat_attention_masks': memory_nat_attention_masks, 'nat_segments': memory_nat_segments,  \
        'ver_labels': memory_ver_labels, 'nat_labels': memory_nat_labels, \
        'pair_labels': memory_pair_labels, 'reason_labels': memory_reason_labels}
        self.example_seen = 0

    def get_random(self, seed=1):
        random_state = None
        for i in range(seed):
            if random_state is None:
                random_state = np.random.RandomState(self.example_seen + self.args.seed)
            else:
                random_state = np.random.RandomState(random_state.randint(0, int(1e5)))
        return random_state

    def pseudo_update(self, inputs):
        loss, _, _, _, _ = self.model(**inputs)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if isinstance(self.optimizer, AdamW):
            step_wo_state_update_adamw(self.optimizer)
        else:
            raise NotImplementedError

    def mir_memory(self, train_inputs, memory_inputs):
        if memory_inputs['input_ids'].size(0) < self.mir_k:
            return memory_inputs
        else:
            # store cache
            self.cache = copy.deepcopy(self.model.state_dict())

            # mir strategy
            with torch.no_grad():
                rst_memory_before_loss, _, _, _, _ = self.model(**memory_inputs, reduce=False)
            self.pseudo_update(train_inputs)
            with torch.no_grad():
                rst_memory_after_loss, _, _, _, _ = self.model(**memory_inputs, reduce=False)
            loss_change = rst_memory_after_loss - rst_memory_before_loss
            loss_change, _ = loss_change.view(memory_inputs['input_ids'].size(0), -1).max(1)
            _, topk = loss_change.topk(self.mir_k, largest=True)

            mir_memory_inputs = {"input_ids": memory_inputs['input_ids'][topk],
                             "attention_mask": memory_inputs['attention_mask'][topk],
                             "token_type_ids": memory_inputs['token_type_ids'][topk],
                             "ver_input_ids": memory_inputs['ver_input_ids'][topk],
                             "ver_attention_mask": memory_inputs['ver_attention_mask'][topk],
                             "ver_token_type_ids": memory_inputs['ver_token_type_ids'][topk],
                             "nat_input_ids": memory_inputs['nat_input_ids'][topk],
                             "nat_attention_mask": memory_inputs['nat_attention_mask'][topk],
                             "nat_token_type_ids": memory_inputs['nat_token_type_ids'][topk],
                             "primitive_labels": memory_inputs['primitive_labels'][topk],
                             "reason_labels": memory_inputs['reason_labels'][topk],
                             "ver_labels": memory_inputs['ver_labels'][topk],
                             "nat_labels": memory_inputs['nat_labels'][topk],
                             "mode": "train"}

            # load cache
            self.model.load_state_dict(self.cache)
            self.model.zero_grad()

            return mir_memory_inputs

    def sample_mem_batch(self, device, train_inputs, k=None, seed=1):
        random_state = self.get_random(seed)
        if k is None:
            k = self.mem_bs

        n_max = min(self.mem_limit, self.example_seen)
        available_indices = [_ for _ in range(n_max)]

        if not available_indices:
            return None
        elif len(available_indices) < k:
            indices = np.arange(n_max)
        else:
            indices = random_state.choice(available_indices, k, replace=False)

        indices = torch.tensor(indices).to(device)
        memory_inputs = {"input_ids": self.memory['inputs'][indices].to(self.args.device),
                  "attention_mask": self.memory['attention_masks'][indices].to(self.args.device),
                  "token_type_ids": self.memory['segments'][indices].to(self.args.device),
                  "ver_input_ids": self.memory['ver_inputs'][indices].to(self.args.device),
                  "ver_attention_mask": self.memory['ver_attention_masks'][indices].to(self.args.device),
                  "ver_token_type_ids": self.memory['ver_segments'][indices].to(self.args.device),
                  "nat_input_ids": self.memory['nat_inputs'][indices].to(self.args.device),
                  "nat_attention_mask": self.memory['nat_attention_masks'][indices].to(self.args.device),
                  "nat_token_type_ids": self.memory['nat_segments'][indices].to(self.args.device),
                  "primitive_labels": self.memory['pair_labels'][indices].to(self.args.device),
                  "reason_labels": self.memory['reason_labels'][indices].to(self.args.device),
                  "ver_labels": self.memory['ver_labels'][indices].to(self.args.device),
                  "nat_labels": self.memory['nat_labels'][indices].to(self.args.device),
                  "mode": "train"}
        mir_memory_inputs = self.mir_memory(train_inputs, memory_inputs)

        return mir_memory_inputs

    def update_mem(self, inputs, index):
        if self.example_seen < self.mem_limit:
            self.memory['inputs'][self.example_seen] = inputs['input_ids'][index]
            self.memory['attention_masks'][self.example_seen] = inputs['attention_mask'][index]
            self.memory['segments'][self.example_seen] = inputs['token_type_ids'][index]
            self.memory['ver_inputs'][self.example_seen] = inputs['ver_input_ids'][index]
            self.memory['ver_attention_masks'][self.example_seen] = inputs['ver_attention_mask'][index]
            self.memory['ver_segments'][self.example_seen] = inputs['ver_token_type_ids'][index]
            self.memory['nat_inputs'][self.example_seen] = inputs['nat_input_ids'][index]
            self.memory['nat_attention_masks'][self.example_seen] = inputs['nat_attention_mask'][index]
            self.memory['nat_segments'][self.example_seen] = inputs['nat_token_type_ids'][index]
            self.memory['pair_labels'][self.example_seen] = inputs['primitive_labels'][index]
            self.memory['reason_labels'][self.example_seen] = inputs['reason_labels'][index]
            self.memory['ver_labels'][self.example_seen] = inputs['ver_labels'][index]
            self.memory['nat_labels'][self.example_seen] = inputs['nat_labels'][index]
        else:
            rand_num = np.random.RandomState(self.example_seen + self.args.seed).randint(0, self.example_seen)
            if rand_num < self.mem_limit:
                self.memory['inputs'][rand_num] = inputs['input_ids'][index]
                self.memory['attention_masks'][rand_num] = inputs['attention_mask'][index]
                self.memory['segments'][rand_num] = inputs['token_type_ids'][index]
                self.memory['ver_inputs'][rand_num] = inputs['ver_input_ids'][index]
                self.memory['ver_attention_masks'][rand_num] = inputs['ver_attention_mask'][index]
                self.memory['ver_segments'][rand_num] = inputs['ver_token_type_ids'][index]
                self.memory['nat_inputs'][rand_num] = inputs['nat_input_ids'][index]
                self.memory['nat_attention_masks'][rand_num] = inputs['nat_attention_mask'][index]
                self.memory['nat_segments'][rand_num] = inputs['nat_token_type_ids'][index]
                self.memory['pair_labels'][rand_num] = inputs['primitive_labels'][index]
                self.memory['reason_labels'][rand_num] = inputs['reason_labels'][index]
                self.memory['ver_labels'][rand_num] = inputs['ver_labels'][index]
                self.memory['nat_labels'][rand_num] = inputs['nat_labels'][index]

        self.example_seen += 1

    def train_(self, inputs, batch):
        """
        :param inputs: inputs to the model
        :param batch: batch of original data
        """
        # standard training
        #loss, primitive_logits, reason_logits, reason_label_logits = self.model(**inputs)
        loss, veridical_logits, natural_logits, primitive_logits, reason_logits  = self.model(**inputs)

        # update gradient
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.max_grad_norm)
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

        #### memory training
        memory_input = self.sample_mem_batch(self.args.device, inputs)
        if memory_input is not None:
            mem_loss, _, _, _, _ = self.model(**memory_input)
            mem_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.model.zero_grad()

            loss = (loss + mem_loss) / 2

        # update memory
        for index in range(self.args.batch_size):
            self.update_mem(inputs, index)


        return loss, ver_acc, nat_acc, primitive_acc, reason_acc



def step_wo_state_update_adamw(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p]

            # State initialization
            #if len(state) == 0:
            #    state['step'] = 0
            #    # Exponential moving average of gradient values
            #    state['exp_avg'] = torch.zeros_like(p.data)
            #    # Exponential moving average of squared gradient values
            #    state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            #state['step'] += 1

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg = exp_avg.mul(beta1).add(1.0 - beta1, grad)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(1.0 - beta2, grad, grad)
            denom = exp_avg_sq.sqrt().add(group['eps'])

            step_size = group['lr']
            if group['correct_bias']:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(-step_size, exp_avg, denom)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            # Add weight decay at the end (fixed version)
            if group['weight_decay'] > 0.0:
                p.data.add_(-group['lr'] * group['weight_decay'], p.data)

    return loss
