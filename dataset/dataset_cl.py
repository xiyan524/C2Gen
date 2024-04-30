from multiprocessing import context
import os
import json
import random
import torch

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Subset


class ComposNLIMTCLSEPDataset(Dataset):
    """seperate inputs"""
    def __init__(self, file_path, params, file_name, do_shuffle=False, do_multiple_round=False, do_incre_train=False, do_continual=False, do_noise_label=False, mode='train'):
        self.mode = mode
        self.seed= params['seed']
        self.tokenizer = params['tokenizer']
        self.max_seq_len = params['max_seq_len']
        self.multiple_round_num = params['multiple_round_num']
        self.do_noise_label = params['do_noise_label']
        #self.multiple_round_num = 0
        self.do_shuffle = do_shuffle
        self.do_multiple_round = do_multiple_round
        self.do_incre_train = do_incre_train
        self.do_continual = do_continual
        self.file_name = file_name
        self.file_path = os.path.join(file_path, self.file_name)
        
        self.candidate_labels()
        self.set_seed(self.seed)
        self.load_dataset()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_dataset(self):
        self.inputs = []
        self.segments = []
        self.attention_masks = []

        self.ver_inputs = []
        self.ver_segments = []
        self.ver_attention_masks = []

        self.nat_inputs = []
        self.nat_segments = []
        self.nat_attention_masks = []

        self.ver_labels = []
        self.nat_labels = []
        self.pair_labels = []
        self.reason_labels = []

        self.change_count = 0

        with open(self.file_path) as f:
            data_lines = f.readlines()
            for data_line in data_lines:
                data_line = data_line.strip()
                if data_line:
                    data = json.loads(data_line)
                    sent1 = data['sent1']
                    sent2 = data['sent2']
                    mid_sent = data['mid_sent']
                    
                    # for dax experiments
                    if self.do_noise_label:
                        sent1, sent2, mid_sent = self.nat_dax_replace(sent1, sent2, mid_sent)
                        if sent1 == "":
                            continue
                        sent1 = self.verb_dax_replace(sent1)

                    veridical_label = data['veridical_label']
                    sick_label = data['sick_label']
                    label = data['label']

                    # composition inputs
                    res = self.tokenizer(
                        text=sent1,
                        text_pair=sent2,
                        max_length=self.max_seq_len,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    input_ids = res.input_ids[0]
                    attention_mask = res.attention_mask[0]
                    if "token_type_ids" in res:
                        segment_ids = res.token_type_ids[0]
                    else:
                        segment_ids = torch.zeros_like(input_ids)

                    # veridical inference inputs
                    ver_res = self.tokenizer(
                        text=sent1,
                        text_pair=mid_sent,
                        max_length=self.max_seq_len,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    ver_input_ids = ver_res.input_ids[0]
                    ver_attention_mask = ver_res.attention_mask[0]
                    if "token_type_ids" in res:
                        ver_segment_ids = ver_res.token_type_ids[0]
                    else:
                        ver_segment_ids = torch.zeros_like(ver_input_ids)

                    # natural inference inputs
                    nat_res = self.tokenizer(
                        text=mid_sent,
                        text_pair=sent2,
                        max_length=self.max_seq_len,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    nat_input_ids = nat_res.input_ids[0]
                    nat_attention_mask = nat_res.attention_mask[0]
                    if "token_type_ids" in res:
                        nat_segment_ids = nat_res.token_type_ids[0]
                    else:
                        nat_segment_ids = torch.zeros_like(nat_input_ids)

                    # input labels
                    pair_label = [veridical_label + "_v", sick_label + "_n"]
                    pair_label = " ".join(pair_label)
                    ver_id = self.ver_dict[veridical_label]
                    nat_id = self.nat_dict[sick_label]
                    pair_id = self.primitive_pairs2idx[pair_label]
                    reason_id = self.reason_pairs_dict[label]

                    self.inputs.append(input_ids)
                    self.segments.append(segment_ids)
                    self.attention_masks.append(attention_mask)
                    self.ver_inputs.append(ver_input_ids)
                    self.ver_segments.append(ver_segment_ids)
                    self.ver_attention_masks.append(ver_attention_mask)
                    self.nat_inputs.append(nat_input_ids)
                    self.nat_segments.append(nat_segment_ids)
                    self.nat_attention_masks.append(nat_attention_mask)
                    self.ver_labels.append(ver_id)
                    self.nat_labels.append(nat_id)
                    self.pair_labels.append(pair_id)
                    self.reason_labels.append(reason_id)
                    
                else:
                    print("wrong line in file" + data_line)

        def split_task():
            total_num = len(self.inputs)
            each_task_num = int(total_num / 4)
            print("1.++", total_num, each_task_num)

            if self.do_incre_train:
                self.task1 = list(zip(self.inputs[each_task_num * 0:each_task_num * 1],
                                 self.segments[each_task_num * 0:each_task_num * 1],
                                 self.attention_masks[each_task_num * 0:each_task_num * 1],
                                 self.pair_labels[each_task_num * 0:each_task_num * 1],
                                 self.reason_labels[each_task_num * 0:each_task_num * 1]))

                self.task2 = list(zip(self.inputs[each_task_num * 0:each_task_num * 2],
                                 self.segments[each_task_num * 0:each_task_num * 2],
                                 self.attention_masks[each_task_num * 0:each_task_num * 2],
                                 self.pair_labels[each_task_num * 0:each_task_num * 2],
                                 self.reason_labels[each_task_num * 0:each_task_num * 2]))

                self.task3 = list(zip(self.inputs[each_task_num * 0:each_task_num * 3],
                                 self.segments[each_task_num * 0:each_task_num * 3],
                                 self.attention_masks[each_task_num * 0:each_task_num * 3],
                                 self.pair_labels[each_task_num * 0:each_task_num * 3],
                                 self.reason_labels[each_task_num * 0:each_task_num * 3]))

                self.task4 = list(zip(self.inputs[each_task_num * 0:], self.segments[each_task_num * 0:],
                                 self.attention_masks[each_task_num * 0:], self.pair_labels[each_task_num * 0:],
                                 self.reason_labels[each_task_num * 0:]))
                print("2.++ ++")
            else:
                self.task1 = list(zip(self.inputs[each_task_num * 0:each_task_num * 1],
                                      self.segments[each_task_num * 0:each_task_num * 1],
                                      self.attention_masks[each_task_num * 0:each_task_num * 1],
                                      self.ver_inputs[each_task_num * 0:each_task_num * 1],
                                      self.ver_segments[each_task_num * 0:each_task_num * 1],
                                      self.ver_attention_masks[each_task_num * 0:each_task_num * 1],
                                      self.nat_inputs[each_task_num * 0:each_task_num * 1],
                                      self.nat_segments[each_task_num * 0:each_task_num * 1],
                                      self.nat_attention_masks[each_task_num * 0:each_task_num * 1],
                                      self.ver_labels[each_task_num * 0:each_task_num * 1],
                                      self.nat_labels[each_task_num * 0:each_task_num * 1],
                                      self.pair_labels[each_task_num * 0:each_task_num * 1],
                                      self.reason_labels[each_task_num * 0:each_task_num * 1]))

                self.task2 = list(zip(self.inputs[each_task_num * 1:each_task_num * 2],
                                      self.segments[each_task_num * 1:each_task_num * 2],
                                      self.attention_masks[each_task_num * 1:each_task_num * 2],
                                      self.ver_inputs[each_task_num * 1:each_task_num * 2],
                                      self.ver_segments[each_task_num * 1:each_task_num * 2],
                                      self.ver_attention_masks[each_task_num * 1:each_task_num * 2],
                                      self.nat_inputs[each_task_num * 1:each_task_num * 2],
                                      self.nat_segments[each_task_num * 1:each_task_num * 2],
                                      self.nat_attention_masks[each_task_num * 1:each_task_num * 2],
                                      self.ver_labels[each_task_num * 1:each_task_num * 2],
                                      self.nat_labels[each_task_num * 1:each_task_num * 2],
                                      self.pair_labels[each_task_num * 1:each_task_num * 2],
                                      self.reason_labels[each_task_num * 1:each_task_num * 2]))

                self.task3 = list(zip(self.inputs[each_task_num * 2:each_task_num * 3],
                                      self.segments[each_task_num * 2:each_task_num * 3],
                                      self.attention_masks[each_task_num * 2:each_task_num * 3],
                                      self.ver_inputs[each_task_num * 2:each_task_num * 3],
                                      self.ver_segments[each_task_num * 2:each_task_num * 3],
                                      self.ver_attention_masks[each_task_num * 2:each_task_num * 3],
                                      self.nat_inputs[each_task_num * 2:each_task_num * 3],
                                      self.nat_segments[each_task_num * 2:each_task_num * 3],
                                      self.nat_attention_masks[each_task_num * 2:each_task_num * 3],
                                      self.ver_labels[each_task_num * 2:each_task_num * 3],
                                      self.nat_labels[each_task_num * 2:each_task_num * 3],
                                      self.pair_labels[each_task_num * 2:each_task_num * 3],
                                      self.reason_labels[each_task_num * 2:each_task_num * 3]))

                self.task4 = list(zip(self.inputs[each_task_num * 3:], 
                                      self.segments[each_task_num * 3:],
                                      self.attention_masks[each_task_num * 3:],
                                      self.ver_inputs[each_task_num * 3:], 
                                      self.ver_segments[each_task_num * 3:],
                                      self.ver_attention_masks[each_task_num * 3:], 
                                      self.nat_inputs[each_task_num * 3:], 
                                      self.nat_segments[each_task_num * 3:],
                                      self.nat_attention_masks[each_task_num * 3:], 
                                      self.ver_labels[each_task_num * 3:],
                                      self.nat_labels[each_task_num * 3:], 
                                      self.pair_labels[each_task_num * 3:],
                                      self.reason_labels[each_task_num * 3:]))

            stage1 = self.task1+self.task2
            stage2 = self.task3+self.task4
            random.shuffle(stage1)
            random.shuffle(stage2)
            self.task1 = stage1[:each_task_num]
            self.task2 = stage1[each_task_num:]
            self.task3 = stage2[:each_task_num]
            self.task4 = stage2[each_task_num:]
            

            def multiple_round_task(task_data):
                """repaet multiple round for each task"""
                task_data_inputs = []
                task_data_segments = []
                task_data_attention_masks = []

                task_data_ver_inputs = []
                task_data_ver_segments = []
                task_data_ver_attention_masks = []

                task_data_nat_inputs = []
                task_data_nat_segments = []
                task_data_nat_attention_masks = []

                task_data_ver_labels = []
                task_data_nat_labels = []
                task_data_pair_labels = []
                task_data_reason_labels = []
                for i in range(self.multiple_round_num):
                    random.seed(self.seed + i)
                    random.shuffle(task_data)
                    cur_inputs, cur_segments, cur_attention_masks, cur_ver_inputs, cur_ver_segments, cur_ver_attention_masks, cur_nat_inputs, cur_nat_segments, cur_nat_attention_masks, cur_ver_labels, cur_nat_labels, cur_pair_labels, cur_reason_labels = zip(*task_data)
                    task_data_inputs.extend(cur_inputs)
                    task_data_segments.extend(cur_segments)
                    task_data_attention_masks.extend(cur_attention_masks)
                    task_data_ver_inputs.extend(cur_ver_inputs)
                    task_data_ver_segments.extend(cur_ver_segments)
                    task_data_ver_attention_masks.extend(cur_ver_attention_masks)
                    task_data_nat_inputs.extend(cur_nat_inputs)
                    task_data_nat_segments.extend(cur_nat_segments)
                    task_data_nat_attention_masks.extend(cur_nat_attention_masks)
                    task_data_ver_labels.extend(cur_ver_labels)
                    task_data_nat_labels.extend(cur_nat_labels)
                    task_data_pair_labels.extend(cur_pair_labels)
                    task_data_reason_labels.extend(cur_reason_labels)
                return task_data_inputs, task_data_segments, task_data_attention_masks, \
                    task_data_ver_inputs, task_data_ver_segments, task_data_ver_attention_masks, \
                    task_data_nat_inputs, task_data_nat_segments, task_data_nat_attention_masks, \
                    task_data_ver_labels, task_data_nat_labels, \
                    task_data_pair_labels, task_data_reason_labels

            if self.do_multiple_round:
                print("do multiple round")
                task1_inputs, task1_segments, task1_attention_masks, \
                task1_ver_inputs, task1_ver_segments, task1_ver_attention_masks, \
                task1_nat_inputs, task1_nat_segments, task1_nat_attention_masks, \
                task1_ver_labels, task1_nat_labels, \
                task1_pair_labels, task1_reason_labels = multiple_round_task(self.task1)

                task2_inputs, task2_segments, task2_attention_masks, \
                task2_ver_inputs, task2_ver_segments, task2_ver_attention_masks, \
                task2_nat_inputs, task2_nat_segments, task2_nat_attention_masks,  \
                task2_ver_labels, task2_nat_labels, \
                task2_pair_labels, task2_reason_labels = multiple_round_task(self.task2)

                task3_inputs, task3_segments, task3_attention_masks, \
                task3_ver_inputs, task3_ver_segments, task3_ver_attention_masks, \
                task3_nat_inputs, task3_nat_segments, task3_nat_attention_masks,  \
                task3_ver_labels, task3_nat_labels, \
                task3_pair_labels, task3_reason_labels = multiple_round_task(self.task3)

                task4_inputs, task4_segments, task4_attention_masks, \
                task4_ver_inputs, task4_ver_segments, task4_ver_attention_masks, \
                task4_nat_inputs, task4_nat_segments, task4_nat_attention_masks,  \
                task4_ver_labels, task4_nat_labels, \
                task4_pair_labels, task4_reason_labels = multiple_round_task(self.task4)
            else:
                task1_inputs, task1_segments, task1_attention_masks, \
                task1_ver_inputs, task1_ver_segments, task1_ver_attention_masks, \
                task1_nat_inputs, task1_nat_segments, task1_nat_attention_masks, \
                task1_ver_labels, task1_nat_labels, \
                task1_pair_labels, task1_reason_labels = zip(*self.task1)
                
                task2_inputs, task2_segments, task2_attention_masks, \
                task2_ver_inputs, task2_ver_segments, task2_ver_attention_masks, \
                task2_nat_inputs, task2_nat_segments, task2_nat_attention_masks,  \
                task2_ver_labels, task2_nat_labels, \
                task2_pair_labels, task2_reason_labels = zip(*self.task2)

                task3_inputs, task3_segments, task3_attention_masks, \
                task3_ver_inputs, task3_ver_segments, task3_ver_attention_masks, \
                task3_nat_inputs, task3_nat_segments, task3_nat_attention_masks,  \
                task3_ver_labels, task3_nat_labels, \
                task3_pair_labels, task3_reason_labels = zip(*self.task3)

                task4_inputs, task4_segments, task4_attention_masks, \
                task4_ver_inputs, task4_ver_segments, task4_ver_attention_masks, \
                task4_nat_inputs, task4_nat_segments, task4_nat_attention_masks,  \
                task4_ver_labels, task4_nat_labels, \
                task4_pair_labels, task4_reason_labels = zip(*self.task4)
                

            self.inputs = task1_inputs + task2_inputs + task3_inputs + task4_inputs
            self.segments = task1_segments + task2_segments + task3_segments + task4_segments
            self.attention_masks = task1_attention_masks + task2_attention_masks + \
                                    task3_attention_masks + task4_attention_masks

            self.ver_inputs = task1_ver_inputs + task2_ver_inputs + task3_ver_inputs + task4_ver_inputs
            self.ver_segments = task1_ver_segments + task2_ver_segments + task3_ver_segments + task4_ver_segments
            self.ver_attention_masks = task1_ver_attention_masks + task2_ver_attention_masks + \
                                    task3_ver_attention_masks + task4_ver_attention_masks

            self.nat_inputs = task1_nat_inputs + task2_nat_inputs + task3_nat_inputs + task4_nat_inputs
            self.nat_segments = task1_nat_segments + task2_nat_segments + task3_nat_segments + task4_nat_segments
            self.nat_attention_masks = task1_nat_attention_masks + task2_nat_attention_masks + \
                                    task3_nat_attention_masks + task4_nat_attention_masks
                                  

            self.ver_labels = task1_ver_labels + task2_ver_labels + task3_ver_labels + task4_ver_labels
            self.nat_labels = task1_nat_labels + task2_nat_labels + task3_nat_labels + task4_nat_labels
            self.pair_labels = task1_pair_labels + task2_pair_labels + task3_pair_labels + task4_pair_labels
            self.reason_labels = task1_reason_labels + task2_reason_labels + task3_reason_labels + task4_reason_labels
           

        if self.do_continual:
            print("do continual")
            split_task()

        self.total_size = len(self.inputs)
        self.indexes = list(range(self.total_size))
        self.primitve_label_size = 9
        self.reason_label_size = 3
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def verb_dax_replace(self, sent):
        verb_dax_dict = {"manage":"blicke", "begin":"dmaop", "serve":"fanuo", "start":"dqpor", "dare":"dnje", "use":"aol", "get":"dew", "come":"fqoo", "hope":"lugi", "wish":"fepo", 
        "expect":"kikioa", "try":"zup", "plan":"wifr", "want":"nvru", "intend":"fajwiw", "appear":"askjei", "forget":"qbhdua", "fail":"mfkd", "refuse":"qneopl", "decline":"qnreiui", "remain":"qmaoip"}

        for verb in verb_dax_dict.keys():
            if verb in sent:
                sent = sent.replace(verb, verb_dax_dict[verb])
                break

        return sent

    def nat_dax_replace(self, str1, str2, mid_sent):
        rule_lst = ["isn't-is", 'Nobody-Someone', 'woman-man', 'sitting-standing', 'outside-indoors', 'Someone-Nobody', 'soccer-tennis', 'outside-inside', "is-isn't", 'unfolding-folding', 'on-off', 'bow-gun', 'off-onto', 'speaking-silent', 'striking-missing', 'onto-off', 'small-big', 'friends-enemies', 'wet-dry', 'hitting-missing', 'cleaning-dirtying', 'bottle-pot', 'bottom-top', 'stopping-running', 'standing-running', 'football-tennis', 'sleeping-eating', 'shunning-following', 'short-long', 'day-night', 'indoors-outdoors', 'sitting-dancing', 'fasting-eating', 'hot-cold', 'person-man', 'kickboxing-fighting', 'boys-kids', 'kids-children', 'gun-weapon', 'boy-child', 'tree-plant', 'schoolgirl-girl', 'resting-sitting', 'ball-toy', 'racing-running', 'walking-pacing', 'child-kid', 'outside-outdoors', 'watched-attended', 'men-people', 'The-An', 'guys-people', 'woman-lady', 'in-on', 'riding-racing', 'man-person', 'fish-food', 'prawn-shrimp', 'bowl-container', 'hiking-walking', 'slicing-cutting', 'amalgamating-mixing', 'performing-playing', 'pianist-person', 'man-musician', 'playing-strumming', 'carefully-cautiously', 'car-vehicle', 'quietly-peacefully', 'polished-cleaned', 'riding-driving', 'cord-rope', 'desk-table', 'lady-woman', 'motorbike-motorcycle', 'people-persons', 'strumming-playing', 'pacing-walking', 'cutting-slicing', 'breaking-cracking', 'boulder-rock', 'devouring-eating', 'cooking-preparing', 'practicing-playing', 'dicing-cutting', 'teenage-young', 'car-automobile', 'chopping-slicing', 'scaring-frightening', 'placing-putting', 'strolling-walking', 'cleaning-cleansing', 'Someone-Somebody', 'wiping-spreading', 'throwing-serving', 'Somebody-Someone', 'playing-practicing', 'bike-bicycle', 'striking-hitting', 'looking-staring', 'taking-picking', 'cap-hat', 'creek-stream', 'path-track', 'man-male', 'hurling-throwing', 'resting-lying', 'jumping-bouncing', 'cleaning-erasing', 'board-panel', 'eating-biting', 'scratching-stroking', 'rapidly-quickly', 'cutting-chopping', 'airplane-aircraft', 'picking-taking', 'banana-fruit', 'jet-plane', 'note-paper', 'lipstick-makeup', 'rhino-animal', 'ocean-water', 'brushing-styling', 'healing-reviving', 'sofa-couch', 'battling-fighting', 'boy-kid', 'cutting-shortening', 'seashore-beach', 'monkey-chimp', 'fixing-applying', 'pistol-weapon', 'acrobatics-tricks', 'cooking-roasting', 'little-young', 'woman-person', 'pot-bowl', 'pull-ups-exercises', 'knife-weapon', 'wok-pan', 'shouting-barking', 'chopped-sliced', 'jumping-climbing', 'speaking-talking', 'talking-speaking', 'frying-cooking', 'rifle-weapon', 'preparing-cooking', 'ringers-wrestlers', 'piece-slice', 'baby-child', 'motorcycle-motorbike', 'walking-strolling', 'meat-food', 'horse-animal', 'rabbit-bunny', 'jumping-diving', 'sausages-meat', 'guitar-instrument', 'box-container', 'checking-reading', 'boy-guy', 'bowl-dish', 'drums-instrument', 'marching-walking', 'laptop-computer', 'aiming-handling', 'girl-person', 'slab-block', 'sword-blade', 'wiping-cleaning', 'kettle-pot', 'dog-animal', 'shotgun-gun', 'shotgun-weapon', 'famous-great', 'doll-toy', 'baby-cub', 'cracking-breaking', 'container-box', 'canoe-boat', 'noodles-food', 'women-persons', 'boxing-fighting', 'walking-wading', 'beach-shore', 'ship-boat', 'group-pack', 'speeding-riding', 'female-girl', 'running-speeding', 'little-small', 'large-big', 'jumping-leaping', 'outdoors-outside', 'shades-sunglasses', 'leaping-jumping', 'children-kids', 'frolicking-playing', 'sprinting-running', 'kid-child', 'dark-darkened', 'guys-blokes', 'small-little', 'rocks-boulders', 'shore-sand', 'lunging-jumping', 'photograph-photo', 'river-stream', 'moving-splashing', 'ATVs-vehicles', 'a-jacket', 'sea-water', 'pool-water', 'Boys-People', 'man-player', 'grass-lawn', 'diving-jumping', 'road-path', 'path-trail', 'boys-children', 'ruined-tattered', 'volleyball-ball', 'purchasing-buying', 'beers-drinks', 'drawing-tattoo', 'lady-girl', 'barrier-hurdle', 'bottle-container', 'grouped-gathered', 'dress-veil', 'dogs-animals', 'walking-moving', 'getting-pushing', 'beside-near', 'cars-vehicles', 'running-moving', 'man-racer', 'shop-building', 'bride-girl', 'turning-going', 'lake-water', 'running-sprinting', 'sleeping-lying', 'field-grass', 'jeep-car', 'jeep-vehicle', 'sniffing-investigating', 'left-side', 'person-cyclist', 'against-near', 'pausing-stopping', 'baseball-ball', 'street-road', 'table-console', 'men-hikers', 'clustered-sitting', 'sitting-gathered', 'in-across', 'behind-near', 'coat-jacket', 'by-people', 'beach-sand', 'couch-sofa', 'man-rider', 'dunes-sand', 'black-dark', 'dirty-muddy', 'shop-store', 'standing-waiting', 'top-of', 'animal-dog', 'vehicle-car', 'women-girl', 'over-jumping', 'construction-work', 'palace-building', 'big-huge', 'hitting-kicking', 'man-model', 'Men-People', 'air-wind', 'stuntman-person', 'boy-swimmer', 'floor-ground', 'cluster-group', 'nap-rest', 'puppy-dog', 'trail-path', 'man-biker', "church-building's", 'carrying-biting', 'stars-sky', 'river-water', 'woman-girl', 'young-little', 'firing-shooting', 'carrying-holding', 'arriving-leaving', 'unstitching-sewing', 'into-past', 'beach-park', 'guitar-keyboard', 'girl-boy', 'dancing-sleeping', 'cats-dogs', 'biting-dropping', 'trekking-sitting', 'black-white', 'dog-cat', 'outdoor-indoor', 'down-up', 'stage-podium', 'desert-wooded', 'playing-dropping', 'combing-arranging', 'right-left', 'new-broken', 'happy-sad', 'excitement-boredom', 'cat-dog', 'knife-pencil', 'chasing-losing', 'hanging-leaning', 'women-men', 'seashore-sidewalk', 'bike-car', 'bun-tomato', 'man-woman', 'boy-girl', 'dashing-jumping', 'skateboard-bike', 'paper-sheets', 'dropping-carrying', 'flying-perching', 'man-girl', 'painting-drawing', 'small-large', 'folding-unfolding', 'sitting-walking', 'skating-resting', 'rain-snow', 'woods-city', 'gym-park', 'watching-playing', 'carrying-planting', 'man-surfer', 'man-monkey', 'opening-closing', 'white-black', 'placing-cooking', 'sleeping-playing', 'large-small', 'writing-typing', 'football-basketball', 'water-dirt', 'eating-seasoning', 'potatoes-carrots', 'hand-feet', 'pool-ocean', 'talking-laughing', 'man-dog', 'laughing-spitting', 'dirt-clean', 'boy-woman', 'standing-sitting', 'resting-walking', 'dancing-motionless']
        
        dax_word_dict = {"isn't": 'fvboh', 'is': 'rw', 'Nobody': 'mqjfjr', 'Someone': 'fiurbgb', 'woman': 'oyutm', 'man': 'qpj', 'sitting': 'yxecqtz', 'standing': 'filjyhyc', 'outside': 'yuwuaws', 'indoors': 'czxeymn', 'soccer': 'jyddzb', 'tennis': 'xmvexo', 'inside': 'shvmds', 'unfolding': 'vnvpennhs', 'folding': 'srvadyp', 'on': 'nz', 'off': 'cyf', 'bow': 'khr', 'gun': 'hta', 'onto': 'dpko', 'speaking': 'usuofkud', 'silent': 'iunfxh', 'striking': 'xfdhiect', 'missing': 'znanvon', 'small': 'noquz', 'big': 'srv', 'friends': 'psiczru', 'enemies': 'bylkweo', 'wet': 'xiw', 'dry': 'vcs', 'hitting': 'lkqkeix', 'cleaning': 'lwqagpjs', 'dirtying': 'thvczomr', 'bottle': 'dtqutd', 'pot': 'lvz', 'bottom': 'noyctc', 'top': 'mnc', 'stopping': 'xzncddhy', 'running': 'hpzsyfu', 'football': 'phiwfvnt', 'sleeping': 'gxlfswli', 'eating': 'bnwixh', 'shunning': 'msbjxpam', 'following': 'kswbtpxpg', 'short': 'bihvm', 'long': 'lpey', 'day': 'ahc', 'night': 'fdyoc', 'outdoors': 'jfbbwezx', 'dancing': 'legahae', 'fasting': 'mxidgbt', 'hot': 'vwv', 'cold': 'pyqv', 'person': 'fibqpc', 'kickboxing': 'erteskmpvb', 'fighting': 'ffwtkftu', 'boys': 'slck', 'kids': 'rsva', 'children': 'ihyixggm', 'weapon': 'ayknso', 'boy': 'ytr', 'child': 'oelmm', 'tree': 'eyle', 'plant': 'reoxz', 'schoolgirl': 'veuaehvsap', 'girl': 'cink', 'resting': 'okonahr', 'ball': 'buat', 'toy': 'gdg', 'racing': 'ajmmpy', 'walking': 'aoqsiph', 'pacing': 'ogxfjc', 'kid': 'wtt', 'watched': 'nvaeknk', 'attended': 'mksrxtta', 'men': 'jwe', 'people': 'cqhrpx', 'The': 'bsr', 'An': 'cu', 'guys': 'ptht', 'lady': 'shnx', 'in': 'li', 'riding': 'fwitkr', 'fish': 'ddwj', 'food': 'qore', 'prawn': 'jmcgf', 'shrimp': 'epdyav', 'bowl': 'cqus', 'container': 'ffzueuvch', 'hiking': 'hlteto', 'slicing': 'mvdurmw', 'cutting': 'zoioack', 'amalgamating': 'zbimuoxdqvmw', 'mixing': 'avhnuj', 'performing': 'wbaoyqxyks', 'playing': 'ekpjiyl', 'pianist': 'yzqzfgv', 'musician': 'bpzahcsk', 'strumming': 'llwewydgo', 'carefully': 'wsoenksds', 'cautiously': 'pvjgpbyiac', 'car': 'cly', 'vehicle': 'bwmckhc', 'quietly': 'tazgsqd', 'peacefully': 'cdjltiioho', 'polished': 'zdmyapjc', 'cleaned': 'ggcsqxa', 'driving': 'lajdmxb', 'cord': 'smqu', 'rope': 'twuc', 'desk': 'kmwn', 'table': 'mlyqr', 'motorbike': 'wsdyofgaf', 'motorcycle': 'aoozgvzexf', 'persons': 'cugmwlb', 'breaking': 'niwtzvjg', 'cracking': 'gmrwkybn', 'boulder': 'azhpprc', 'rock': 'sxyr', 'devouring': 'zgyiphmab', 'cooking': 'tuhknhf', 'preparing': 'hqplrwlas', 'practicing': 'nuderjuwnx', 'dicing': 'pvyilj', 'teenage': 'saxwmtb', 'young': 'emssk', 'automobile': 'tnjhgxrxeo', 'chopping': 'gdcdmkof', 'scaring': 'uejpxmf', 'frightening': 'vscglqzddpj', 'placing': 'eoonywp', 'putting': 'jzlgovm', 'strolling': 'ryzzpdubk', 'cleansing': 'hlkkldufd', 'Somebody': 'jzgcgiyd', 'wiping': 'fmcjpk', 'spreading': 'xzdgtjnvb', 'throwing': 'gfmjzgrp', 'serving': 'isathuh', 'bike': 'qsay', 'bicycle': 'sbwaprv', 'looking': 'pqbbuhn', 'staring': 'ccunspk', 'taking': 'ukifxq', 'picking': 'etxtmqq', 'cap': 'kda', 'hat': 'dqj', 'creek': 'ipbzp', 'stream': 'xhzqne', 'path': 'pggc', 'track': 'xwfdg', 'male': 'bjek', 'hurling': 'yjubpdz', 'lying': 'sccui', 'jumping': 'hhafqqx', 'bouncing': 'cagwysbc', 'erasing': 'dvbgtqn', 'board': 'bqmri', 'panel': 'vhhsp', 'biting': 'qibbue', 'scratching': 'pmqkcwfitx', 'stroking': 'smiconnq', 'rapidly': 'sxaokpw', 'quickly': 'zssgjuk', 'airplane': 'ledpfsbd', 'aircraft': 'jztstoya', 'banana': 'siabct', 'fruit': 'hccuz', 'jet': 'vul', 'plane': 'tsvcp', 'note': 'sdzn', 'paper': 'dwktm', 'lipstick': 'epsujeif', 'makeup': 'mwrkft', 'rhino': 'acujs', 'animal': 'ykzlqj', 'ocean': 'yoeac', 'water': 'udkqs', 'brushing': 'tsjacuwt', 'styling': 'bxbownu', 'healing': 'yuangpd', 'reviving': 'wsdxvuar', 'sofa': 'vnjs', 'couch': 'mmvbe', 'battling': 'hfosxkos', 'shortening': 'fgongsvfau', 'seashore': 'zknlcdix', 'beach': 'enjlz', 'monkey': 'dmcmpk', 'chimp': 'tgskd', 'fixing': 'rrsuum', 'applying': 'bvaafxgx', 'pistol': 'xzfydp', 'acrobatics': 'wtanisftts', 'tricks': 'llbdvy', 'roasting': 'fmxvlzsk', 'little': 'tzkotf', 'pull-ups': 'xjstptae', 'exercises': 'tekxnenly', 'knife': 'wieqa', 'wok': 'hyl', 'pan': 'gwz', 'shouting': 'qatljirg', 'barking': 'ltlcfcr', 'chopped': 'ixiyyjb', 'sliced': 'cttgad', 'climbing': 'mvgmjopd', 'talking': 'alzwqwy', 'frying': 'botldy', 'rifle': 'xofgu', 'ringers': 'lljubmk', 'wrestlers': 'khvmheahi', 'piece': 'fnhit', 'slice': 'nfijj', 'baby': 'owdh', 'meat': 'fdwa', 'horse': 'igige', 'rabbit': 'rizvxs', 'bunny': 'ruigz', 'diving': 'hzuzxa', 'sausages': 'vdngmoyf', 'guitar': 'fefwah', 'instrument': 'pyulvpegis', 'box': 'bqa', 'checking': 'qneyffva', 'reading': 'xvcqnou', 'guy': 'jtz', 'dish': 'axnz', 'drums': 'ezrtp', 'marching': 'xeklanaj', 'laptop': 'mmhiap', 'computer': 'divknrsn', 'aiming': 'zgvxpe', 'handling': 'mxxxpbpq', 'slab': 'pesg', 'block': 'irohq', 'sword': 'kmrsu', 'blade': 'fhhhk', 'kettle': 'wyavwc', 'dog': 'ozf', 'shotgun': 'rdkotuw', 'famous': 'foejgk', 'great': 'fdbvm', 'doll': 'qghl', 'cub': 'rcn', 'canoe': 'ytbzt', 'boat': 'adxb', 'noodles': 'xsngulo', 'women': 'brzue', 'boxing': 'rpnwzl', 'wading': 'lguzjk', 'shore': 'igwcz', 'ship': 'xwes', 'group': 'ztvzr', 'pack': 'rioa', 'speeding': 'bhzjnhqi', 'female': 'prihmh', 'large': 'snnbb', 'leaping': 'zsswpms', 'shades': 'nvuesg', 'sunglasses': 'egbvofumih', 'frolicking': 'hjboqagfsl', 'sprinting': 'yqcsvojqw', 'dark': 'btvc', 'darkened': 'lqdppjwf', 'blokes': 'shnrgg', 'rocks': 'smbgo', 'boulders': 'grmuuaxh', 'sand': 'ritd', 'lunging': 'eitgfwx', 'photograph': 'ombyqbemhg', 'photo': 'vdmqz', 'river': 'uuqxa', 'moving': 'vjuauh', 'splashing': 'aacesnknv', 'ATVs': 'omck', 'vehicles': 'bmqkmbhk', 'a': 'i', 'jacket': 'walhra', 'sea': 'sww', 'pool': 'hkpd', 'Boys': 'ixgz', 'People': 'rpmmwp', 'player': 'bzlpdk', 'grass': 'opidx', 'lawn': 'bamd', 'road': 'hmko', 'trail': 'urofo', 'ruined': 'ivvmzp', 'tattered': 'yeuwmdmu', 'volleyball': 'lxwaeqclhu', 'purchasing': 'skwxuzqdsv', 'buying': 'hxuyna', 'beers': 'ijzgw', 'drinks': 'crpwfq', 'drawing': 'lpohwbe', 'tattoo': 'ngnmgh', 'barrier': 'jwakohl', 'hurdle': 'brlqiz', 'grouped': 'rxphmrz', 'gathered': 'kdouuwau', 'dress': 'qrzhw', 'veil': 'pszn', 'dogs': 'qsie', 'animals': 'yuhycnx', 'getting': 'zkeemmx', 'pushing': 'itqewns', 'beside': 'ncqdbr', 'near': 'glmj', 'cars': 'nhat', 'racer': 'oojlv', 'shop': 'nxxz', 'building': 'gqbdqlhw', 'bride': 'icuyu', 'turning': 'ovrpuze', 'going': 'rxutq', 'lake': 'mwsr', 'field': 'odvgs', 'jeep': 'rswg', 'sniffing': 'qndosrju', 'investigating': 'wslkeykjpmjbw', 'left': 'mrdl', 'side': 'dpaj', 'cyclist': 'yngzwzw', 'against': 'xaapcfp', 'pausing': 'bhxzszl', 'baseball': 'jimmmeiz', 'street': 'pnbodt', 'console': 'knssvjf', 'hikers': 'auelhm', 'clustered': 'vrqdpcxpl', 'across': 'evlcfr', 'behind': 'wrnpxx', 'coat': 'nlnx', 'by': 'xf', 'rider': 'noyvk', 'dunes': 'mechf', 'black': 'mawau', 'dirty': 'okzsd', 'muddy': 'znoqf', 'store': 'fvmec', 'waiting': 'hrbrmgh', 'of': 'id', 'over': 'xsat', 'construction': 'llbwigtywvxj', 'work': 'zlaq', 'palace': 'vnorjj', 'huge': 'vmzu', 'kicking': 'xiryeey', 'model': 'kxawc', 'Men': 'deg', 'air': 'gom', 'wind': 'zndd', 'stuntman': 'fxudrjcd', 'swimmer': 'cdinbln', 'floor': 'amycn', 'ground': 'znstip', 'cluster': 'itfguek', 'nap': 'ukt', 'rest': 'inov', 'puppy': 'okehp', 'biker': 'yovju', 'church': 'eirnjv', "building's": 'onlzxmfxmy', 'carrying': 'ffjxjzlp', 'stars': 'nvymf', 'sky': 'bjg', 'firing': 'keyzam', 'shooting': 'xnkdfpyc', 'holding': 'iirbyca', 'arriving': 'tjqguaqt', 'leaving': 'acabocj', 'unstitching': 'avqftmwyhob', 'sewing': 'cjmrys', 'into': 'lyup', 'past': 'jjcn', 'park': 'wlxj', 'keyboard': 'uyzbhpoc', 'cats': 'lmll', 'dropping': 'vhinzpnn', 'trekking': 'ghxnruhx', 'white': 'dspyo', 'cat': 'hup', 'outdoor': 'jbrorls', 'indoor': 'qqerwu', 'down': 'tzab', 'up': 'zm', 'stage': 'uxnbv', 'podium': 'xwgvaq', 'desert': 'vpfuvs', 'wooded': 'nhqtll', 'combing': 'fghhdec', 'arranging': 'zcjtjkfzy', 'right': 'nbuaz', 'new': 'ptn', 'broken': 'cehfcp', 'happy': 'aelbf', 'sad': 'zer', 'excitement': 'tfqwlnrfnu', 'boredom': 'koqiovo', 'pencil': 'umebhe', 'chasing': 'mxxjiir', 'losing': 'inumyp', 'hanging': 'tuttqav', 'leaning': 'llrpaaz', 'sidewalk': 'ypumvmhi', 'bun': 'lhj', 'tomato': 'hxkmcc', 'dashing': 'cqbekcv', 'skateboard': 'eumuthgkli', 'sheets': 'sbzxif', 'flying': 'fmbzjy', 'perching': 'oamuiwel', 'painting': 'wwdoemap', 'skating': 'gdejaxf', 'rain': 'fcxn', 'snow': 'vplw', 'woods': 'amjav', 'city': 'tiyy', 'gym': 'ybz', 'watching': 'mdkmvayj', 'planting': 'qhgibfac', 'surfer': 'ymlnju', 'opening': 'ygktxxt', 'closing': 'qtrukrl', 'writing': 'lllijed', 'typing': 'eucbbl', 'basketball': 'xoixqhyxap', 'dirt': 'myoy', 'seasoning': 'xcvnbiaxo', 'potatoes': 'jjcudfbq', 'carrots': 'xoikmtr', 'hand': 'rcfi', 'feet': 'rvbe', 'laughing': 'zbtfycpo', 'spitting': 'rnkytkmn', 'clean': 'jfeyb', 'motionless': 'lnlwwdzivq'}

        for rule in rule_lst:
            rule = rule.split("-")
            if rule[0] in str1.split(" ") and rule[1] in str2.split(" "):
                #print(rule[0])
                #print(rule[1])
                new_str1 = str1.replace(rule[0], dax_word_dict[rule[0]])
                new_mid_sent = mid_sent.replace(rule[0], dax_word_dict[rule[0]])
                new_str2 = str2.replace(rule[1], dax_word_dict[rule[1]])
                self.change_count += 1

                if self.ver_noise_id == 2 and self.nat_noise_id == 2:
                    return new_str1, new_str2, new_mid_sent
                elif self.ver_noise_id == 1 and self.nat_noise_id == 1:
                    return str1, str2, mid_sent

        return "", "", ""

    def candidate_labels(self):
        self.primitive_pairs = [['positive_v', 'entailment_n'],
                                ['positive_v', 'neutral_n'],
                                ['positive_v', 'contradiction_n'],
                                ['neutral_v', 'entailment_n'],
                                ['neutral_v', 'neutral_n'],
                                ['neutral_v', 'contradiction_n'],
                                ['negative_v', 'entailment_n'],
                                ['negative_v', 'neutral_n'],
                                ['negative_v', 'contradiction_n']]
        self.primitive_pairs2idx = {" ".join(pair): idx for idx, pair in
                                           enumerate(self.primitive_pairs)}

        self.reason_pairs_dict = {"entailment":0, "neutral":1, "contradiction":2}
        self.ver_dict = {"positive":0, "neutral":1, "negative":2}
        self.nat_dict = {"entailment":0, "neutral":1, "contradiction":2}

    def label_noise_change(self, label, label_type, ver_noise_id, nat_noise_id):
        """change label to noise label"""
        ver_noise_dict1 = {"positive":"neutral", "neutral":"negative", "negative":"positive"}
        ver_noise_dict2 = {"positive":"negative", "neutral":"positive", "negative":"neutral"}
        nat_noise_dict1 = {"entailment":"neutral", "neutral":"contradiction", "contradiction":"entailment"}
        nat_noise_dict2 = {"entailment":"contradiction", "neutral":"entailment", "contradiction":"neutral"}

        if label_type == "ver":
            if ver_noise_id == 1:
                return ver_noise_dict1[label]
            elif ver_noise_id == 2:
                return ver_noise_dict2[label]
        elif label_type == "nat":
            if nat_noise_id == 1:
                return nat_noise_dict1[label]
            elif nat_noise_id == 2:
                return nat_noise_dict2[label]
        else:
            print("wrong label type")

    def reasoning_rules(self, ver_label, nat_label):
        """reasoning rules"""
        reasoning_dict = {"positive_entailment": "entailment", 
                        "positive_neutral": "neutral", 
                        "positive_contradiction": "contradiction",
                        "neutral_entailment": "neutral",
                        "neutral_neutral": "neutral",
                        "neutral_contradiction": "neutral",
                        "negative_entailment": "contradiction",
                        "negative_neutral": "neutral",
                        "negative_contradiction": "entailment"}
        return reasoning_dict[ver_label + "_" + nat_label]

    def __len__(self):
        return self.total_size

    def __getitem__(self, item):
        index = self.indexes[item]
        return [
            self.inputs[index],
            self.attention_masks[index],
            self.segments[index],
            torch.tensor(self.pair_labels[index]),
            torch.tensor(self.reason_labels[index]),
            self.ver_inputs[index],
            self.ver_attention_masks[index],
            self.ver_segments[index],
            self.nat_inputs[index],
            self.nat_attention_masks[index],
            self.nat_segments[index],
            self.ver_labels[index],
            self.nat_labels[index],
        ]


