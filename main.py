import torch
import random
import numpy as np
import os
import time
import logging
from parser import get_argparse
from copy import deepcopy
from models.model_roberta import RoBertaMTSep
#from models.model_roberta_continual import RoBertaMTSep
from tqdm.auto import tqdm, trange
from transformers import RobertaTokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from dataset.dataset_cl import ComposNLIDataset, ComposNLIMTDataset, ComposNLIMTCLDataset
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cosine_similarity

from continual.er import ExperienceReplay, ExperienceReplayBuffer, ExperienceReplayMIR
from continual.agem import AGEM
from continual.naive import NaiveWrapper
from continual.kd import KD

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # output to handler
chlr.setFormatter(formatter)
logfile = './log/test_{}.txt'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))
fh = logging.FileHandler(logfile)
fh.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fh)

PREFIX_CHECKPOINT_DIR = "checkpoint"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, args, train_data, val_data, test_data, middle_evaluation_num):
    # data loader
    if args.do_continual:
        sampler = SequentialSampler(train_data)
        order = args.train_file_name.split("/")[0][8:] # order_format = y_order_aa_bb_cc_dd
        order_lst = order.split("_")
    else:
        sampler = RandomSampler(train_data)
    data_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=sampler
    )
    check_step_num = int(middle_evaluation_num / args.batch_size)

    # prepare optimizer and scheduler
    train_steps = int(len(data_loader) * args.train_epoch_num)
    params = [p for n, p in model.named_parameters()]
    for n, p in model.named_parameters():
        if "roberta" not in n:
            print(n)
    optimizer = AdamW(params, lr=args.learning_rate)

    # prepare continual learning strategy
    if args.cl_strategy == "er":
        cl_strategy = ExperienceReplay(model, optimizer, args)
    elif args.cl_strategy == "er-buff":
        cl_strategy = ExperienceReplayBuffer(model, train_data, args.task_num, optimizer, args)
    elif args.cl_strategy == "er-mir":
        cl_strategy = ExperienceReplayMIR(model, optimizer, args)
    elif args.cl_strategy == "agem":
        cl_strategy = AGEM(model, optimizer, args)
    elif args.cl_strategy == "kd":
        cl_strategy = KD(model, optimizer, args)
    elif args.cl_strategy == "naive":
        cl_strategy = NaiveWrapper(model, optimizer, args)

    # train
    global_step = 0
    avg_loss = 0.0
    all_primitve_acc = []
    all_reason_acc = []
    train_iterator = trange(0, args.train_epoch_num, desc="epoch")
    for epoch in train_iterator:
        # train
        # epoch_iterator = tqdm(data_loader, desc="iteration")
        epoch_iterator = data_loader
        model.train()
        for step, batch in enumerate(epoch_iterator):
            # batch data: input_ids, attention_mask, segment_ids, labels
            new_batch = (batch[0], batch[1], batch[2], batch[3], batch[4])
            batch = tuple(t.to(args.device) for t in new_batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "primitive_labels": batch[3], "reason_labels": batch[4], "mode": "train"}

            if args.cl_strategy == "kd":
                if (step + 1) % check_step_num == 0:
                    updata_cache = True
                else:
                    updata_cache = False
                if (step + 1) > check_step_num:
                    task_follow = True
                else:
                    task_follow = False
                loss, primitive_acc, reason_acc = cl_strategy.train_(inputs, batch, task_follow, updata_cache)
            else:
                loss, primitive_acc, reason_acc = cl_strategy.train_(inputs, batch)
            avg_loss += loss.item()
            global_step += 1

            # log
            logger.info("loss = %.6f", loss.item())
            logger.info("avg_loss = %.6f", avg_loss / global_step)
            logger.info("primitive_acc = %.6f", primitive_acc)
            logger.info("reason_acc = %.6f", reason_acc)

            all_primitve_acc.append(primitive_acc)
            all_reason_acc.append(reason_acc)

            # test
            if not args.do_shuffle:       
                if (step + 1) % check_step_num == 0:
                    evaluate_middle(model, args, val_data, order_lst)

        logger.info("all_primitive_acc = %.6f", np.sum(all_primitve_acc) / len(all_primitve_acc))
        logger.info("all_reason_acc = %.6f", np.sum(all_reason_acc) / len(all_reason_acc))

        evaluate(model, args, val_data, "val")
        evaluate(model, args, test_data, "test")


def evaluate_middle(model, args, test_data, order_lst):
    """evaluate middle phase"""
    # data loader
    sampler = SequentialSampler(test_data)
    data_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    primitive_acc = []
    reason_acc = []

    phase1_primitive_acc = []
    phase1_reason_acc = []
    phase2_primitive_acc = []
    phase2_reason_acc = []
    phase3_primitive_acc = []
    phase3_reason_acc = []
    phase4_primitive_acc = []
    phase4_reason_acc = []

    # for batch in tqdm(data_loader):
    model.eval()
    for batch in data_loader:
        new_batch = (batch[0], batch[1], batch[2], batch[3], batch[4])
        batch = tuple(t.to(args.device) for t in new_batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                  "token_type_ids": batch[2], "mode": "test"}

        with torch.no_grad():
            _, primitive_logits, reason_logits, _ = model(**inputs)

        primitive_logits = torch.softmax(primitive_logits, dim=1)
        primitive_predictions = torch.argmax(primitive_logits, dim=1).detach().cpu().numpy()
        primitive_labels = batch[3].detach().cpu().numpy()
        primitive_acc += list(np.equal(primitive_predictions, primitive_labels))

        reason_logits = torch.softmax(reason_logits, dim=1)
        reason_predictions = torch.argmax(reason_logits, dim=1).detach().cpu().numpy()
        reason_labels = batch[4].detach().cpu().numpy()
        reason_acc += list(np.equal(reason_predictions, reason_labels))

        # separate phase
        current_primitive_acc = list(np.equal(primitive_predictions, primitive_labels))
        current_reason_acc = list(np.equal(reason_predictions, reason_labels))
        for index in range(len(primitive_labels)):
            primitive_label = str(primitive_labels[index] + 1)  # match the order
            if primitive_label in order_lst[0]:
                phase1_primitive_acc.append(current_primitive_acc[index])
                phase1_reason_acc.append(current_reason_acc[index])
            elif primitive_label in order_lst[1]:
                phase2_primitive_acc.append(current_primitive_acc[index])
                phase2_reason_acc.append(current_reason_acc[index])
            elif primitive_label in order_lst[2]:
                phase3_primitive_acc.append(current_primitive_acc[index])
                phase3_reason_acc.append(current_reason_acc[index])
            elif primitive_label in order_lst[3]:
                phase4_primitive_acc.append(current_primitive_acc[index])
                phase4_reason_acc.append(current_reason_acc[index])
            else:
                print("error")

    accuracy = np.sum(primitive_acc) / len(primitive_acc)
    reason_accuracy = np.sum(reason_acc) / len(reason_acc)
    logger.info("middle_accuracy = %.6f", accuracy)
    logger.info("middle_reason_accuracy = %.6f", reason_accuracy)

    phase1_accuracy = np.sum(phase1_primitive_acc) / len(phase1_primitive_acc)
    phase1_reason_accuracy = np.sum(phase1_reason_acc) / len(phase1_reason_acc)
    logger.info("phase1_accuracy = %.6f", phase1_accuracy)
    logger.info("phase1_reason_accuracy = %.6f", phase1_reason_accuracy)

    phase2_accuracy = np.sum(phase2_primitive_acc) / len(phase2_primitive_acc)
    phase2_reason_accuracy = np.sum(phase2_reason_acc) / len(phase2_reason_acc)
    logger.info("phase2_accuracy = %.6f", phase2_accuracy)
    logger.info("phase2_reason_accuracy = %.6f", phase2_reason_accuracy)

    phase3_accuracy = np.sum(phase3_primitive_acc) / len(phase3_primitive_acc)
    phase3_reason_accuracy = np.sum(phase3_reason_acc) / len(phase3_reason_acc)
    logger.info("phase3_accuracy = %.6f", phase3_accuracy)
    logger.info("phase3_reason_accuracy = %.6f", phase3_reason_accuracy)

    phase4_accuracy = np.sum(phase4_primitive_acc) / len(phase4_primitive_acc)
    phase4_reason_accuracy = np.sum(phase4_reason_acc) / len(phase4_reason_acc)
    logger.info("phase4_accuracy = %.6f", phase4_accuracy)
    logger.info("phase4_reason_accuracy = %.6f", phase4_reason_accuracy)


def evaluate(model, args, test_data, mode):
    # data loader
    sampler = SequentialSampler(test_data)
    data_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    total_acc = []
    veridical_acc = []
    sick_acc = []
    total_prob = []
    reason_acc = []

    # for batch in tqdm(data_loader):
    model.eval()
    for batch in data_loader:
        new_batch = (batch[0], batch[1], batch[2], batch[3], batch[4])
        batch = tuple(t.to(args.device) for t in new_batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "mode": "test"}

        with torch.no_grad():
            _, primitive_logits, reason_logits, _ = model(**inputs)

        def primitive_labels_search(predictions, labels, primitive_pairs):
            primitive_veridical = []
            primitive_sick = []
            for i in range(len(predictions)):
                predict_rep = primitive_pairs[predictions[i]]
                label_rep = primitive_pairs[labels[i]]
                if predict_rep[0] == label_rep[0]:
                    primitive_veridical.append(1)
                else:
                    primitive_veridical.append(0)
                if predict_rep[1] == label_rep[1]:
                    primitive_sick.append(1)
                else:
                    primitive_sick.append(0)
            return primitive_veridical, primitive_sick

        primitive_logits = torch.softmax(primitive_logits, dim=1)
        primitive_predictions = torch.argmax(primitive_logits, dim=1).detach().cpu().numpy()
        primitive_predictions_prob = torch.max(primitive_logits, dim=1).values.detach().cpu().numpy()
        primitive_labels = batch[3].detach().cpu().numpy()
        primitive_veridical, primitive_sick = primitive_labels_search(primitive_predictions, primitive_labels,
                                                                      test_data.primitive_pairs)
        total_acc += list(np.equal(primitive_predictions, primitive_labels))
        veridical_acc += primitive_veridical
        sick_acc += primitive_sick
        total_prob += list(primitive_predictions_prob)

        reason_logits = torch.softmax(reason_logits, dim=1)
        reason_predictions = torch.argmax(reason_logits, dim=1).detach().cpu().numpy()
        reason_labels = batch[4].detach().cpu().numpy()
        reason_acc += list(np.equal(reason_predictions, reason_labels))


    accuracy = np.sum(total_acc) / len(total_acc)
    veridical_accuracy = np.sum(veridical_acc) / len(veridical_acc)
    sick_accuracy = np.sum(sick_acc) / len(sick_acc)
    prob = np.sum(total_prob) / len(total_prob)
    reason_accuracy = np.sum(reason_acc) / len(reason_acc)

    if mode == "test":
        logger.info("test_veridical_accuracy = %.6f", veridical_accuracy)
        logger.info("test_sick_accuracy = %.6f", sick_accuracy)
        logger.info("test_accuracy = %.6f", accuracy)
        logger.info("test_prob = %.6f", prob)
        logger.info("test_reason_accuracy = %.6f", reason_accuracy)
    else:
        logger.info("val_veridical_accuracy = %.6f", veridical_accuracy)
        logger.info("val_sick_accuracy = %.6f", sick_accuracy)
        logger.info("val_accuracy = %.6f", accuracy)
        logger.info("val_prob = %.6f", prob)
        logger.info("val_reason_accuracy = %.6f", reason_accuracy)


def main():
    args = get_argparse().parse_args()
    args.no_cuda = not torch.cuda.is_available()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device

    set_seed(args.seed)

    print("--batch size=", args.batch_size, '--learning_rate=', args.learning_rate,
          "--max_input_num=", args.max_input_num, "--data_dir", args.data_dir,
          "--train_file_name", args.train_file_name, "--test_file_name", args.test_file_name,
          "--dropout", args.dropout, "max_grad_norm", args.max_grad_norm,
          "--do_shuffle", args.do_shuffle, "--seed", args.seed, "--loss_ratio", args.loss_ratio,
          "--cl_strategy", args.cl_strategy, "--do_continual", args.do_continual,
          "--memory_size", args.memory_size, "--task_num_buff", args.task_num,
          "--model_type", args.model_type)

    # prepare data
    model_type = args.model_type
    tokenizer = RobertaTokenizer.from_pretrained(model_type)
    dataset_params = {
        'tokenizer': tokenizer,
        'max_seq_len': args.max_input_num,
        'data_dir': args.data_dir,
        'seed': args.seed,
        'multiple_round_num': args.multiple_round_num,
    }

    train_data = ComposNLIMTCLDataset(args.data_dir, dataset_params, args.train_file_name,
                                     do_shuffle=args.do_shuffle, do_multiple_round=args.do_multiple_round,
                                      do_incre_train=args.do_incre_train, do_continual=args.do_continual, do_noise=args.do_noise,
                                      mode="train")
    val_data = ComposNLIMTCLDataset(args.data_dir, dataset_params, args.val_file_name, mode="val")
    test_data = ComposNLIMTCLDataset(args.data_dir, dataset_params, args.test_file_name, mode="test")

    logger.info("train_data = %d", len(train_data))
    logger.info("val_data = %d", len(val_data))
    logger.info("test_data = %d", len(test_data))

    model = RoBertaMTSep(model_type, primitive_class=train_data.primitve_label_size,
                      reason_class=train_data.reason_label_size, loss_ratio=args.loss_ratio,
                      dropout=args.dropout)

    if not args.no_cuda:
        model = model.cuda()

    # train/test
    if args.do_train:
        middle_evaluation_num = len(train_data) / args.task_num
        train(model, args, train_data, val_data, test_data, middle_evaluation_num)


if __name__ == '__main__':
    main()

