# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import scipy
from typing import Dict, List, Tuple
from torch.nn import functional as F

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils.encoder_decoder_framework_duplicate import TwoStageModel
import json

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,

)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (BertConfig, TwoStageModel, BertTokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, file_path: str, args, tokenizer, block_size=512):
        self.p = re.compile(r'([\d ,' ']+ - \d{1,2} - \d{1,2})')
        self.bucket_size = [17, 33, 65, 129, 512]
        self.bucket_max = [33, 17, 13, 7, 1]
        self.tokenizer = tokenizer
        logger.info("Prepare Data...")
        cached_features_file = os.path.join(file_path, "data_" + args.stage + ('_noise' if args.is_masking else '') + '.pkl')
        if os.path.exists(cached_features_file):
            self.histories, self.segments, self.knowledges, self.know_segments = pickle.load(
                open(cached_features_file, "rb"))
        else:
            self.histories = []
            self.segments = []
            self.knowledges = []
            self.know_segments = []

            max_len = 0
            with open(os.path.join(file_path, "samples_" + args.stage + ".txt"), encoding='utf-8') as f:
                x = f.readlines()

            for i, sample in enumerate(x):
                sample = json.loads(sample, encoding='utf-8')
                sample = self.masking(sample, args.is_masking)
                sample['knowledge'] = [' '.join(k) for k in sample['knowledge']]
                sample['knowledge'] = [sample['goal']] + sample['knowledge'] + [sample['history'][0]]

                tmp_goal = self.sentence2id(self.my_extra(sample['goal']))
                history = [self.sentence2id(sample['history'][0])]
                segment = [[0] * len(history[0])]
                sample['history'] = sample['history'][1:]
                for _ in range(len(sample["history"])):
                    # if _ % 2 == 1 - sample["Bot"]:
                    #     sen_ids = [_ + 10] + self.sentence2id(sample["history"][_], is_last=int(_ == len(sample["history"])-1))
                    #     if len(sen_ids) > 119:
                    #         sen_ids = sen_ids[:119] + sen_ids[-1:]
                    #     segment.append([2 - sample["Bot"]] * len(sen_ids))
                    # else:
                    sen_ids = [_ + 10] + self.sentence2id(sample["history"][_], is_last=int(_ == len(sample["history"])-1))
                    if len(sen_ids) > 119:
                        sen_ids = sen_ids[:119] + sen_ids[-1:]
                    segment.append([1 + (sample["Bot"] + _) % 2] * len(sen_ids))
                    history.append(sen_ids)

                max_his = 0
                win_length = 1
                for his in history[::-1][1:]:
                    if len(his) + max_his < block_size:
                        max_his += len(his)
                        win_length += 1
                    else:
                        break

                for _, k in enumerate(sample['knowledge']):
                    if _ != 0:
                        sample['knowledge'][_] = self.sentence2id(''.join(k.split())[:128]) ##bug
                    else:
                        sample['knowledge'][_] = self.sentence2id(k)

                knowledge, know_segments = self.dividing_bucket(sample['knowledge'])

                history[len(history) - win_length] = tmp_goal + [6] + history[len(history) - win_length]
                segment[len(history) - win_length] = [3] * len(tmp_goal) + [0] + segment[len(history) - win_length]

                max_len = max(max_len, sum([len(his) for his in history[len(history) - win_length:]]))

                self.histories.append(history[len(history) - win_length:])
                self.segments.append(segment[len(history) - win_length:])
                self.knowledges.append(knowledge)
                self.know_segments.append(know_segments)

            logger.info(f"max_len: {max_len}")
            pickle.dump((self.histories, self.segments, self.knowledges, self.know_segments), open(cached_features_file, "wb"))

        logger.info(str(len(self.histories)))

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, item):
        return self.histories[item], self.knowledges[item], \
               self.segments[item], self.know_segments[item]

    def sentence2id(self, sentence, is_start=0, is_last=0):
        sentence = self.tokenizer.tokenize(sentence)[:511]
        sentence = ["[START]"] * is_start + sentence + (1-is_last) * ["[SEP]"] + is_last * ["[CLS]"]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sentence)
        return indexed_tokens

    def dividing_bucket(self, knowledge):
        tmp_kg = [[], [], [], [], []]
        tmp_sg = [[], [], [], [], []]
        for i, k in enumerate(knowledge):
            for _, s in enumerate(self.bucket_size):
                if len(k) <= self.bucket_size[_]:
                    k = [7 + min(i, 2)] + k
                    tmp_kg[_].append(k)
                    tmp_sg[_].append([3 + min(i, 2)] * len(k))
                    break
        # for _, i in enumerate(tmp_kg):
        #     if len(i) >self.bucket_max[_]:
        #         print([len(x) for x in tmp_kg])
        #         break
        return tmp_kg, tmp_sg

    def delete_blank(self, sentence):
        tmp = ""
        last = False
        for i in sentence.split():
            now = all(ord(c) < 128 for c in i)
            if last and now:
                tmp += ' '
            last = now
            tmp += i
        return tmp

    def masking(self, sample, is_masking):
        sample['goal'] = self.delete_blank(sample['goal'])
        for i in range(len(sample['knowledge'])):
            for j in range(len(sample['knowledge'][i])):
                sample['knowledge'][i][j] = self.delete_blank(sample['knowledge'][i][j])
        for i in range(len(sample['history'])):
            sample['history'][i] = self.delete_blank(sample['history'][i])

        if is_masking:
            have_mask = set()
            for i, k in enumerate(sample['knowledge']):
                masking = np.random.randint(0, 2)
                if masking == 0:
                    # mask head
                    if len(k[0]) >= 3 and len(k[0]) <= 8 and k[0] not in have_mask:
                        p = np.random.randint(0, 3, len(k[0]))
                        noise_label = np.random.randint(700, 13317, len(k[0]))
                        tmp_k = ""
                        ori = k[0]
                        for j in range(len(k[0])):
                            if p[j] == 0:
                                tmp_k += self.tokenizer._convert_id_to_token(noise_label[j])
                            else:
                                tmp_k += k[0][j]

                        have_mask.add(tmp_k)

                        sample['goal'] = sample['goal'].replace(ori, tmp_k)
                        for i in range(len(sample['knowledge'])):
                            sample['knowledge'][i][0] = sample['knowledge'][i][0].replace(ori, tmp_k)
                            sample['knowledge'][i][-1] = sample['knowledge'][i][-1].replace(ori, tmp_k)
                        for i in range(len(sample['history'])):
                            sample['history'][i] = sample['history'][i].replace(ori, tmp_k)

                    # mask tail
                    if len(k[-1]) >= 3 and len(k[-1]) <= 8 and k[-1] not in have_mask:
                        p = np.random.randint(0, 3, len(k[-1]))
                        noise_label = np.random.randint(700, 13317, len(k[-1]))
                        tmp_k = ""
                        ori = k[-1]
                        for j in range(len(k[-1])):
                            if p[j] == 0:
                                tmp_k += self.tokenizer._convert_id_to_token(noise_label[j])
                            else:
                                tmp_k += k[-1][j]

                        have_mask.add(tmp_k)

                        sample['goal'] = sample['goal'].replace(k[-1], tmp_k)
                        for i in range(len(sample['knowledge'])):
                            sample['knowledge'][i][0] = sample['knowledge'][i][0].replace(ori, tmp_k)
                            sample['knowledge'][i][-1] = sample['knowledge'][i][-1].replace(ori, tmp_k)
                        for i in range(len(sample['history'])):
                            sample['history'][i] = sample['history'][i].replace(ori, tmp_k)

        return sample

    def extra(self, goals):
        goals = goals.split('[')
        new_goal = ''
        for goal in goals:
            goal = goal + '['
            intent_pattern = "^\[\d\](.*?)(\(.*\))?$"
            intent_match = re.match(intent_pattern, goal)
            if intent_match:
                intent = intent_match.group(1)
                new_goal += ' ' + intent + ' '
        return ' '.join(new_goal.split())

    def my_extra(self, goals):
        goals = goals.split('[')
        new_goal = ''
        for goal in goals:
            if goal == '':
                continue
            goal = '[' + goal
            goal = goal.strip()
            new_goal += ' ' + goal.split('(')[0].strip() + ' '
        return ''.join(new_goal.split())


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return TextDataset(file_path=file_path, args=args, tokenizer=tokenizer, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def pad_data(insts, pad_len, pad_num=-1, pad_id=0):
    """ padding ids """
    insts_pad = []
    if pad_num != -1:
        for inst in insts:
            inst_pad = inst + [pad_id] * (pad_len - len(inst))
            insts_pad.append(inst_pad)
        if len(insts_pad) < pad_num:
            insts_pad += [[pad_id] * pad_len] * (pad_num - len(insts_pad))
    else:
        insts_pad = insts + [pad_id] * (pad_len - len(insts))
    return insts_pad


def cal_max_len(ids):
    """ calculate max sequence length """
    if isinstance(ids[0], list):
        pad_len = max([cal_max_len(k) for k in ids])
    else:
        pad_len = len(ids)
    return pad_len


def pad_history(history, pad_id=0):
    h_length = [sum([len(sentence) for sentence in h_inst]) for h_inst in history]
    max_length = max(h_length)
    for i in range(len(h_length)):
        history[i][-1] += (max_length - h_length[i]) * [pad_id]
    return history


def pad_knowledge(knowledges, pad_id=0):
    new_knowledges = []
    bucket = len(knowledges[0])
    for bucket in range(bucket):
        max_length = 0
        max_number = 0
        for batch in knowledges:
            if len(batch[bucket]) == 0: continue
            max_length = max(max_length, max([len(sentence) for sentence in batch[bucket]]))
            max_number = max(max_number, len(batch[bucket]))
        if max_length == 0: continue
        tmp = []
        for i in range(len(knowledges)):
            knowledges[i][bucket] = pad_data(knowledges[i][bucket], max_length, max_number)
            tmp.append(knowledges[i][bucket])
        new_knowledges.append(tmp)
    return new_knowledges


def sop(histories, segments):
    sop_labels = []
    for _, his in enumerate(histories):
        if np.random.randint(2) == 1 and len(his) - 3 > 0:
            pos = np.random.randint(1, len(his)-2)
            histories[_][pos], histories[_][pos+1] = histories[_][pos+1], histories[_][pos]
            segments[_][pos], segments[_][pos+1] = segments[_][pos+1], segments[_][pos]
            sop_labels.append(1)
        else:
            sop_labels.append(0)
    return sop_labels


def cls_three_way(histories, segments, data):
    cls_label = []
    for _ in range(len(histories)):
        r = np.random.randint(4)
        if r <= 1 or len(histories[_]) < 2:
            cls_label.append(0)
        elif r == 2:
            cls_label.append(1)
            rdata = np.random.randint(len(data))
            pos = np.random.randint(len(data[rdata][0]))
            histories[_][-1] = data[rdata][0][pos][:-1] + [101]
            segments[_][-1] = [2] * len(histories[_][-1])
        elif r == 3:
            cls_label.append(2)
            pos = np.random.randint(len(histories[_])-1)
            histories[_][-1] = histories[_][pos][:-1] + [101]
            segments[_][-1] = [2] * len(histories[_][-1])
        while True:
            if sum([len(sentence) for sentence in histories[_]]) > 512:
                histories[_] = histories[_][1:]
                segments[_] = segments[_][1:]
            else: break
    return cls_label


def collate(example):
    example = np.array(example)
    histories = pad_history(example[:, 0])
    knowledge = pad_knowledge(example[:, 1])
    segments = pad_history(example[:, 2])
    knowledge_segments = pad_knowledge(example[:, 3])

    return histories.tolist(), knowledge, segments.tolist(), knowledge_segments


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()

            histories, knowledges, segments, knowledge_segments = batch

            sop_labels = None
            cls_token_location = None
            if args.use_sop:
                if args.stage == 'sequence': sop_labels = torch.LongTensor(sop(histories, segments)).to(args.device)
                else:
                    sop_labels = torch.LongTensor(cls_three_way(histories, segments, train_dataset)).to(args.device)
                    histories = pad_history(histories)
                    segments = pad_history(segments)

            lm_labels = []
            for _, his in enumerate(histories):
                if args.only_response:
                    lm_labels.append([-100] * (sum([len(sentence) for sentence in his]) - len(his[-1])) + his[-1])
                else:
                    tmp = []
                    for zzz, sentence in enumerate(his):
                        if zzz == 0:
                            tmp += [-100] * len(sentence)
                        else:
                            tmp += sentence
                    lm_labels.append(tmp)

                histories[_] = np.concatenate(histories[_], 0).tolist()
                segments[_] = np.concatenate(segments[_], 0).tolist()

            if args.use_sop:
                cls_token_location = torch.LongTensor([tokens.index(tokenizer.cls_token_id) for tokens in histories]).to(args.device)

            lm_labels = torch.LongTensor(lm_labels).to(args.device)
            histories = torch.LongTensor(histories).to(args.device)
            knowledges = [torch.LongTensor(k_inst).to(args.device) for k_inst in knowledges]
            segments = torch.LongTensor(segments).to(args.device)
            knowledge_segments = [torch.LongTensor(ks_inst).to(args.device) for ks_inst in knowledge_segments]

            # logger.info(f"\nhis_len: {histories.shape}, res_len: {responses.shape}, "
            #             f"kn_len: {knowledges.shape}, com_len: {comments.shape}")
            lm_labels[lm_labels == 0] = -100

            outputs = model(input_ids=(histories, knowledges),
                            lm_labels=lm_labels, token_type_ids=(segments, knowledge_segments),
                            mc_token_ids=cls_token_location, mc_labels=sop_labels, use_copy=args.use_copy, label_smoothing=args.label_smoothing)
            if args.use_sop:
                loss = outputs[0] + 0.5 * outputs[1]
            else:
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        # logger.info(f"c_loss: {c_loss}")
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    sop_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        histories, knowledges, segments, knowledge_segments = batch

        sop_labels = None
        cls_token_location = None
        if args.use_sop:
            if args.stage == 'sequence': sop_labels = torch.LongTensor(sop(histories, segments)).to(args.device)
            else:
                sop_labels = torch.LongTensor(cls_three_way(histories, segments, eval_dataset)).to(args.device)
                histories = pad_history(histories)
                segments = pad_history(segments)

        lm_labels = []
        for _, his in enumerate(histories):
            if args.only_response:
                lm_labels.append([-100] * (sum([len(sentence) for sentence in his]) - len(his[-1])) + his[-1])
            else:
                tmp = []
                for zzz, sentence in enumerate(his):
                    if zzz == 0:
                        tmp += [-100] * len(sentence)
                    else:
                        tmp += sentence
                lm_labels.append(tmp)
            histories[_] = np.concatenate(histories[_], 0).tolist()
            segments[_] = np.concatenate(segments[_], 0).tolist()

        if args.use_sop:
            cls_token_location = torch.LongTensor([tokens.index(tokenizer.cls_token_id) for tokens in histories]).to(args.device)

        lm_labels = torch.LongTensor(lm_labels).to(args.device)
        histories = torch.LongTensor(histories).to(args.device)
        knowledges = [torch.LongTensor(k_inst).to(args.device) for k_inst in knowledges]
        segments = torch.LongTensor(segments).to(args.device)
        knowledge_segments = [torch.LongTensor(ks_inst).to(args.device) for ks_inst in knowledge_segments]

        # logger.info(f"\nhis_len: {histories.shape}, res_len: {responses.shape}, "
        #             f"kn_len: {knowledges.shape}, com_len: {comments.shape}")
        lm_labels[lm_labels == 0] = -100

        with torch.no_grad():
            outputs = model(input_ids=(histories, knowledges),
                            lm_labels=lm_labels, token_type_ids=(segments, knowledge_segments),
                            mc_token_ids=cls_token_location, mc_labels=sop_labels, use_copy=args.use_copy, label_smoothing=0.1)
            loss = outputs[0]

            eval_loss += loss.mean().item()
            if args.use_sop:
                sop_loss += outputs[1].mean().item()
            # kl_loss += outputs[0].mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    sop_loss = sop_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "SOP loss": sop_loss}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--stage", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--only_response",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--use_sop",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--use_copy",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--fix_para", action="store_true", help="fix parameter.")
    parser.add_argument("--is_masking", action="store_true", help="add noise to knowledge")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--use_bow", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="label smoothing probability")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    config.stage = args.stage

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )
    tokenizer.add_special_tokens({'additional_special_tokens': ['<name>', '<high>', '<weight>', '<constell>', '<blood>']})
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    logger.info(config)
    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)
    for p in model.parameters():
        p.requires_grad = not args.fix_para

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, config=config)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint, config=config).to(args.device)

            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
