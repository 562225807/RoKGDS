#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

from collections import defaultdict
import argparse
import logging
import pickle
import json
from tqdm import tqdm, trange
import re
import scipy
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from utils.encoder_decoder_framework_duplicate import TwoStageModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BertTokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (TwoStageModel, BertTokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


class TextDataset(Dataset):
    def __init__(self, file_path: str, args, tokenizer, block_size=380):
        self.p = re.compile(r'([\d ,' ']+ - \d{1,2} - \d{1,2})')
        self.bucket_size = [17, 33, 65, 129, 512]
        self.bucket_max = [33, 17, 13, 7, 1]
        self.tokenizer = tokenizer
        logger.info("Prepare Data...")
        cached_features_file = os.path.join(file_path, "data.pkl")
        if os.path.exists(cached_features_file):
            self.histories, self.segments, self.knowledges, self.know_segments = pickle.load(
                open(cached_features_file, "rb"))
        else:
            self.histories = []
            self.segments = []
            self.knowledges = []
            self.know_segments = []

            with open(os.path.join(file_path, "samples_rebuild_test.txt"), encoding='utf-8') as f:
                x = f.readlines()

            for i, sample in enumerate(x):
                sample = json.loads(sample, encoding='utf-8')
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
                    sen_ids = [_ + 10] + self.sentence2id(sample["history"][_], is_last=int(_ == len(sample["history"]) - 1))
                    if len(sen_ids) > 119:
                        sen_ids = sen_ids[:119] + sen_ids[-1:]
                    segment.append([1 + (sample["Bot"] + _) % 2] * len(sen_ids))
                    history.append(sen_ids)

                max_his = 0
                win_length = 0
                for his in history[::-1]:
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

                win_length += 1
                history.append([len(sample["history"]) + 10])
                segment.append([2])

                self.histories.append(history[len(history) - win_length:])
                self.segments.append(segment[len(history) - win_length:])
                self.knowledges.append(knowledge)
                self.know_segments.append(know_segments)

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


def collate(example):
    example = np.array(example)
    histories = pad_history(example[:, 0])
    knowledge = pad_knowledge(example[:, 1])
    segments = pad_history(example[:, 2])
    knowledge_segments = pad_knowledge(example[:, 3])

    return histories.tolist(), knowledge, segments.tolist(), knowledge_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--output_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument(
        "--do_sample", action="store_true", help=""
    )
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)


    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    eval_dataset = TextDataset(file_path=args.input_file, args=args, tokenizer=tokenizer, block_size=args.block_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    output_sequences = []
    hiss = []
    for batch in tqdm(eval_dataloader, desc="Testing"):
        histories, knowledges, segments, knowledge_segments = batch
        for _, his in enumerate(histories):
            histories[_] = np.concatenate(histories[_], 0).tolist()
            segments[_] = np.concatenate(segments[_], 0).tolist()

        histories = torch.LongTensor(histories).to(args.device)
        knowledges = [torch.LongTensor(k_inst).to(args.device) for k_inst in knowledges]
        segments = torch.LongTensor(segments).to(args.device)
        knowledge_segments = [torch.LongTensor(ks_inst).to(args.device) for ks_inst in knowledge_segments]

        exist_length = torch.sum(histories != 0, 1)
        hiss.extend(histories.tolist())
        o_s = model.generate(
            input_ids=histories,
            max_length=args.length + len(histories[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=0,
            eos_token_ids=[101],
            knowledges=knowledges,
            segments=segments,
            knowledge_segments=knowledge_segments,
            exist_length=exist_length,
        )
        output_sequences.extend(o_s)

    generated_sequences = []
    ori_sequences = []
    ave_lenth = 0
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        hiss[generated_sequence_idx] = hiss[generated_sequence_idx] + [101]
        generated_sequence = generated_sequence.tolist() + [101]

        hiss[generated_sequence_idx] = hiss[generated_sequence_idx][:hiss[generated_sequence_idx].index(101)]
        generated_sequence = generated_sequence[:generated_sequence.index(101)]

        hiss[generated_sequence_idx] = tokenizer.decode(hiss[generated_sequence_idx]).split()
        generated_sequence = tokenizer.decode(generated_sequence).split()

        # Decode text
        message = ''.join(hiss[generated_sequence_idx])
        response = ''.join(generated_sequence)

        new_response = ""
        for i in range(len(response)):
            if response[i] != '\n': new_response += response[i]
            else: new_response += ' '

        new_response = new_response.strip()

        ave_lenth += len(new_response)
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
                message + '\n' + new_response + '\n\n'
        )

        generated_sequences.append(f"{new_response}\n")
        ori_sequences.append(total_sequence)

    ave_lenth /= len(output_sequences)
    logger.info(f"Average length: {ave_lenth}")

    with open(args.output_file, "w", encoding='utf-8') as f:
        f.writelines(generated_sequences)

    with open(args.output_file + '.ori', "w", encoding='utf-8') as f:
        f.writelines(ori_sequences)

    return generated_sequences


if __name__ == "__main__":
    main()
