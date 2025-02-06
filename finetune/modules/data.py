import logging
import os
import random
from typing import Dict, List

import pandas as pd
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from .arguments import DataArguments, ModelArguments
from .prompt_en import all_prompt_en, sft_prompt_en
from .utils import load_json, load_pickle

logger = logging.getLogger(__name__)


def load_prompt():
    return all_prompt_en, sft_prompt_en


def load_vocab(data_args: DataArguments):
    user_vocab = load_pickle(
        os.path.join('../data', data_args.dataset, "vocab/user_vocab.pkl"))
    item_vocab = load_pickle(
        os.path.join('../data', data_args.dataset, "vocab/item_vocab.pkl"))

    query_vocab = load_pickle(
        os.path.join('../data', data_args.dataset, "vocab/query_vocab.pkl"))

    src_session_vocab: Dict = load_pickle(
        os.path.join('../data', data_args.dataset,
                     "vocab/src_session_vocab.pkl"))

    if 'query' not in list(src_session_vocab.values())[0].keys():
        for key in src_session_vocab.keys():
            src_session_vocab[key]['query'] = query_vocab[
                src_session_vocab[key]['query_id']]['query']

    vocab_dict = {
        "user_vocab": user_vocab,
        "item_vocab": item_vocab,
        "query_vocab": query_vocab,
        "src_session_vocab": src_session_vocab,
    }

    semantic_indices: Dict = load_json(
        os.path.join('../data', data_args.dataset,
                     f"emb/{data_args.semantic_index_file}"))
    collaborate_indices: Dict = load_json(
        os.path.join('../data', data_args.dataset,
                     f"emb/{data_args.collaborate_index_file}"))
    vocab_dict['semantic_indices'] = semantic_indices
    vocab_dict['collaborate_indices'] = collaborate_indices

    return vocab_dict


def get_dataset_class(task):
    task = task.strip()
    if task == "SeqRecWithSrcHis":
        return SeqRecWithSrcHisDataset
    elif task == "PerSrcWithRecHis":
        return PerSrcWithRecHisDataset
    elif task == "QGenWithRecHis":
        return QueryGenWithRecHisDataset
    elif task == "AdHocSrc_Doc2Query":
        return AdHocSrcDataset_Doc2Query
    elif task == "Item2Index":
        return Item2IndexDataset
    elif task == "Index2Item":
        return Index2ItemDataset
    elif task == 'Doc2Query':
        return Doc2QueryDataSet
    else:
        raise ValueError(f"{task} Error")


def load_dataset(data_args: DataArguments, model_args: ModelArguments,
                 tokenizer: PreTrainedTokenizer, mode):

    all_prompt, sft_prompt = load_prompt()

    if mode == 'train':
        logger.info("sft_prompt: {}".format(sft_prompt))
        for task, prompts in all_prompt.items():
            logger.info("task: {} num_prompts: {} prompt_sample: {}".format(
                task, len(prompts), prompts[0]))

    all_vocabs = load_vocab(data_args)

    if mode == 'train':
        train_tasks = data_args.train_tasks.split(",")
        train_prompt_sample_num = [
            int(_) for _ in data_args.train_prompt_sample_num.split(",")
        ]
        assert len(train_tasks) == len(
            train_prompt_sample_num
        ), "prompt sample number does not match task number"
        train_data_sample_num = [
            int(_) for _ in data_args.train_data_sample_num.split(",")
        ]
        assert len(train_tasks) == len(
            train_data_sample_num
        ), "data sample number does not match task number"

        train_datasets = []
        for task, prompt_sample_num, data_sample_num in zip(
                train_tasks, train_prompt_sample_num, train_data_sample_num):

            dataset_class = get_dataset_class(task)
            dataset = dataset_class(data_args=data_args,
                                    model_args=model_args,
                                    tokenizer=tokenizer,
                                    mode="train",
                                    prompt_sample_num=prompt_sample_num,
                                    sample_num=data_sample_num,
                                    **all_vocabs)
            train_datasets.append(dataset)

            logger.info(
                "task: {} data_num: {} sample_prompt: {} label: {}".format(
                    task, len(dataset), dataset[0]['input_ids'],
                    dataset[0]['labels']))

        train_data = ConcatDataset(train_datasets)
        return train_data
    elif (mode == 'valid') or (mode == 'test'):
        if mode == 'valid':
            val_tasks = data_args.val_tasks.split(",")
            val_data_sample_num = [
                int(_) for _ in data_args.val_data_sample_num.split(",")
            ]
        elif mode == 'test':
            val_tasks = data_args.test_tasks.split(",")
            val_data_sample_num = [
                int(_) for _ in data_args.test_data_sample_num.split(",")
            ]

        assert len(val_tasks) == len(
            val_data_sample_num
        ), "data sample number does not match task number"
        val_datasets = {}
        for val_task, data_sample_num in zip(val_tasks, val_data_sample_num):
            val_dataset_class = get_dataset_class(val_task)
            val_dataset = val_dataset_class(data_args=data_args,
                                            model_args=model_args,
                                            tokenizer=tokenizer,
                                            mode=mode,
                                            sample_num=data_sample_num,
                                            **all_vocabs)
            val_datasets[val_task] = val_dataset
        return val_datasets
    else:
        raise NotImplementedError


class BaseDataset(Dataset):

    def __init__(self,
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 tokenizer: PreTrainedTokenizer,
                 user_vocab,
                 item_vocab,
                 query_vocab,
                 src_session_vocab,
                 semantic_indices=None,
                 collaborate_indices=None,
                 mode="train",
                 prompt_sample_num=1,
                 prompt_id=0,
                 sample_num=-1):
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer

        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.query_vocab = query_vocab
        self.src_session_vocab = src_session_vocab

        self.semantic_indices = semantic_indices
        self.collaborate_indices = collaborate_indices

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.new_tokens = None
        self.all_items = None
        self.allowed_tokens = None

        self.all_prompt, self.sft_prompt = load_prompt()

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()

        for index in self.semantic_indices.values():
            for token in index:
                self.new_tokens.add(token)

        for index in self.collaborate_indices.values():
            for token in index:
                self.new_tokens.add(token)

        logger.info("add behavior token into tokenizer vocab")
        self.new_tokens.add(self.data_args.rec_item_token)
        self.new_tokens.add(self.data_args.src_query_token)
        self.new_tokens.add(self.data_args.src_item_token)

        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_item_token(self, item_id, semantic=False):
        if semantic:
            return "".join(self.semantic_indices[str(item_id)])
        else:
            return "".join(self.collaborate_indices[str(item_id)])

    def get_item_text(self, item_id):
        item_data = self.item_vocab[item_id]
        item_text = ' '.join([item_data['title'], item_data['description']])

        item_text = self.tokenizer.decode(
            self.tokenizer(item_text,
                           max_length=self.data_args.max_doc_len,
                           truncation=True)['input_ids'],
            skip_special_tokens=True)

        return item_text

    def get_query_text(self, query=None, query_id=None):
        if query_id is not None:
            query = self.query_vocab[int(query_id)]['query']
        else:
            assert query is not None
        query = self.tokenizer.decode(
            self.tokenizer(query,
                           max_length=self.data_args.max_query_len,
                           truncation=True)['input_ids'],
            skip_special_tokens=True)
        return query

    def get_prefix_allowed_tokens_fn(self,
                                     allowed_tokens=None,
                                     semantic=False,
                                     behavior_token=None):
        if (self.allowed_tokens is None) and (allowed_tokens is None):
            self.allowed_tokens = self.get_prefix_allowed_tokens(
                semantic=semantic, behavior_token=behavior_token)
            allowed_tokens = self.allowed_tokens

        eos_token_id = self.tokenizer.eos_token_id

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()[1:]  # skip start token
            tmp_prefix = ' '.join([str(x) for x in sentence])

            if type(allowed_tokens) == list:
                cur_allowed_tokens = allowed_tokens[batch_id]
            else:
                cur_allowed_tokens = allowed_tokens

            if tmp_prefix in cur_allowed_tokens.keys():
                return list(cur_allowed_tokens[tmp_prefix])
            else:
                return [eos_token_id]

        return prefix_allowed_tokens_fn

    def get_prefix_allowed_tokens(self,
                                  cand_items=None,
                                  semantic=False,
                                  behavior_token=None):
        if cand_items is not None:
            if semantic:
                allowed_index = [
                    self.semantic_indices[str(item)] for item in cand_items
                ]
            else:
                allowed_index = [
                    self.collaborate_indices[str(item)] for item in cand_items
                ]

        else:
            if semantic:
                allowed_index = self.semantic_indices.values()
            else:
                allowed_index = self.collaborate_indices.values()

        if behavior_token is not None:
            allowed_index = [[behavior_token] + x for x in allowed_index]

        allowed_tokens = {}
        for index in allowed_index:
            prefix_token_id = []
            for pos, token in enumerate(index):
                token_id = self.tokenizer(
                    token, add_special_tokens=False)["input_ids"]

                assert len(token_id) == 1
                token_id = token_id[0]

                tmp_prefix = ' '.join(prefix_token_id)
                if tmp_prefix not in allowed_tokens.keys():
                    allowed_tokens[tmp_prefix] = set()
                allowed_tokens[tmp_prefix].add(token_id)

                prefix_token_id.append(str(token_id))

            tmp_prefix = ' '.join(prefix_token_id)
            if tmp_prefix not in allowed_tokens.keys():
                allowed_tokens[tmp_prefix] = set()
            allowed_tokens[tmp_prefix].add(self.tokenizer.eos_token_id)
        return allowed_tokens

    def get_rec_his(self, user, rec_his_num):
        rec_his = self.user_vocab[user]['rec_his'][:rec_his_num]
        rec_his_ts = self.user_vocab[user]['rec_his_ts'][:rec_his_num]

        rec_his = rec_his[-self.data_args.max_rec_his_len:]
        rec_his_ts = rec_his_ts[-self.data_args.max_rec_his_len:]

        rec_his = [
            self.data_args.rec_item_token +
            self.get_item_token(i, semantic=False) for i in rec_his
        ]
        return list(zip(*[rec_his, rec_his_ts]))

    def get_rec_his_prompt(self, user, rec_his_num):
        rec_his = self.get_rec_his(user, rec_his_num)
        rec_his_prompt = self.data_args.his_sep.join([x[0] for x in rec_his])
        return rec_his_prompt

    def load_session_info(self, session_id):
        session_info = self.src_session_vocab[session_id]

        query = self.get_query_text(query=session_info['query'])
        query = self.data_args.src_query_token + query

        pos_item_ids = session_info['pos_items'][:self.data_args.
                                                 max_session_item_len]
        pos_item_ts = session_info['time_list'][:self.data_args.
                                                max_session_item_len]

        pos_item_tokens = [
            self.data_args.src_item_token +
            self.get_item_token(i, semantic=True) for i in pos_item_ids
        ]
        query_ts = pos_item_ts[0]

        src_session_his = [query] + pos_item_tokens
        src_session_his_ts = [query_ts] + pos_item_ts
        return list(zip(*[src_session_his, src_session_his_ts]))

    def get_src_session_his(self, user, src_session_his_num):
        src_session_his_ids = self.user_vocab[user][
            'src_session_his'][:src_session_his_num]
        src_session_his_ids = src_session_his_ids[-self.data_args.
                                                  max_src_session_his_len:]

        src_session_his = sum([
            self.load_session_info(session_id)
            for session_id in src_session_his_ids
        ], [])
        return src_session_his

    def get_src_session_his_prompt(self, user, src_session_his_num):
        src_session_his = self.get_src_session_his(user, src_session_his_num)
        src_session_his_prompt = self.data_args.his_sep.join(
            [x[0] for x in src_session_his])
        return src_session_his_prompt

    def get_merge_rec_src_session_prompt(self, user, rec_his_num,
                                         src_session_his_num):
        rec_his = self.get_rec_his(user, rec_his_num)
        src_session_his = self.get_src_session_his(user, src_session_his_num)

        all_his = rec_his + src_session_his
        all_his = sorted(all_his, key=lambda x: x[1])

        all_his_prompt = self.data_args.his_sep.join([x[0] for x in all_his])

        return all_his_prompt


class Item2IndexDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prompts = self.all_prompt["item2index"]

        src_item_set: List = load_pickle(
            os.path.join('../data', self.data_args.dataset,
                         "vocab/src_item_set.pkl"))
        rec_item_set: List = load_pickle(
            os.path.join('../data', self.data_args.dataset,
                         "vocab/rec_item_set.pkl"))

        self.all_item_ids = sorted(list(set(src_item_set) | set(rec_item_set)))

        if self.sample_num > 0:
            self.data_num = self.sample_num
        else:
            self.data_num = len(self.all_item_ids)

    def _get_text_data(self, item_id, prompt, semantic=False):
        item_content = self.get_item_text(item_id)
        item_indice = self.get_item_token(item_id, semantic=semantic)

        data = {"content": item_content, "item": item_indice}

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = self.sft_prompt.format(instruction=instruction, response="")

        return dict(input_ids=input, labels=response)

    def __len__(self):
        return self.data_num * self.prompt_sample_num * 2

    def __getitem__(self, index):
        item_id = random.choice(self.all_item_ids)
        if index % 2 == 0:
            semantic = True
        else:
            semantic = False

        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        data = self._get_text_data(item_id, prompt, semantic=semantic)
        return data


class Index2ItemDataset(Item2IndexDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prompts = self.all_prompt["index2item"]


class SeqRecWithSrcHisDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts = self.all_prompt["seqrec"]
        self.load_data()

        if self.mode == 'valid':
            self.prompts = [self.prompts[self.prompt_id]]

    def load_data(self):
        if self.mode == 'train':
            file_name = 'rec_train.pkl'
        elif self.mode == 'valid':
            file_name = 'rec_val.pkl'
        elif self.mode == 'test':
            file_name = 'rec_test.pkl'
        else:
            raise NotImplementedError

        self.inter_data: pd.DataFrame = load_pickle(
            os.path.join('../data', self.data_args.dataset,
                         f"dataset/{file_name}"))

        if self.mode == 'train':
            self.inter_idx = list(range(len(self.inter_data)))
            if self.sample_num > 0:
                self.data_num = self.sample_num
            else:
                self.data_num = len(self.inter_data)
        else:
            if self.sample_num > 0:
                if 'timestamp' in self.inter_data.columns:
                    self.inter_data = self.inter_data.sort_values(
                        by=['timestamp']).iloc[-self.sample_num:].reset_index(
                            drop=True)
                elif 'ts' in self.inter_data.columns:
                    self.inter_data = self.inter_data.sort_values(
                        by=['ts']).iloc[-self.sample_num:].reset_index(
                            drop=True)
                else:
                    raise KeyError
            self.data_num = len(self.inter_data)

    def __len__(self):
        if self.mode == 'train':
            return self.data_num * self.prompt_sample_num
        elif (self.mode == 'valid') or (self.mode == 'test'):
            return self.data_num
        else:
            raise NotImplementedError

    def _get_text_data(self, line, prompt):
        user = int(line['user_id'])
        item = self.get_item_token(line['item_id'], semantic=False)

        all_his_prompt = self.get_merge_rec_src_session_prompt(
            user, int(line['rec_his']), int(line['src_session_his']))
        item = self.data_args.rec_item_token + item

        data = {"history": all_his_prompt, "target": item}

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = self.sft_prompt.format(instruction=instruction, response="")

        return dict(input_ids=input, labels=response)

    def __getitem__(self, index):
        if (self.sample_num > 0) and (self.mode == 'train'):
            idx = random.choice(self.inter_idx)
        else:
            idx = index // self.prompt_sample_num
        line = self.inter_data.iloc[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif (self.mode == 'valid') or (self.mode == 'test'):
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        data = self._get_text_data(line, prompt)

        if self.mode != 'train':
            cand_items = [line['item_id']
                          ] + line['neg_items'][:self.data_args.num_negs]
            cur_prefix_allowed_tokens = self.get_prefix_allowed_tokens(
                cand_items, semantic=False)
            data['test_prefix_allowed_tokens'] = cur_prefix_allowed_tokens

        return data

    def get_prefix_allowed_tokens_fn(self,
                                     allowed_tokens=None,
                                     semantic=False,
                                     behavior_token=None):
        return super().get_prefix_allowed_tokens_fn(
            allowed_tokens=allowed_tokens,
            semantic=semantic,
            behavior_token=behavior_token)

    def get_prefix_allowed_tokens(self, cand_items=None, semantic=False):
        return super().get_prefix_allowed_tokens(
            cand_items=cand_items,
            semantic=semantic,
            behavior_token=self.data_args.rec_item_token)


class PerSrcWithRecHisDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts = self.all_prompt["persrc"]

        self.load_data()

        if self.mode == 'valid':
            self.prompts = [self.prompts[self.prompt_id]]

    def load_data(self):
        if self.mode == 'train':
            file_name = 'src_train.pkl'
        elif self.mode == 'valid':
            file_name = 'src_val.pkl'
        elif self.mode == 'test':
            file_name = 'src_test.pkl'
        else:
            raise NotImplementedError

        self.inter_data: pd.DataFrame = load_pickle(
            os.path.join('../data', self.data_args.dataset,
                         f"dataset/{file_name}"))

        if self.mode == 'train':
            self.inter_idx = list(range(len(self.inter_data)))
            if self.sample_num > 0:
                self.data_num = self.sample_num
            else:
                self.data_num = len(self.inter_data)

        else:
            if self.sample_num > 0:
                if 'timestamp' in self.inter_data.columns:
                    self.inter_data = self.inter_data.sort_values(
                        by=['timestamp']).iloc[-self.sample_num:].reset_index(
                            drop=True)
                elif 'ts' in self.inter_data.columns:
                    self.inter_data = self.inter_data.sort_values(
                        by=['ts']).iloc[-self.sample_num:].reset_index(
                            drop=True)
                else:
                    raise KeyError
            self.data_num = len(self.inter_data)

    def __len__(self):
        if self.mode == 'train':
            return self.data_num * self.prompt_sample_num
        elif (self.mode == 'valid') or (self.mode == 'test'):
            return self.data_num
        else:
            raise NotImplementedError

    def _get_text_data(self, line, prompt):
        user = int(line['user_id'])
        item = self.get_item_token(line['item_id'], semantic=True)
        query = self.get_query_text(query_id=line['query_id'])

        all_his_prompt = self.get_merge_rec_src_session_prompt(
            user, int(line['rec_his']), int(line['src_session_his']))

        item = self.data_args.src_item_token + item
        query = self.data_args.src_query_token + query

        data = {
            "history": prompt['history'].format(**{'history': all_his_prompt}),
            "query": query,
            "target": item
        }

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = self.sft_prompt.format(instruction=instruction, response="")

        return dict(input_ids=input, labels=response)

    def __getitem__(self, index):
        if (self.sample_num > 0) and (self.mode == 'train'):
            idx = random.choice(self.inter_idx)
        else:
            idx = index // self.prompt_sample_num
        line = self.inter_data.iloc[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif (self.mode == 'valid') or (self.mode == 'test'):
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        data = self._get_text_data(line, prompt)

        if self.mode != 'train':
            cand_items = [line['item_id']
                          ] + line['neg_items'][:self.data_args.num_negs]
            cur_prefix_allowed_tokens = self.get_prefix_allowed_tokens(
                cand_items, semantic=True)
            data['test_prefix_allowed_tokens'] = cur_prefix_allowed_tokens

        return data

    def get_prefix_allowed_tokens_fn(self,
                                     allowed_tokens=None,
                                     semantic=True,
                                     behavior_token=None):
        return super().get_prefix_allowed_tokens_fn(
            allowed_tokens=allowed_tokens,
            semantic=semantic,
            behavior_token=behavior_token)

    def get_prefix_allowed_tokens(self, cand_items=None, semantic=False):
        return super().get_prefix_allowed_tokens(
            cand_items=cand_items,
            semantic=semantic,
            behavior_token=self.data_args.src_item_token)


class QueryGenWithRecHisDataset(PerSrcWithRecHisDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts = self.all_prompt["queryrec"]

    def _get_text_data(self, line, prompt):
        user = int(line['user_id'])
        query = self.get_query_text(query_id=line['query_id'])

        all_his_prompt = self.get_merge_rec_src_session_prompt(
            user, int(line['rec_his']), int(line['src_session_his']))

        query = self.data_args.src_query_token + query

        data = {"history": all_his_prompt, "target": query}

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = self.sft_prompt.format(instruction=instruction, response="")

        return dict(input_ids=input, labels=response)


class AdHocSrcDataset_Doc2Query(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts = self.all_prompt["persrc"]
        self.load_data()

    def load_data(self):
        self.item2query: Dict = load_pickle(
            os.path.join('../data', self.data_args.dataset,
                         f"vocab/{self.data_args.doc2query_file}.pkl"))

        src_item_set = load_pickle(
            os.path.join('../data', self.data_args.dataset,
                         "vocab/src_item_set.pkl"))
        self.iter_items = list(src_item_set)

        if self.sample_num > 0:
            self.data_num = self.sample_num
        else:
            self.data_num = len(self.iter_items)

    def _get_text_data(self, item_id, prompt):
        item = self.get_item_token(item_id, semantic=True)
        query = random.choice(list(set(self.item2query[item_id])))

        query = self.data_args.src_query_token + query
        item = self.data_args.src_item_token + item

        data = {"history": "", "query": query, "target": item}

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = self.sft_prompt.format(instruction=instruction, response="")

        return dict(input_ids=input, labels=response)

    def __len__(self):
        return self.data_num * self.prompt_sample_num * self.data_args.max_query_per_doc

    def __getitem__(self, index):
        if self.sample_num > 0:
            item_id = random.choice(self.iter_items)
        else:
            idx = index // self.prompt_sample_num // self.data_args.max_query_per_doc
            item_id = self.iter_items[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        data = self._get_text_data(item_id, prompt)

        return data


class Doc2QueryDataSet(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.load_data()

    def load_data(self):
        file_name = 'src_train.pkl'

        self.inter_data: pd.DataFrame = load_pickle(
            os.path.join('../data', self.data_args.dataset,
                         f"dataset/{file_name}"))

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        line = self.inter_data.iloc[index]

        item_content = self.get_item_text(line['item_id'])
        query = self.get_query_text(query_id=line['query_id'])

        return dict(input_ids=f"{item_content} ", labels=query)

    def get_new_tokens(self):
        return []
