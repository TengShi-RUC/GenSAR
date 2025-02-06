import argparse
import logging
import os
import pickle

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer


def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


class ItemCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __call__(self, batch):
        item_ids = [d["item_id"] for d in batch]
        item_texts = [d["item_text"] for d in batch]

        inputs = self.tokenizer(
            text=item_texts,
            return_tensors="pt",
            padding=True,
            max_length=self.args.doc_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        return item_ids, inputs


class ItemDataset(Dataset):

    def __init__(self, args) -> None:
        super().__init__()
        self.dataset = args.dataset
        self.item_vocab = load_pickle(
            os.path.join(args.dataset, 'vocab/item_vocab.pkl'))
        if args.sample_number > 0:
            self.data_num = args.sample_number
        else:
            self.data_num = len(self.item_vocab) - 1  # skip 0 for pad

    def __getitem__(self, index):
        item_data = self.item_vocab[index + 1]
        item_text = ' '.join([item_data['title'], item_data['description']])

        return {"item_id": index + 1, "item_text": item_text}

    def __len__(self):
        return self.data_num


def doc2query(args):
    if args.use_tuned_llm:
        args.llm_path = args.tuned_llm_path

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )

    if local_rank == 0:
        for flag, value in args.__dict__.items():
            logging.info('{}: {} {}'.format(flag, value, type(value)))

    dist.init_process_group(backend="nccl",
                            world_size=world_size,
                            rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)

    tokenizer = T5Tokenizer.from_pretrained(args.llm_path, padding_side="left")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.llm_path,
                                                  device_map=device_map)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    dataset = ItemDataset(args)
    ddp_sampler = DistributedSampler(dataset,
                                     shuffle=False,
                                     num_replicas=world_size,
                                     rank=local_rank,
                                     drop_last=False)
    collator = ItemCollator(args, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            collate_fn=collator,
                            sampler=ddp_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)

    model.eval()

    with torch.no_grad():
        item2query = {}
        for step, batch in enumerate(tqdm(dataloader)):
            item_ids = batch[0]
            inputs = batch[1].to(device)

            if args.div:
                output_ids = model.module.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=args.query_max_length,
                    do_sample=True,
                    top_p=0.95,
                    top_k=10,
                    num_return_sequences=args.num_querys)
            else:
                output_ids = model.module.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=args.query_max_length,
                    num_beams=args.num_querys,
                    no_repeat_ngram_size=2,
                    num_return_sequences=args.num_querys,
                    early_stopping=True)

            output = tokenizer.batch_decode(output_ids,
                                            skip_special_tokens=True)

            item_id_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=item_ids,
                                   object_list=item_id_gather_list)
            out_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=output, object_list=out_gather_list)

            if local_rank == 0:
                all_device_output = []
                for ga_res in out_gather_list:
                    all_device_output += ga_res
                all_device_item_ids = []
                for ga_res in item_id_gather_list:
                    all_device_item_ids += ga_res

                B = len(all_device_item_ids)
                K = args.num_querys
                for b in range(B):
                    cur_item_out = all_device_output[b * K:(b + 1) * K]
                    cur_item_id = all_device_item_ids[b]
                    item2query[cur_item_id] = cur_item_out
            dist.barrier()

    dist.barrier()

    if local_rank == 0:
        logging.info("Save file")
        file_name = 'item2query'
        if args.use_tuned_llm:
            file_name += f"_{args.llm_path.split('/')[-1]}"
        if args.div:
            file_name += "_div"
        file_name += f"_nq-{args.num_querys}"

        with open(os.path.join(args.dataset, f'vocab/{file_name}.pkl'),
                  'wb') as fp:
            pickle.dump(item2query, fp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='../data/Amazon_Electronics')

    parser.add_argument('--llm_path',
                        type=str,
                        default='doc2query/msmarco-t5-base-v1')
    parser.add_argument('--tuned_llm_path', type=str, default='')
    parser.add_argument('--use_tuned_llm', type=int, default=0)

    parser.add_argument('--sample_number', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--doc_max_length', type=int, default=256)
    parser.add_argument('--query_max_length', type=int, default=64)
    parser.add_argument('--num_querys', type=int, default=20)

    parser.add_argument('--div', type=int, default=1)

    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    doc2query(args)
