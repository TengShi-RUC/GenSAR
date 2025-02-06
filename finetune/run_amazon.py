import itertools
import os
import random
import subprocess
from datetime import datetime


def run_cmd(cmd, arguments_dict=None):
    if arguments_dict is not None:
        for k, v in arguments_dict.items():
            cmd += f" --{k} {v}"

    print("\nrunning cmd: ", cmd)
    start = datetime.now()
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

    end = datetime.now()
    print("runnning used time:{}\n".format(end - start))


DATASET = "Amazon_Electronics"

LLM_NAME = "t5-small"
BASE_MODEL = f"{LLM_NAME}"
DATA_PATH = "../data"

TRAIN_TASKS = "SeqRecWithSrcHis,PerSrcWithRecHis,QGenWithRecHis,AdHocSrc_Doc2Query,Item2Index,Index2Item"
TEST_TASKS = "SeqRecWithSrcHis,PerSrcWithRecHis"
RUNNER = 'sar'

OUTPUT_DIR = f"./output/{DATASET}_{LLM_NAME}_{RUNNER}/"

CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
nproc_per_node = len(CUDA_VISIBLE_DEVICES.split(','))
master_port = str(random.randint(10000, 50000))

finetune_args = {
    # ModelArguments
    "model_name_or_path": BASE_MODEL,
    # DataArguments
    "dataset": DATASET,
    "doc2query_file": "\"\"",
    "train_tasks": TRAIN_TASKS,
    "val_tasks": TEST_TASKS,
    "test_tasks": TEST_TASKS,
    "train_prompt_sample_num": ",".join(['1'] * len(TRAIN_TASKS.split(','))),
    "train_data_sample_num": '0,100000,1000,1000,1000,1000',
    "val_prompt_ids": "0",
    "val_data_sample_num": ",".join(['10000'] * len(TEST_TASKS.split(','))),
    "test_data_sample_num": ",".join(['0'] * len(TEST_TASKS.split(','))),
    # LLMTrainingArguments
    "output_dir": OUTPUT_DIR,
    "dataloader_num_workers": 8,
    "do_train": True,
    "do_eval": True,
    "do_predict": True,
    "num_train_epochs": 20,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0,
    "weight_decay": 0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "ndcg@5",
    "eval_metrics": "hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20",
    "resume_from_checkpoint": "\"\""
}

new_config_ls = []


def add_to_new_config(tune_paras):
    for key, value in tune_paras.items():
        print("{}: {}".format(key, value))
    print()

    para_key_names = list(tune_paras.keys())
    for values in itertools.product(*(v for _, v in tune_paras.items())):
        tmp_config = {}
        for index, element in enumerate(values):
            tmp_config[para_key_names[index]] = element
        new_config_ls.append(tmp_config)


tune_paras = {
    'dataloader_num_workers': [8],
    'max_rec_his_len': [5],
    'max_src_session_his_len': [5],
    'max_session_item_len': [1],
    "do_train": [True],
    "do_predict": [True],
    'semantic_index_file': [''],
    'collaborate_index_file': [''],
    "doc2query_file": ['item2query_div_nq-20'],
    "max_query_per_doc": [1],
    "num_train_epochs": [200],
    "learning_rate": [1e-3],
    "lr_scheduler_type": ["cosine"],
    "warmup_ratio": [0.01],
    "weight_decay": [1e-7],
    "per_device_train_batch_size": [64],
    "per_device_eval_batch_size": [32],
    'gradient_accumulation_steps': [16],
    # "resume_from_checkpoint":
    # [f"output/{DATASET}_{LLM_NAME}_{RUNNER}/"],
    'num_beams': [30],
}

add_to_new_config(tune_paras)

for new_config in new_config_ls:
    cur_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")

    finetune_args['output_dir'] = os.path.join(OUTPUT_DIR, cur_time)
    finetune_args.update(new_config)

    if finetune_args['do_train']:
        printDir = os.path.join(OUTPUT_DIR, cur_time)
        os.makedirs(printDir, exist_ok=True)
        printFile = os.path.join(printDir, "train_out.log")
    else:
        if os.path.exists(finetune_args['resume_from_checkpoint']):
            printDir = finetune_args['resume_from_checkpoint']
        else:
            printDir = os.path.join(OUTPUT_DIR, cur_time)
            os.makedirs(printDir, exist_ok=True)
        printFile = os.path.join(printDir, "test_out.log")

    cmd = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} nohup torchrun --nproc_per_node={nproc_per_node} --master_port={master_port} main.py'

    for k, v in finetune_args.items():
        if v is not None:
            cmd += f' --{k} {v}'

    cmd += " > {} 2>&1 ".format(printFile)

    run_cmd(cmd)
