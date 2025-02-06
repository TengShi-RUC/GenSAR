import itertools
import os
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


def run_generate_indices(cur_time):
    generate_index_args = {
        "dataset": dataset,
        "ckpt_path": os.path.join(OUTPUT_DIR, cur_time),
        "ckpt_type": "best_collision_model",
        "device": train_index_args['device']
    }
    printDir = os.path.join(OUTPUT_DIR, cur_time)
    os.makedirs(printDir, exist_ok=True)
    printFile = os.path.join(printDir, "generate.log")

    generate_cmd = "nohup python -u generate_indices.py"
    for k, v in generate_index_args.items():
        if v is not None:
            generate_cmd += f' --{k} {v}'

    generate_cmd += " > {} 2>&1 ".format(printFile)

    run_cmd(generate_cmd)


dataset = "Amazon_Electronics"
OUTPUT_DIR = f"./output/{dataset}"

semantic_emb = "bge-base-en-v1.5"
collaborate_emb = "UniSAR"

OUTPUT_DIR += "_BGE"
OUTPUT_DIR += "_UniSAR"

train_index_args = {
    "seed": 2024,
    "lr": 1e-3,
    "epochs": 1000,
    "batch_size": 1024,
    "weight_decay": 1e-4,
    "lr_scheduler_type": "linear",
    "dropout_prob": 0.0,
    "bn": 0,
    "quant_loss_weight": 1.0,
    "beta": 0.25,
    "layers": "2048 1024 512 256 128 64",
    "device": "cuda:0",
    "dataset": dataset,
    "ckpt_dir": dataset,
    "eval_step": 50,
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
    "lr": [1e-3],
    "epochs": [500],
    "weight_decay": [1e-5],
    "device": ["cuda:0"],
    "quant_loss_weight": [1.0],
    "layers": ["512 256 128 64"],
    "eval_step": [10],
    'semantic_emb': [semantic_emb],
    'collaborate_emb': [collaborate_emb],
    'shared_e_dim': [64],
    'specific_e_dim': [32],
    'shared_num_emb_list': ["256 256"],
    'semantic_num_emb_list': ["256 256"],
    'collaborate_num_emb_list': ['256 256'],
    'shared_sk_epsilons': ["0 0"],
    'specific_sk_epsilons': ["0 0.003"],
}
add_to_new_config(tune_paras)

for new_config in new_config_ls:
    cur_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")

    train_index_args['ckpt_dir'] = os.path.join(OUTPUT_DIR, cur_time)
    train_index_args.update(new_config)

    printDir = os.path.join(OUTPUT_DIR, cur_time)
    os.makedirs(printDir, exist_ok=True)
    printFile = os.path.join(printDir, "out.log")

    cmd = 'nohup python -u main.py'

    for k, v in train_index_args.items():
        if v is not None:
            cmd += f' --{k} {v}'

    cmd += " > {} 2>&1 ".format(printFile)

    run_cmd(cmd)
    run_generate_indices(cur_time)
