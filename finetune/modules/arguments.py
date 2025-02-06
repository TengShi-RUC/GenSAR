from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy, SchedulerType
from transformers.training_args import OptimizerNames, trainer_log_levels


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default='t5-small',
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })

    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from s3"
        })

    max_input_len: int = field(default=2048)


@dataclass
class DataArguments:
    dataset: str = field(default="")

    semantic_index_file: str = field(default='')
    collaborate_index_file: str = field(default='')

    rec_item_token: str = field(default="<rec_item>")
    src_query_token: str = field(default="<src_query>")
    src_item_token: str = field(default="<src_item>")

    max_rec_his_len: int = field(default=10)
    max_src_session_his_len: int = field(default=5)
    max_session_item_len: int = field(default=5)
    max_query_len: int = field(default=64)
    max_doc_len: int = field(default=256)
    max_query_per_doc: int = field(default=5)
    doc2query_file: str = field(default="")

    add_prefix: bool = field(default=False)
    his_sep: str = field(default="; ")
    src_session_sep: str = field(default=", ")

    train_tasks: str = field(default="AdHocSrc")
    val_tasks: str = field(default="AdHocSrc")
    test_tasks: str = field(default="AdHocSrc")

    train_prompt_sample_num: str = field(default="1")
    train_data_sample_num: str = field(default="1000")

    val_prompt_ids: str = field(default="0")
    val_data_sample_num: str = field(default="100")
    test_data_sample_num: str = field(default="100")

    num_negs: int = field(default=99)
    num_beams: int = field(default=30)
    max_new_tokens: int = field(default=32)

    only_train_response: bool = field(default=True)


@dataclass
class LLMTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default='',
        metadata={
            "help":
            "The output directory where the model predictions and checkpoints will be written."
        })

    seed: int = field(
        default=2024,
        metadata={
            "help":
            "Random seed that will be set at the beginning of training."
        })

    dataloader_num_workers: int = field(
        default=8,
        metadata={
            "help":
            ("Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
             " in the main process.")
        },
    )

    bf16: bool = field(
        default=False,
        metadata={
            "help":
            ("Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
             " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
             )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use fp16 (mixed) precision instead of 32-bit"
        },
    )

    deepspeed: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help":
            ("Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
             " loaded json file as a dict")
        },
    )

    do_train: bool = field(default=False,
                           metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set."})

    default_optim = "adamw_torch"
    # XXX: enable when pytorch==2.0.1 comes out - we want to give it time to get all the bugs sorted out
    # if is_torch_available() and version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.1.0"):
    #     default_optim = "adamw_torch_fused"
    # and update the doc above to:
    # optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch_fused"` (for torch<2.1.0 `"adamw_torch"`):
    optim: Union[OptimizerNames, str] = field(
        default=default_optim,
        metadata={"help": "The optimizer to use."},
    )
    num_train_epochs: float = field(
        default=2.0,
        metadata={"help": "Total number of training epochs to perform."})

    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW."})
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="reduce_lr_on_plateau",
        metadata={"help": "The scheduler type to use."},
    )

    lr_scheduler_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={
            "help":
            ("Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
             )
        },
    )

    warmup_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Linear warmup over warmup_ratio fraction of total steps."
        })

    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."})

    per_device_train_batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."
        })
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."
        })
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={
            "help":
            "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )

    log_level: Optional[str] = field(
        default="info",
        metadata={
            "help":
            ("Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
             " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
             " lets the application set the level. Defaults to 'passive'."),
            "choices":
            trainer_log_levels.keys(),
        },
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help":
            ("Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
             "If smaller than 1, will be interpreted as ratio of total training steps."
             )
        },
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The path to a folder with a valid checkpoint for your model."
        },
    )

    eval_strategy: Union[IntervalStrategy, str] = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use."},
    )
    """
    - `"no"`: No evaluation is done during training.
    - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
    - `"epoch"`: Evaluation is done at the end of each epoch.
    """
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help":
            ("Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
             "If smaller than 1, will be interpreted as ratio of total training steps."
             )
        },
    )
    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help":
            ("Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
             " eval_strategy.")
        },
    )

    save_strategy: Union[IntervalStrategy, str] = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    """
    - `"no"`: No save is done during training.
    - `"epoch"`: Save is done at the end of each epoch.
    - `"steps"`: Save is done every `save_steps`.
    """
    save_steps: float = field(
        default=500,
        metadata={
            "help":
            ("Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
             "If smaller than 1, will be interpreted as ratio of total training steps."
             )
        },
    )
    save_total_limit: Optional[int] = field(
        default=5,
        metadata={
            "help":
            ("If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
             " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
             " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
             " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
             " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
             " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
             " Default is unlimited checkpoints")
        },
    )

    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={
            "help":
            ("Whether or not to load the best model found during training at the end of training. When this option"
             " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
             )
        },
    )
    metric_for_best_model: Optional[str] = field(
        default='ndcg@5',
        metadata={
            "help": "The metric to use to compare two different models."
        })
    greater_is_better: Optional[bool] = field(
        default=None,
        metadata={
            "help":
            "Whether the `metric_for_best_model` should be maximized or not."
        })

    eval_metrics: str = field(
        default="hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20")

    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help":
            "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
