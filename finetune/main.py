import logging
import os
from pathlib import Path

import torch
from modules.arguments import (DataArguments, LLMTrainingArguments,
                               ModelArguments)
from modules.collator import TrainDataCollator
from modules.data import load_dataset
from modules.trainer import LLMTrainer
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          HfArgumentParser, PreTrainedTokenizer, set_seed)

logger = logging.getLogger(__name__)


def log_args(args):
    for flag, value in args.__dict__.items():
        logger.info('{}: {} {}'.format(flag, value, type(value)))
    logger.info("")


def save_trainer(trainer: LLMTrainer, tokenizer: PreTrainedTokenizer, config,
                 training_args: LLMTrainingArguments):
    trainer.save_state()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: LLMTrainingArguments

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16 or training_args.bf16,
    )

    logger.info("Training/evaluation parameters")
    log_args(training_args)
    logger.info("Model parameters")
    log_args(model_args)
    logger.info("Data parameters")
    log_args(data_args)

    # Set seed
    set_seed(training_args.seed)

    if (training_args.resume_from_checkpoint is not None) and (
            not os.path.exists(training_args.resume_from_checkpoint)):
        logger.info("resume_from_checkpoint: {} not exist".format(
            training_args.resume_from_checkpoint))
        training_args.resume_from_checkpoint = None

    if training_args.resume_from_checkpoint is not None:
        tokenizer_path = training_args.resume_from_checkpoint
        config_path = training_args.resume_from_checkpoint
        if not training_args.do_train:
            training_args.output_dir = training_args.resume_from_checkpoint
    else:
        tokenizer_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
        config_path = model_args.config_name if model_args.config_name else model_args.model_name_or_path

    logger.info(f"tokenizer path: {tokenizer_path}")
    logger.info(f"config path: {config_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              cache_dir=model_args.cache_dir,
                                              use_fast=False)

    config = AutoConfig.from_pretrained(config_path)
    logger.info('Config: %s', config)

    if training_args.do_train:
        train_dataset = load_dataset(data_args=data_args,
                                     model_args=model_args,
                                     tokenizer=tokenizer,
                                     mode='train')
        logger.info("train data num: {}".format(len(train_dataset)))
    else:
        train_dataset = None

    if training_args.resume_from_checkpoint is None:
        add_num = tokenizer.add_tokens(
            train_dataset.datasets[0].get_new_tokens())
        config.vocab_size = len(tokenizer)
        if training_args.local_rank == 0:
            logger.info("add {} new token.".format(add_num))

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    collator = TrainDataCollator(data_args=data_args,
                                 model_args=model_args,
                                 tokenizer=tokenizer)

    if training_args.do_eval:
        val_dataset = load_dataset(data_args=data_args,
                                   model_args=model_args,
                                   tokenizer=tokenizer,
                                   mode='valid')
        for task, dataset in val_dataset.items():
            logger.info("val: {} data num: {}".format(task, len(dataset)))
        trainer = LLMTrainer(model=model,
                             train_dataset=train_dataset,
                             eval_dataset=val_dataset,
                             args=training_args,
                             tokenizer=tokenizer,
                             data_collator=collator)
    else:
        val_dataset = None
        trainer = LLMTrainer(model=model,
                             train_dataset=train_dataset,
                             args=training_args,
                             tokenizer=tokenizer,
                             data_collator=collator)

    if training_args.do_train:
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        try:
            trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint)
            save_trainer(trainer, tokenizer, config, training_args)
        except (KeyboardInterrupt, torch.cuda.OutOfMemoryError):
            if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is not None:
                trainer._load_best_model()
            save_trainer(trainer, tokenizer, config, training_args)
            raise KeyboardInterrupt
        except Exception as e:
            if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is not None:
                trainer._load_best_model()
            save_trainer(trainer, tokenizer, config, training_args)
            raise e
    else:
        if training_args.resume_from_checkpoint is not None:
            trainer._load_from_checkpoint(training_args.resume_from_checkpoint)

        else:
            raise ValueError("not train and no resume_from_checkpoint")

    if training_args.do_predict:
        test_dataset = load_dataset(data_args=data_args,
                                    model_args=model_args,
                                    tokenizer=tokenizer,
                                    mode='test')
        for task, dataset in test_dataset.items():
            logger.info("test: {} data num: {}".format(task, len(dataset)))
        test_results = trainer.evaluate(test_dataset, metric_key_prefix='test')


if __name__ == "__main__":
    main()
