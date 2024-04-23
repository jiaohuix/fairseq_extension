#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 05.02.22 09:27
# @Author  : Wen Lai
# @Site    :
# @File    : run_translation.py
# @Usage information:

# Copyright (c) 2021-present, CIS, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
数据格式：
{ "translation": { "en": "Others have dismissed him as a joke.", "ro": "Alții l-au numit o glumă." }}
{ "translation": { "en": "And some are holding out for an implosion.", "ro": "Iar alții așteaptă implozia." }}

#todo : 指定固定语言

"""
import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import Pool
from functools import partial
from types import MethodType

import torch
import datasets
import pandas as pd
from datasets import Dataset
import evaluate
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    # M2MSeq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from torch.utils.data import ConcatDataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.17.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    use_badam: bool = field(
        default=True,
        metadata={"help": "Whether to use BAdam or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(default=None, metadata={"help": "data path for all seen domains."})

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    lang_pairs: Optional[str] = field(
        default=None, metadata={"help": "Language pairs,such as zh-ru,fr-zh"}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
                    "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
                    "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
                    "needs to be the target language token.(Usually it is the target language token)"
        },
    )


def process_lang_pair_dataset(lang_pair, data_args, model_args):
    '''


    Note: multiprocessing uses pickle, which can only serialize top-module level functions in general.
    AttributeError: Can't pickle local object in Multiprocessing
    https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing
    '''
    data_name_suffix = data_args.dataset_name.rstrip("/").split("/")[-1]
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    source_lang = lang_pair.split('-')[0]
    target_lang = lang_pair.split('-')[1]
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        forced_bos_list = [forced_bos_token_id for i in range(len(model_inputs['labels']))]
        model_inputs["forced_bos_token_id"] = forced_bos_list
        return model_inputs

    dataset_config_name = data_name_suffix + "-" + lang_pair

    dataset = load_dataset(data_args.dataset_name, dataset_config_name, cache_dir="./datasets/",
                           verification_mode="no_checks")

    train_datasets, valid_datasets = dataset["train"], dataset["validation"]
    # tokenize
    lang = f'{source_lang}-{target_lang}'
    column_names = train_datasets.column_names
    train_datasets_token = train_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Running tokenizer on {lang} train dataset",
    )
    valid_datasets_token = valid_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Running tokenizer on {lang} valid dataset",
    )
    return train_datasets_token, valid_datasets_token


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    ### process the datasets for all language pairs
    data_name_suffix = data_args.dataset_name.rstrip("/").split("/")[-1]
    data_langs_map = {
        "ikcest2022": ["zh-th", "th-zh", "zh-fr", "fr-zh", "zh-ru", "ru-zh", "zh-ar", "ar-zh"],
        "iwslt2017": ["en-it", "it-en", "en-ro", "ro-en", "en-nl", "nl-en", "it-ro", "ro-it"]
    }

    # infer language pair
    if data_args.lang_pairs is not None:
        cfg_pairs = data_args.lang_pairs.strip().split(",")
        valid_pairs = []
        for pair in cfg_pairs:
            langs = pair.split("-")
            if len(langs) == 2 and pair in data_langs_map[data_name_suffix]:
                valid_pairs.append(pair)
        if len(valid_pairs) > 0:
            lang_pairs = valid_pairs
        else:
            # Handle case when no valid pairs found
            raise ValueError("No valid language pairs found in provided config.")
    else:
        lang_pairs = data_langs_map[data_name_suffix]

    # ========= parallel =========
    logger.info(f"tokenize data")

    with Pool(processes=len(lang_pairs)) as pool:
        results = pool.map(
            partial(process_lang_pair_dataset, data_args=data_args, model_args=model_args), lang_pairs)

    train_datasets_list, valid_datasets_list = zip(*results)

    ### create dataloaders
    train_dataset_merge = ConcatDataset(train_datasets_list)
    eval_dataset_merge = ConcatDataset(valid_datasets_list)
    logger.info(f"train_dataset size {len(train_dataset_merge)}")
    logger.info(f"eval_dataset size {len(eval_dataset_merge)}")

    # Data collator
    logger.info(f"Collator data")
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    logger.info(f"load metric")
    metric = evaluate.load("sacrebleu")
    logger.info(f"load metric over")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    # trainer = M2MSeq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset_merge if training_args.do_train else None,
    #     eval_dataset=eval_dataset_merge if training_args.do_eval else None,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    # )
    optimizer = None
    # optimizers = (None, None)
    # if model_args.use_badam:
    #     try:
    #         from badam import BlockOptimizer
    #     except ImportError:
    #         # raise ImportError("Unable to import badam module. Please install badam or disable its usage.")
    #         logger.info("Unable to import badam module. Please install badam or disable its usage.")
    #
    #     # Optimizer
    #     # Split weights in two groups, one with weight decay and the other not.
    #     no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #             "weight_decay": training_args.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     original_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    #
    #     # before training, add this line to wrap the original optimizer
    #     optimizer = BlockOptimizer(
    #         base_optimizer=original_optimizer,  # can be any torch.Optimizer
    #         named_parameters_list=list(model.named_parameters()),
    #         switch_block_every=100,
    #         # switch to the new block every 50 updates, the $K$ Adam steps in paper. It can be set adaptively by $K = n/(BD)$, where $n$ is the number of training data points, $B$ is the batch size, and $D$ is the number of blocks in BAdam; see "Hyperparameter Suggestion" section for a detailed explaination about setting this hyperparameter.
    #         switch_mode="random",
    #         # update order of blocks, one can choose "random" (random reshuffling update order), "ascending" (update from input layer to output layer), or "descending" (update from output layer to input layer). The default is "random".
    #         verbose=2,  # information level, will print trainable parameters when setting to 2
    #         block_prefix_list = None
    #     )

        # # Scheduler and math around the number of training steps.
        # overrode_max_train_steps = False
        # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps) # train_dataloader获取不到
        # if training_args.max_train_steps is None:
        #     training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        #     overrode_max_train_steps = True
        #
        # lr_scheduler = get_scheduler(
        #     name=training_args.lr_scheduler_type,
        #     optimizer=optimizer,
        #     num_warmup_steps=training_args.num_warmup_steps,
        #     num_training_steps=training_args.max_train_steps,
        # )
        #
        # optimizers = (optimizer, lr_scheduler)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_merge if training_args.do_train else None,
        eval_dataset=eval_dataset_merge if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        # optimizers = (optimizer, None)
    )

    if model_args.use_badam:
        try:
            from badam import BlockOptimizer,clip_grad_norm_for_sparse_tensor

        except ImportError:
            # raise ImportError("Unable to import badam module. Please install badam or disable its usage.")
            logger.info("Unable to import badam module. Please install badam or disable its usage.")

        # Optimizer
        trainer.create_optimizer_and_scheduler(num_training_steps=training_args.max_steps)
        original_optimizer = trainer.optimizer

        # before training, add this line to wrap the original optimizer
        trainer.optimizer = BlockOptimizer(
            base_optimizer=original_optimizer,  # can be any torch.Optimizer
            named_parameters_list=list(model.named_parameters()),
            switch_block_every=100,
            # switch to the new block every 50 updates, the $K$ Adam steps in paper. It can be set adaptively by $K = n/(BD)$, where $n$ is the number of training data points, $B$ is the batch size, and $D$ is the number of blocks in BAdam; see "Hyperparameter Suggestion" section for a detailed explaination about setting this hyperparameter.
            switch_mode="random",
            # update order of blocks, one can choose "random" (random reshuffling update order), "ascending" (update from input layer to output layer), or "descending" (update from output layer to input layer). The default is "random".
            verbose=2,  # information level, will print trainable parameters when setting to 2
            block_prefix_list = None
        )

        trainer.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, trainer.accelerator)



    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset_merge)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset_merge))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset_merge)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset_merge))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # if training_args.do_predict:
    #     logger.info("*** Predict ***")
    #
    #     predict_results = trainer.predict(
    #         predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
    #     )
    #     metrics = predict_results.metrics
    #     max_predict_samples = (
    #         data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    #     )
    #     metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
    #
    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)
    #
    #     if trainer.is_world_process_zero():
    #         if training_args.predict_with_generate:
    #             predictions = tokenizer.batch_decode(
    #                 predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #             predictions = [pred.strip() for pred in predictions]
    #             output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    #             with open(output_prediction_file, "w", encoding="utf-8") as writer:
    #                 writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
