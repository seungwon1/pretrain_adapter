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
import argparse
import random
import pickle
import dataclasses
import logging
import os


import sys
import json
import torch
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Optional
import numpy as np
from datasets import load_dataset, load_metric #for glue tasks

#os.environ["CUDA_VISIBLE_DEVICES"]= "2"

from transformers import (
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoModelWithHeads,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    default_data_collator,
)
from transformers.adapter_config import PfeifferConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    MultiLingAdapterArguments,
    TrainingArguments,
    set_seed,
)
import transformers
from trainer.trainer import MyTrainer
from trainer.trainer_callback import EarlyStoppingCallback

transformers.logging.set_verbosity_info()
logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

glue_list = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

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
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    train_fusion: bool = field(default=False, metadata={"help": "whether train adapter fusion model or not."})
    train_adapter_wop: bool = field(default=False, metadata={"help": "whether train adapter without pretraining."})
    fusion_adapter_path1: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path2: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path2: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path3: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path4: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path5: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path6: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path7: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path8: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the task to train on"},
    )

    data_dir: Optional[str] = field(default=None, metadata={"help": "data dir."})

    metric: Optional[str] = field(default="macro", metadata={"help": "evaluation metric."})

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    sanity_check: bool = field(default=False, metadata={"help": "saved mdoels for sanity check."})

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    patience_factor: int = field(
        default=10,
        metadata={"help": "The number of epochs to be considered to execute early stopping."}
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
    data_dir = data_args.data_dir
    #print(data_dir)

    # datasets setup
    # 1. don't stop pre-training

    # label to id, set same as in don't stop repo
    if data_dir is not None:
        label_to_id_ft = {}
        if "chemprot" in data_dir:
            label_to_id_ft = {
                "INHIBITOR": 0,
                "SUBSTRATE": 1,
                "INDIRECT-DOWNREGULATOR": 2,
                "INDIRECT-UPREGULATOR": 3,
                "ACTIVATOR": 4,
                "ANTAGONIST": 5,
                "PRODUCT-OF": 6,
                "AGONIST": 7,
                "DOWNREGULATOR": 8,
                "UPREGULATOR": 9,
                "AGONIST-ACTIVATOR": 10,
                "SUBSTRATE_PRODUCT-OF": 11,  ## test set modify needed
                "AGONIST-INHIBITOR": 12,
            }
            num_labels = 13

        elif "rct" in data_dir:
            label_to_id_ft = {"METHODS": 0, "RESULTS": 1, "CONCLUSIONS": 2, "BACKGROUND": 3, "OBJECTIVE": 4}
            num_labels = 5

        elif "citation" in data_dir or "acl" in data_dir:
            label_to_id_ft = {
                "Background": 0,
                "Uses": 1,
                "CompareOrContrast": 2,
                "Motivation": 3,
                "Extends": 4,
                "Future": 5,
            }
            num_labels = 6

        elif "scierc" in data_dir or "sciie" in data_dir:
            label_to_id_ft = {
                "USED-FOR": 0,
                "CONJUNCTION": 1,
                "EVALUATE-FOR": 2,
                "HYPONYM-OF": 3,
                "PART-OF": 4,
                "FEATURE-OF": 5,
                "COMPARE": 6,
            }
            num_labels = 7

        elif "hyper" in data_dir:
            label_to_id_ft = {"false": 0, "true": 1}
            num_labels = 2

        elif "ag" in data_dir or "agnews" in data_dir:
            label_to_id_ft = {1: 0, 2: 1, 3: 2, 4: 3}
            num_labels = 4

        elif "amazon" in data_dir or "helpful" in data_dir:
            label_to_id_ft = {"helpful": 0, "unhelpful": 1}
            num_labels = 2

        elif "imdb" in data_dir:
            label_to_id_ft = {0: 0, 1: 1}
            num_labels = 2

        else:
            assert False, (
                "Data_dir not in [chemprot, rct-20k, rct-sample, citation_intent(ACL_ARC), "
                "sciie(SCIERC), ag(AGNEWS), hyperpartisan_new (HYPERPARTISAN), imdb, amazon(helpfulness)] "
            )

    output_mode = "classification"


    # 2. GLUE tasks
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    elif data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
            "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )

    # Labels
    label_list = None
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    ) # use_fast=model_args.use_fast_tokenizer,

    model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    #if data_args.sanity_check:
    #    bsw = {}
    #    for i in model.state_dict():
    #        bsw[i] = model.state_dict()[i]
    #    np.save(training_args.output_dir + 'after_model_loaded.npy', bsw)  # Just used for sanity check, (500MB)

    if model_args.train_adapter_wop:
        model.add_adapter(data_args.task_name, AdapterType.text_task,
                          config=PfeifferConfig(reduction_factor=adapter_args.adapter_reduction_factor))
        model.train_adapter([data_args.task_name])
        if data_args.task_name in glue_list:
            model.add_classification_head(data_args.task_name, num_labels=num_labels,
                                          id2label={i: v for i, v in enumerate(label_list)} if num_labels > 1 else None,
                                          )

        else:
            model.add_classification_head(data_args.task_name, num_labels=num_labels)
        # Set the adapters to be used in every forward pass
        model.set_active_adapters([[data_args.task_name]])

    elif adapter_args.train_adapter:
        model.train_adapter(model.config.adapters.adapter_list(AdapterType.text_lang)[0])  ###model.train_adapter([task_name])
        if data_args.task_name in glue_list:
            model.add_classification_head(data_args.task_name, num_labels=num_labels,
                                          id2label={i: v for i, v in enumerate(label_list)} if num_labels > 1 else None,
                                          )
        else:
            model.add_classification_head(model.config.adapters.adapter_list(AdapterType.text_lang)[0], num_labels=num_labels)
        # Set the adapters to be used in every forward pass
        model.set_active_adapters(model.config.adapters.adapter_list(AdapterType.text_lang)[0])

    elif model_args.train_fusion:
        fusion_path = []
        fusion_path.append(model_args.fusion_adapter_path1)
        fusion_path.append(model_args.fusion_adapter_path2)
        fusion_path.append(model_args.fusion_adapter_path3)
        fusion_path.append(model_args.fusion_adapter_path4)
        fusion_path.append(model_args.fusion_adapter_path5)
        fusion_path.append(model_args.fusion_adapter_path6)
        fusion_path.append(model_args.fusion_adapter_path7)
        fusion_path.append(model_args.fusion_adapter_path8)
        while "" in fusion_path:
            fusion_path.remove("")

        for each in fusion_path:
            model.load_adapter(each, "text_lang",
                               config=PfeifferConfig(reduction_factor=adapter_args.adapter_reduction_factor), with_head=False)

        ADAPTER_SETUP = [
            [
                list(model.config.adapters.adapters.keys())[i]
                for i in range(len(list(model.config.adapters.adapters.keys())))
            ]
        ]

        # Add a fusion layer and tell the model to train fusion
        logger.info(f"Using adapter fusion with the following setup {ADAPTER_SETUP}")
        logger.info(f"Model adapters = {ADAPTER_SETUP}")
        model.add_fusion(ADAPTER_SETUP[0], "dynamic")
        model.train_fusion(ADAPTER_SETUP)
        if data_args.task_name in glue_list:
            model.add_classification_head(data_args.task_name, num_labels=num_labels,
                                          id2label={i: v for i, v in enumerate(label_list)} if num_labels > 1 else None,
                                          )
        else:
            model.add_classification_head(data_args.task_name, num_labels=num_labels)

    else:
        if data_args.task_name in glue_list:
            model.add_classification_head(data_args.task_name, num_labels=num_labels,
                                          id2label={i: v for i, v in enumerate(label_list)} if num_labels > 1 else None,
                                          )
        else:
            model.add_classification_head(data_args.task_name, num_labels=num_labels)

    #if data_args.sanity_check:
    #    bsw = {}
    #    for i in model.state_dict():
    #        bsw[i] = model.state_dict()[i]
    #    np.save(training_args.output_dir + 'after_add_heads.npy', bsw) # Just used for sanity check, (500MB)

    # if data_args.sanity_check:
    #    bsw = {}
    #    for i in model.state_dict():
    #        bsw[i] = model.state_dict()[i]
    #    np.save('after_fusion_merged.npy', bsw)  # Just used for sanity check, (500MB)

    # preprocess data for glue tasks
    if data_args.task_name in glue_list:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
            max_length = data_args.max_seq_length
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False
            max_length = None

        label_to_id = None
        if (
                model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
                and data_args.task_name is not None
                and is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
            else:
                logger.warn(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif data_args.task_name is None:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [label_to_id[l] for l in examples["label"]]
            return result

        datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

        train_dataset = datasets["train"]
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.task_name is not None:
            test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

        # Get the metric function
        if data_args.task_name is not None:
            metric = load_metric("glue", data_args.task_name)
            #print('metric:' , metric)


        def compute_metrics_ft(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    else:
        ### load dataset, define compute_metrics ###
        data_dir = data_args.data_dir
        train_texts = []
        train_labels = []
        val_texts = []
        val_labels = []
        test_texts = []
        test_labels = []

        if data_dir[-1] != "/":
            data_dir += "/"

        for each in ["train", "dev", "test"]:
            with open(data_dir + each + ".jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if each == "train":
                        train_texts.append(json.loads(line)["text"])
                        train_labels.append(label_to_id_ft[json.loads(line)["label"]])
                    elif each == "dev":
                        val_texts.append(json.loads(line)["text"])
                        val_labels.append(label_to_id_ft[json.loads(line)["label"]])
                    else:
                        test_texts.append(json.loads(line)["text"])
                        test_labels.append(label_to_id_ft[json.loads(line)["label"]])

        train_encodings = tokenizer(train_texts, padding="max_length", max_length=512, truncation=True)
        val_encodings = tokenizer(val_texts, padding="max_length", max_length=512, truncation=True)
        test_encodings = tokenizer(test_texts, padding="max_length", max_length=512, truncation=True)

        class FTDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = FTDataset(train_encodings, train_labels)
        eval_dataset = FTDataset(val_encodings, val_labels)
        test_dataset = FTDataset(test_encodings, test_labels)

        def compute_metrics_ft(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average=data_args.metric, labels=[i for i in range(num_labels)], zero_division=0)
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    #############################

    # Initialize our Trainer
    if not adapter_args.train_adapter and not model_args.train_fusion:
        trainer = MyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_ft,
            do_save_full_model=True,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience_factor)]
        )
        if data_args.task_name in glue_list:
            trainer = MyTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                do_save_full_model=True,
                compute_metrics=compute_metrics_ft,
                tokenizer=tokenizer,
                # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
                data_collator=default_data_collator if data_args.pad_to_max_length else None,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience_factor)]
            )

    else:
        save_full = True

        if adapter_args.train_adapter:
            trainer = MyTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics_ft,
                do_save_full_model=save_full,
                do_save_adapters=adapter_args.train_adapter,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience_factor)]
            )
            if data_args.task_name in glue_list:
                trainer = MyTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics_ft,
                    tokenizer=tokenizer,
                    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
                    data_collator=default_data_collator if data_args.pad_to_max_length else None,
                    do_save_full_model=save_full,
                    do_save_adapters=adapter_args.train_adapter,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience_factor)]
                )



        else:
            trainer = MyTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics_ft,
                do_save_full_model=save_full,
                do_save_adapter_fusion=True,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience_factor)]
            )

            if data_args.task_name in glue_list:
                trainer = MyTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics_ft,
                    do_save_full_model=save_full,
                    do_save_adapter_fusion=True,
                    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
                    data_collator=default_data_collator if data_args.pad_to_max_length else None,
                    tokenizer=tokenizer,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience_factor)]
                )

    # Training
    if training_args.do_train:
        # save model before training
        if data_args.sanity_check:
            bsw = {}
            for i in model.state_dict():
                bsw[i] = model.state_dict()[i]
            np.save(training_args.output_dir + "before_training.npy", bsw)  # Just used for sanity check, (500MB)

        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        # after finishing traning, save keys
        if data_args.sanity_check:
            bsw = {}
            for i in model.state_dict():
                bsw[i] = model.state_dict()[i]
            np.save(training_args.output_dir + "after_training.npy", bsw)  # Just used for sanity check, (500MB)

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation: Validation is done during training (e.g., after each epoch)

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]

        eval_datasets = [eval_dataset]

        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        if data_args.task_name in glue_list:
            for eval_dataset, task in zip(eval_datasets, tasks):
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)

                output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_eval_file, "w") as writer:
                        logger.info(f"***** Eval results {task} *****")
                        for key, value in eval_result.items():
                            logger.info(f"  {key} = {value}")
                            writer.write(f"{key} = {value}\n")

                eval_results.update(eval_result)

        else:
            for eval_dataset in eval_datasets:
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{data_args.task_name}.txt")

                if trainer.is_world_master():
                    with open(output_eval_file, "w") as writer:
                        logger.info(
                            "***** Eval results {} *****".format(data_args.task_name)
                        )  ####eval_dataset.args.task_name))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))

                eval_results.update(eval_result)


    if training_args.do_predict:
        logging.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        if data_args.task_name in glue_list:
            for test_dataset, task in zip(test_datasets, tasks):
                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                test_dataset.remove_columns_("label")
                predictions = trainer.predict(test_dataset=test_dataset).predictions
                predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

                output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_test_file, "w") as writer:
                        logger.info(f"***** Test results {task} *****")
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if is_regression:
                                writer.write(f"{index}\t{item:3.3f}\n")
                            else:
                                item = label_list[item]
                                writer.write(f"{index}\t{item}\n")

        else:
            for test_dataset in test_datasets:
                test_eval_result = trainer.predict(test_dataset=test_dataset).metrics
                output_test_eval_file = os.path.join(training_args.output_dir, f"test_results_{data_args.task_name}.txt")

                if trainer.is_world_master():
                    with open(output_test_eval_file, "w") as writer:
                        logger.info(
                            "***** Test results {} *****".format(data_args.task_name)
                        )  ####eval_dataset.args.task_name))
                        for key, value in test_eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))

    if data_args.task_name in glue_list:
        return predictions
    else:
        return test_eval_result


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()