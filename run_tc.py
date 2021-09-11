#!/usr/bin/env python
# coding=utf-8
"""
Finetuning multi-lingual models on SJP (e.g. Bert, DistilBERT, XLM).
Adapted from `examples/text-classification/run_glue.py`
"""
import faulthandler
import glob
import json
import logging
import math
import os
import random
import sys
from json import JSONEncoder

import dataclasses
from pathlib import Path

import wandb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight

import torch
from sklearn.utils.extmath import softmax
from torch.cuda.amp import autocast
from torch import nn
from torch.nn import CrossEntropyLoss

from datasets import load_dataset, concatenate_datasets
import transformers
import transformers.adapters.composition as ac
from transformers import (
    AdapterConfig,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    EarlyStoppingCallback,
    HfArgumentParser,
    LongformerForSequenceClassification,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

import LongBert
import Longformer
from HierarchicalBert import HierarchicalBert
from data_arguments import DataArguments
from model_arguments import ModelArguments, long_input_bert_types

os.environ['TOKENIZERS_PARALLELISM'] = "True"
os.environ['WANDB_PROJECT'] = 'SwissJudgementPrediction'
os.environ['WANDB_MODE'] = "online"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # use this when debugging

# Will error if the minimal version of transformers is not installed. Remove at your own risks.
check_min_version("4.8.2")

logger = logging.getLogger(__name__)

faulthandler.enable()

model_types = ['distilbert', 'bert', 'roberta', 'camembert']
languages = ['de', 'fr', 'it']

logger.warning("This script only supports PyTorch models!")


def main():
    # See all possible arguments in src/transformers/training_args.py or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # for better charts when we have a group run with multiple seeds
    os.environ["WANDB_RUN_GROUP"] = training_args.run_name[:-2]  # remove last two characters "-{seed}"

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Save all params for better reproducibility
    experiment_params = {
        "model_args": dataclasses.asdict(model_args),
        "data_args": dataclasses.asdict(data_args),
        "training_args": dataclasses.asdict(training_args),
        "adapter_args": dataclasses.asdict(adapter_args),
    }

    class SimpleEncoder(JSONEncoder):
        def default(self, o):
            try:
                return o.__dict__
            except AttributeError:
                return str(o)

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{training_args.output_dir}/experiment_params.json', 'w') as file:
        json.dump(experiment_params, file, indent=4, cls=SimpleEncoder)

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    langs = [model_args.evaluation_language]
    if model_args.evaluation_language == 'all':
        langs = languages
        train_datasets = []
        eval_datasets = []
        predict_datasets = []

    assert len(langs) > 0
    for lang in langs:
        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if training_args.do_train:
            train_dataset = load_dataset("csv", data_files={"train": f'data/{lang}/train.csv'})['train']

        if training_args.do_eval:
            eval_dataset = load_dataset("csv", data_files={"validation": f'data/{lang}/val.csv'})['validation']

        if training_args.do_predict:
            predict_dataset = load_dataset("csv", data_files={"test": f'data/{lang}/test.csv'})['test']

        if data_args.test_on_sub_datasets:
            special_splits = dict()
            for file in glob.glob(f'data/{lang}/special_splits/*/*.csv'):
                experiment = Path(file).parent.stem
                part = Path(file).stem.split("-")[1]
                if experiment not in special_splits:
                    special_splits[experiment] = dict()
                special_splits[experiment][part] = load_dataset("csv", data_files={"test": file})['test']

        # Labels: they will get overwritten if there are multiple languages
        with open(f'data/{lang}/labels.json', 'r') as f:
            label_dict = json.load(f)
            label_dict['id2label'] = {int(k): v for k, v in label_dict['id2label'].items()}
            label_dict['label2id'] = {k: int(v) for k, v in label_dict['label2id'].items()}
            label_list = list(label_dict["label2id"].keys())
        num_labels = len(label_list)

        if model_args.evaluation_language == 'all':
            train_datasets.append(train_dataset)
            eval_datasets.append(eval_dataset)
            predict_datasets.append(predict_dataset)

    if model_args.evaluation_language == 'all':
        train_dataset = concatenate_datasets(train_datasets)
        eval_dataset = concatenate_datasets(eval_datasets)
        predict_dataset = concatenate_datasets(predict_datasets)

    if data_args.problem_type == 'multi_label_classification':
        mlb = MultiLabelBinarizer().fit([label_list])

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    finetuning_task = "text-classification"
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_dict["id2label"],
        label2id=label_dict["label2id"],
        finetuning_task=finetuning_task,
        problem_type=data_args.problem_type,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    max_length = data_args.max_seq_length
    if model_args.long_input_bert_type == 'hierarchical':
        max_segment_length = 512  # because RoBERTa has 514 max_seq_length and not 512
        max_segments = math.ceil(max_length / max_segment_length)
        # we need to make space for adding the CLS and SEP token for each segment
        max_length -= max_segments * 2

    def get_encoder_and_classifier(model):
        if config.model_type not in model_types:
            raise ValueError(f"{config.model_type} is not supported. "
                             f"Please use one of the supported model types {model_types}")

        if config.model_type == 'distilbert':
            encoder = model.distilbert
        if config.model_type == 'bert':
            encoder = model.bert
        if config.model_type in ['camembert', 'xlm-roberta']:
            encoder = model.roberta

        classifier = model.classifier
        # classifier = model.heads
        return encoder, classifier

    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # model = BertForSequenceClassification() # for untrained model

        # TODO use more flexible AutoModelWithHeads for better adapter support
        # model = AutoModelWithHeads.from_pretrained(
        #    model_args.model_name_or_path,
        #    from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #    config=config,
        #    cache_dir=model_args.cache_dir,
        #    revision=model_args.model_revision,
        #    use_auth_token=True if model_args.use_auth_token else None,
        # )
        # model.add_classification_head(
        #    data_args.task_name,
        #    num_labels=num_labels,
        #    id2label=label_dict['id2label'],
        # )

        if model_args.use_adapters:
            # Setup adapters
            if adapter_args.train_adapter:
                task_name = data_args.task_name
                # check if adapter already exists, otherwise add it
                if task_name not in model.config.adapters:
                    # resolve the adapter config
                    adapter_config = AdapterConfig.load(
                        adapter_args.adapter_config,
                        non_linearity=adapter_args.adapter_non_linearity,
                        reduction_factor=adapter_args.adapter_reduction_factor,
                    )
                    # load a pre-trained from Hub if specified
                    if adapter_args.load_adapter:
                        model.load_adapter(
                            adapter_args.load_adapter,
                            config=adapter_config,
                            load_as=task_name,
                        )
                    # otherwise, add a fresh adapter
                    else:
                        model.add_adapter(task_name, config=adapter_config)
                # optionally load a pre-trained language adapter
                if adapter_args.load_lang_adapter:
                    # resolve the language adapter config
                    lang_adapter_config = AdapterConfig.load(
                        adapter_args.lang_adapter_config,
                        non_linearity=adapter_args.lang_adapter_non_linearity,
                        reduction_factor=adapter_args.lang_adapter_reduction_factor,
                    )
                    # load the language adapter from Hub
                    lang_adapter_name = model.load_adapter(
                        adapter_args.load_lang_adapter,
                        config=lang_adapter_config,
                        load_as=adapter_args.language,
                    )
                else:
                    lang_adapter_name = None
                # Freeze all model weights except of those of this adapter
                model.train_adapter([task_name])
                # Set the adapters to be used in every forward pass
                if lang_adapter_name:
                    model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
                else:
                    model.set_active_adapters(task_name)
            else:
                if adapter_args.load_adapter or adapter_args.load_lang_adapter:
                    raise ValueError(
                        "Adapters can only be loaded in adapters training mode."
                        "Use --train_adapter to enable adapter training"
                    )

        if model_args.long_input_bert_type in long_input_bert_types:
            if model_args.long_input_bert_type not in ['bigbird']: # nothing to do for bigbird
                encoder, classifier = get_encoder_and_classifier(model)

                if model_args.long_input_bert_type == 'hierarchical':
                    long_input_bert = HierarchicalBert(encoder,
                                                       max_segments=max_segments,
                                                       max_segment_length=max_segment_length,
                                                       cls_token_id=tokenizer.cls_token_id,
                                                       sep_token_id=tokenizer.sep_token_id,
                                                       device=training_args.device,
                                                       seg_encoder_type='lstm')

                if model_args.long_input_bert_type == 'long':
                    long_input_bert = LongBert.resize_position_embeddings(encoder,
                                                                          max_length=max_length,
                                                                          device=training_args.device)

                if model_args.long_input_bert_type in ['hierarchical', 'long']:
                    if config.model_type == 'distilbert':
                        model.distilbert = long_input_bert

                    if config.model_type == 'bert':
                        model.bert = long_input_bert

                    if config.model_type in ['camembert', 'xlm-roberta']:
                        model.roberta = long_input_bert
                        if model_args.long_input_bert_type == 'hierarchical':
                            dense = nn.Linear(config.hidden_size, config.hidden_size)
                            dense.load_state_dict(classifier.dense.state_dict())  # load weights
                            dropout = nn.Dropout(config.hidden_dropout_prob).to(training_args.device)
                            out_proj = nn.Linear(config.hidden_size, config.num_labels).to(training_args.device)
                            out_proj.load_state_dict(classifier.out_proj.state_dict())  # load weights
                            model.classifier = nn.Sequential(dense, dropout, out_proj).to(training_args.device)

                if last_checkpoint or not training_args.do_train:
                    # Make sure we really load all the weights after we modified the models
                    model_path = f'{model_args.model_name_or_path}/model.bin'
                    logger.info(f"loading file {model_path}")
                    model.load_state_dict(torch.load(model_path))

                # NOTE: longformer had quite bad results (probably something is off here)
                if training_args.do_train and model_args.long_input_bert_type == 'longformer':
                    encoder = Longformer.convert2longformer(encoder,
                                                            max_seq_length=max_length,
                                                            attention_window=128)
                    model = LongformerForSequenceClassification(config)
                    model.longformer.encoder.load_state_dict(encoder.encoder.state_dict())  # load weights
                    model.classifier.out_proj.load_state_dict(classifier.state_dict())  # load weights

        return model

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        # IMPORTANT: Can lead to problem with HierarchicalBert
        padding = "longest"

    add_special_tokens = True
    if model_args.long_input_bert_type == 'hierarchical':
        add_special_tokens = False  # because we split it internally and then add the special tokens ourselves

    def preprocess_function(batch):
        # Tokenize the texts
        tokenized = tokenizer(batch["text"], padding=padding, truncation=True,
                              max_length=max_length, add_special_tokens=add_special_tokens,
                              return_token_type_ids=True)

        # Map labels to IDs
        if data_args.problem_type == 'multi_label_classification':
            tokenized["label"] = [mlb.transform([eval(labels)])[0] for labels in batch["label"]]
        if data_args.problem_type == 'single_label_classification':
            if label_dict["label2id"] is not None and "label" in batch:
                tokenized["label"] = [label_dict["label2id"][l] for l in batch["label"]]
        return tokenized

    def preprocess_dataset(dataset):
        return dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=dataset.column_names,
        )

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = preprocess_dataset(train_dataset)
        # Log a random sample from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = preprocess_dataset(eval_dataset)

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = preprocess_dataset(predict_dataset)

    if data_args.test_on_sub_datasets:
        for experiment, parts in special_splits.items():
            for part, dataset in parts.items():
                special_splits[experiment][part] = preprocess_dataset(dataset)

    def labels_to_bools(labels):
        return [tl == 1 for tl in labels]

    def preds_to_bools(preds):
        return [pl > model_args.prediction_threshold for pl in preds]

    def process_results(preds, labels):
        preds = preds[0] if isinstance(preds, tuple) else preds
        probs = softmax(preds)
        if data_args.problem_type == 'multi_label_classification':
            # for multi_label_classification we need boolean arrays for each example
            preds, labels = preds_to_bools(preds), labels_to_bools(labels)
        if data_args.problem_type == 'single_label_classification':
            preds = np.argmax(preds, axis=1)
        return preds, labels, probs

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        labels = p.label_ids
        preds = p.predictions
        preds, labels, probs = process_results(preds, labels)

        accuracy = accuracy_score(labels, preds)
        # weighted averaging is a better evaluation metric for imbalanced label distributions
        precision, recall, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        f1_micro = f1_score(labels, preds, average='micro')
        f1_macro = f1_score(labels, preds, average='macro')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
        }

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if training_args.do_train and model_args.label_imbalance_method == 'class_weights':
        lbls = [item['label'] for item in train_dataset]
        # compute class weights based on label distribution
        class_weight = compute_class_weight('balanced', classes=np.unique(lbls), y=lbls)
        class_weight = torch.tensor(class_weight, dtype=torch.float32, device=training_args.device)  # create tensor

        class CustomTrainer(Trainer):
            # adapt loss function to combat label imbalance
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = CrossEntropyLoss(weight=class_weight)
                with autocast():  # necessary for correct pytorch types. May not work for tensorflow
                    loss = loss_fct(logits, labels)
                    # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer_init = CustomTrainer
    else:
        trainer_init = Trainer

    # NOTE: This is not optimized for multiclass classification
    if training_args.do_train and model_args.label_imbalance_method in ['oversampling', 'undersampling']:
        label_datasets = dict()
        minority_len, majority_len = len(train_dataset), 0
        for label_id in label_dict['id2label'].keys():
            label_datasets[label_id] = train_dataset.filter(lambda item: item['label'] == label_id)
            if len(label_datasets[label_id]) < minority_len:
                minority_len = len(label_datasets[label_id])
                minority_id = label_id
            if len(label_datasets[label_id]) > majority_len:
                majority_len = len(label_datasets[label_id])
                majority_id = label_id

    if training_args.do_train and model_args.label_imbalance_method == 'oversampling':
        logger.info("Oversampling the minority class")
        datasets = [train_dataset]
        num_full_minority_sets = int(majority_len / minority_len)
        for i in range(num_full_minority_sets - 1):  # -1 because one is already included in the trainig dataset
            datasets.append(label_datasets[minority_id])

        remaining_minority_samples = majority_len % minority_len
        random_ids = np.random.choice(minority_len, remaining_minority_samples, replace=False)
        datasets.append(label_datasets[minority_id].select(random_ids))
        train_dataset = concatenate_datasets(datasets)

    if training_args.do_train and model_args.label_imbalance_method == 'undersampling':
        logger.info("Undersampling the majority class")
        random_ids = np.random.choice(majority_len, minority_len, replace=False)
        # just select only the number of minority samples from the majority class
        label_datasets[majority_id] = label_datasets[majority_id].select(random_ids)
        train_dataset = concatenate_datasets(list(label_datasets.values()))

    # Initialize our Trainer
    trainer = trainer_init(
        model=model_init() if not data_args.tune_hyperparams else None,
        model_init=model_init if data_args.tune_hyperparams else None,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    # Hyperparameter Tuning
    if data_args.tune_hyperparams:
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
            }

        best_trial = trainer.hyperparameter_search(
            hp_space=optuna_hp_space,
            direction="maximize",
            backend="optuna",  # ray/optuna
            n_trials=10,  # number of trials
            # Choose among many libraries:
            # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
            # search_alg=HyperOptSearch(),
            # Choose among schedulers:
            # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
            # scheduler=AsyncHyperBand()
        )
        logger.info(best_trial)
        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)

    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        # save entire model ourselves just to be safe
        torch.save(trainer.model.state_dict(), f'{training_args.output_dir}/model.bin')

        if model_args.long_input_bert_type == 'longformer':
            # Amend configuration file
            config_path = f'{training_args.output_dir}/config.json'
            with open(config_path) as config_file:
                configuration = json.load(config_file)
                configuration['model_type'] = "longformer"
            with open(config_path, 'w') as config_file:
                json.dump(configuration, config_file, indent=4)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def remove_metrics(metrics, split):
        # remove unnecessary values to make overview nicer in wandb
        metrics.pop(f"{split}_loss")
        metrics.pop(f"{split}_runtime")
        metrics.pop(f"{split}_steps_per_second")
        metrics.pop(f"{split}_samples_per_second")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        remove_metrics(metrics, 'eval')

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    def predict(predict_dataset):
        preds, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="test")
        remove_metrics(metrics, 'test')

        preds, labels, probs = process_results(preds, labels)
        return preds, labels, probs, metrics

    def write_report_section(writer, title, content):
        writer.write(f"{title}\n")
        writer.write("=" * 75 + "\n\n")
        writer.write(content)
        writer.write("\n" * 3)

    def pred2label(pred):
        if data_args.problem_type == 'multi_label_classification':
            return mlb.inverse_transform(np.array([pred]))[0]
        if data_args.problem_type == 'single_label_classification':
            return label_dict["id2label"][pred]

    def write_reports(base_dir, preds, labels, probs, wandb_prefix):
        if trainer.is_world_process_zero():
            correct_confidences, incorrect_confidences = [], []
            # write predictions to csv
            result = {"index": [], "prediction": [], "label": [], "is_correct": [], "confidence": [], "error": []}
            for index, pred in enumerate(preds):
                confidence = probs[index][pred]
                is_correct = pred == labels[index]
                error = 1 - confidence if is_correct else confidence
                if is_correct:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)
                result['index'].append(index)
                result['prediction'].append(pred2label(pred))
                result['label'].append(pred2label(labels[index]))
                result['is_correct'].append(is_correct)
                result['confidence'].append(confidence)
                result['error'].append(error)
            pd.DataFrame.from_dict(result).to_csv(f'{base_dir}/predictions.csv')

            # write confidences to csv
            confidences = {"correct": {"mean": np.mean(correct_confidences), "std": np.std(correct_confidences)},
                           "incorrect": {"mean": np.mean(incorrect_confidences), "std": np.std(incorrect_confidences)}}
            pd.DataFrame.from_dict(confidences, orient='index').to_csv(f'{base_dir}/confidences.csv')

            if "wandb" in training_args.report_to:
                wandb.log({
                    f"{wandb_prefix}correct_mean": confidences['correct']['mean'],
                    f"{wandb_prefix}correct_std": confidences['correct']['std'],
                    f"{wandb_prefix}incorrect_mean": confidences['incorrect']['mean'],
                    f"{wandb_prefix}incorrect_std": confidences['incorrect']['std'],
                })

            # write report file
            with open(f'{base_dir}/prediction_report.txt', "w") as writer:
                if data_args.problem_type == 'multi_label_classification':
                    title = "Multilabel Confusion Matrix"
                    matrices = multilabel_confusion_matrix(labels, preds)
                if data_args.problem_type == 'single_label_classification':
                    title = "Singlelabel Confusion Matrix"
                    matrices = [confusion_matrix(labels, preds)]
                content = "reading help:\nTN FP\nFN TP\n\n"
                for i in range(len(matrices)):
                    content += f"{label_list[i]}\n{str(matrices[i])}\n"
                write_report_section(writer, title, content)

                report = classification_report(labels, preds, digits=4,
                                               target_names=label_list, labels=list(label_dict['label2id'].values()))
                write_report_section(writer, "Classification Report", str(report))

                content = f"correct:\t{confidences['correct']['mean']}%\t+/-\t{confidences['correct']['std']}\n" \
                          f"incorrect:\t{confidences['incorrect']['mean']}%\t+/-\t{confidences['incorrect']['std']}\n"
                write_report_section(writer, "Mean confidence of predictions", content)

    # Prediction
    if training_args.do_predict and not data_args.tune_hyperparams:
        logger.info("*** Predict ***")
        preds, labels, probs, metrics = predict(predict_dataset)

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["test_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # rename metrics so that they appear in separate section in wandb and filter out unnecessary ones
        if "wandb" in training_args.report_to:
            prefix = "test/"
            metrics = {k.replace("test_", prefix): v for k, v in metrics.items()
                       if "mem" not in k and k != "test_samples"}
            wandb.log(metrics)  # log test metrics to wandb

        write_reports(training_args.output_dir, preds, labels, probs, prefix)

    # Sub Datasets
    if data_args.test_on_sub_datasets:
        logger.info("*** Special Splits ***")
        for experiment, parts in special_splits.items():
            for part, dataset in parts.items():
                if len(dataset) >= 1:  # we need at least one example
                    base_dir = Path(training_args.output_dir) / experiment / part
                    base_dir.mkdir(parents=True, exist_ok=True)
                    preds, labels, probs, metrics = predict(dataset)
                    if "wandb" in training_args.report_to:
                        prefix = f"{experiment}/{part}/"
                        metrics = {k.replace("test_", prefix): v for k, v in metrics.items()}
                        metrics[f'{prefix}support'] = len(dataset)
                        wandb.log(metrics)  # log test metrics to wandb
                    write_reports(base_dir, preds, labels, probs, prefix)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": finetuning_task}
        if data_args.task_name is not None:
            kwargs["language"] = model_args.evaluation_language
            kwargs["dataset_tags"] = "sjp"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"SJP {data_args.task_name.upper()}"

        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
