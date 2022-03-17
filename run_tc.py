#!/usr/bin/env python
# coding=utf-8
"""
Finetuning/Adapting multi-lingual models on SJP (e.g. Bert, RoBERTa DistilBERT, XLM).
Adapted from `examples/text-classification/run_glue.py`
"""
import faulthandler
import glob
import json
import logging
import os
import pprint
import random
import sys
from collections import OrderedDict

import dataclasses
import shutil
import yaml
from pathlib import Path
from enum import Enum

import wandb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score, matthews_corrcoef
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight

import torch
from sklearn.utils.extmath import softmax
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss

from datasets import load_dataset, concatenate_datasets
import transformers
import transformers.adapters.composition as ac
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    EarlyStoppingCallback,
    HfArgumentParser,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed, TrainerCallback, XLMRobertaTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from root import DATA_DIR, AUGMENTED_DIR
from utils.custom_callbacks import CustomWandbCallback
from long import LongBert
from arguments.data_arguments import DataArguments, ProblemType, SegmentationType, DataAugmentationType, LegalArea, \
    OriginCanton, SubDataset, OriginRegion, Jurisdiction
from hierarchical.hier_bert.configuration_hier_bert import HierBertConfig
from hierarchical.hier_bert.modeling_hier_bert import HierBertForSequenceClassification
from hierarchical.hier_camembert.configuration_hier_camembert import HierCamembertConfig
from hierarchical.hier_camembert.modeling_hier_camembert import HierCamembertForSequenceClassification
from hierarchical.hier_roberta.configuration_hier_roberta import HierRobertaConfig
from hierarchical.hier_roberta.modeling_hier_roberta import HierRobertaForSequenceClassification
from hierarchical.hier_xlm_roberta.configuration_hier_xlm_roberta import HierXLMRobertaConfig
from hierarchical.hier_xlm_roberta.modeling_hier_xlm_roberta import HierXLMRobertaForSequenceClassification
from arguments.model_arguments import ModelArguments, LabelImbalanceMethod, LongInputBertType, TrainType
from utils.sentencizer import get_sentencizer, combine_small_sentences, spacy_sentencize, get_spacy_sents

os.environ['WANDB_MODE'] = "online"
os.environ['WANDB_WATCH'] = "false"  # disable gradient logging
# os.environ['WANDB_NOTES'] = "Enter notes here"
os.environ['TOKENIZERS_PARALLELISM'] = "True"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # use this when debugging

# Will error if the minimal version of transformers is not installed. Remove at your own risks.
check_min_version("4.8.2")

logger = logging.getLogger(__name__)

faulthandler.enable()

logger.warning("This script only supports PyTorch models!")


# TODO save all predictions to wandb so we can do significance testing later on with aso
# TODO remove special run-names because in the end, we always filter by model/data/training arguments in wandb ==> so the old models are not overwritten (however, then they are also hard to find)

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
    os.environ['WANDB_PROJECT'] = f'SwissJudgmentPredictionCrossLingualTransfer'

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

    def custom_asdict_factory(data):
        def convert_value(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return dict((k, convert_value(v)) for k, v in data)

    # Save all params for better reproducibility
    experiment_params = {
        "model_args": dataclasses.asdict(model_args, dict_factory=custom_asdict_factory),
        "data_args": dataclasses.asdict(data_args, dict_factory=custom_asdict_factory),
        "training_args": dataclasses.asdict(training_args, dict_factory=custom_asdict_factory),
        "adapter_args": dataclasses.asdict(adapter_args, dict_factory=custom_asdict_factory),
    }
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{training_args.output_dir}/experiment_params.yaml', 'w') as file:
        yaml.safe_dump(experiment_params, file, default_flow_style=False)

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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info(f"Experiment parameters:")
    pprint.pprint(experiment_params)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    def remove_unused_features(datasets, remove_original_cols, remove_augmented_cols=True):
        columns_to_remove = []
        if remove_original_cols:
            columns_to_remove.extend(['chamber', 'num_tokens_spacy', 'num_tokens_bert',
                                      'origin_region', 'origin_canton', 'origin_court', 'origin_chamber', 'legal_area'])
        if remove_augmented_cols:
            columns_to_remove.extend(['source_language', 'Unnamed: 0'])
        for i in range(len(datasets)):
            for column in columns_to_remove:
                if column in datasets[i].column_names:
                    datasets[i] = datasets[i].remove_columns(column)
        return datasets

    def filter_by_sub_datasets(dataset):
        if data_args.train_sub_datasets not in ['None', 'False']:  # if we want to train on special sub datasets only
            train_sub_datasets = data_args.train_sub_datasets.split(",")
            train_sub_datasets = [SubDataset.from_str(sub_dataset) for sub_dataset in train_sub_datasets]
            assert 'en' not in model_args.train_languages  # the Indian dataset does not have these metadata

            def item_is_in_sub_datasets(item):
                for sub_dataset in train_sub_datasets:
                    assert isinstance(sub_dataset, SubDataset) and \
                           (isinstance(sub_dataset, LegalArea) or
                            isinstance(sub_dataset, OriginCanton) or isinstance(sub_dataset, OriginRegion))
                    if item[sub_dataset.get_dataset_column_name()] == sub_dataset:
                        return True
                return False

            dataset = dataset.filter(item_is_in_sub_datasets,
                                     load_from_cache_file=not data_args.overwrite_cache)
        return dataset

    # transform comma separated string into list of languages
    # NOTE multiple test_languages are not possible when using language adapters
    model_args.train_languages = model_args.train_languages.split(",")
    model_args.test_languages = model_args.test_languages.split(",")

    train_datasets, eval_datasets, = [], []
    if training_args.do_train:
        for lang in model_args.train_languages:
            train_files = [(DATA_DIR / lang / 'train.csv').as_posix()]
            if data_args.data_augmentation_type in [DataAugmentationType.TRANSLATION,
                                                    DataAugmentationType.BACK_TRANSLATION]:
                path = AUGMENTED_DIR / data_args.data_augmentation_type / lang
                if data_args.jurisdiction == Jurisdiction.INDIA:
                    path = path / Jurisdiction.INDIA.value  # only take the indian data
                    train_files = []  # remove the main train files
                train_files.extend(glob.glob(f"{path}/*.csv"))  # add all files inside this path
                if data_args.jurisdiction == Jurisdiction.BOTH:
                    train_files.extend(glob.glob(f"{path / Jurisdiction.INDIA.value}/*.csv"))  # add the indian data
            # load files separately so we can remove unused features before merging into one
            for train_file in train_files:
                train_datasets.append(load_dataset("csv", data_files={"train": train_file})['train'])
        # if we train with the Indian dataset
        remove_original_cols = 'en' in model_args.train_languages \
                               or data_args.jurisdiction in [Jurisdiction.INDIA, Jurisdiction.BOTH]
        # we need to remove some columns, so we can merge
        train_datasets = remove_unused_features(train_datasets, remove_original_cols)
        train_dataset = concatenate_datasets(train_datasets)  # we want to train on all datasets at the same time
        train_dataset = filter_by_sub_datasets(train_dataset)

        # Using the Indian cases only in the Swiss train set period (not older ones).
        # train_dataset = train_dataset.filter(lambda item: int(item['year']) >= 2000)
        # TODO Using the Indian cases that are up to 2048 tokens.

    if training_args.do_eval:
        for lang in model_args.train_languages:
            eval_path = (DATA_DIR / lang / 'val.csv').as_posix()
            eval_dataset = load_dataset("csv", data_files={"validation": eval_path})['validation']
            eval_datasets.append(eval_dataset)
        # if we train with the Indian dataset
        remove_original_cols = 'en' in model_args.train_languages
        # we need to remove some columns, so we can merge
        eval_datasets = remove_unused_features(eval_datasets, remove_original_cols)
        eval_dataset = concatenate_datasets(eval_datasets)  # we want to evaluate on all datasets at the same time
        eval_dataset = filter_by_sub_datasets(eval_dataset)

    predict_datasets, sub_datasets = {}, {}
    for lang in model_args.test_languages:
        if training_args.do_predict:
            predict_path = (DATA_DIR / lang / 'test.csv').as_posix()
            predict_dataset = load_dataset("csv", data_files={"test": predict_path})['test']
            predict_datasets[lang] = predict_dataset

        if data_args.test_on_sub_datasets:
            lang_sub_datasets = dict()
            lang_sub_dataset_dir = DATA_DIR / lang / 'sub_datasets'
            if lang_sub_dataset_dir.exists():  # for example in the indian dataset we don't have this
                sub_datasets_to_run = ['input_length', 'legal_area', 'year', 'origin_canton', 'origin_region']
                for file in glob.glob(f'{lang_sub_dataset_dir}/*/*.csv'):
                    experiment = Path(file).parent.stem
                    if experiment in sub_datasets_to_run:  # exclude very large ones like origin_court and origin_chamber
                        part = Path(file).stem.split("-")[1]
                        if experiment not in lang_sub_datasets:
                            lang_sub_datasets[experiment] = dict()
                        lang_sub_datasets[experiment][part] = load_dataset("csv", data_files={"test": file})['test']
                sub_datasets[lang] = lang_sub_datasets

    # Labels: just take the labels from the first language. We assume that they are identical anyway.
    with open(DATA_DIR / model_args.train_languages[0] / 'labels.json', 'r') as f:
        label_dict = json.load(f)
        label_dict['id2label'] = {int(k): v for k, v in label_dict['id2label'].items()}
        label_dict['label2id'] = {k: int(v) for k, v in label_dict['label2id'].items()}
        label_list = list(label_dict["label2id"].keys())
    num_labels = len(label_list)

    if data_args.problem_type == ProblemType.MULTI_LABEL_CLASSIFICATION:
        mlb = MultiLabelBinarizer().fit([label_list])

    # Load pretrained model and tokenizer
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
        max_segments=data_args.max_segments,
        max_segment_length=data_args.max_seg_len,
        segment_encoder_type="transformer",
    )
    tokenizer_class = AutoTokenizer
    if model_args.model_name_or_path == 'microsoft/Multilingual-MiniLM-L12-H384':
        tokenizer_class = XLMRobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def save_model(model, folder):
        # save entire model ourselves just to be safe
        torch.save(model.state_dict(), f'{folder}/model.bin')
        logger.info(f"Model state dict saved to {folder}/model.bin")

        # save adapters
        if model_args.train_type == TrainType.ADAPTERS:
            model.save_adapter(folder, data_args.task_name)

    def load_model(model, folder):
        # load entire model ourselves just to be safe
        model_path = Path(f'{folder}/model.bin')
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=training_args.device))
            logger.info(f"Model state dict loaded from {model_path}")
            model.to(training_args.device)

            # load adapters
            if model_args.train_type == TrainType.ADAPTERS:
                model.load_adapter(folder, load_as=data_args.task_name)

    def model_init():
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

        model_class = AutoModelForSequenceClassification
        configuration = config

        # Future work: Try different learning rates for base encoder and segment encoder
        if model_args.long_input_bert_type == LongInputBertType.HIERARCHICAL:
            if config.model_type == 'bert':
                config_class = HierBertConfig
                model_class = HierBertForSequenceClassification
            if config.model_type == 'roberta':
                config_class = HierRobertaConfig
                model_class = HierRobertaForSequenceClassification
            if config.model_type == 'xlm-roberta':
                config_class = HierXLMRobertaConfig
                model_class = HierXLMRobertaForSequenceClassification
            if config.model_type == 'camembert':
                config_class = HierCamembertConfig
                model_class = HierCamembertForSequenceClassification
            configuration = config_class(**config.to_dict())

        if model_args.use_pretrained_model:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=configuration,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            model = model_class.from_config(configuration)

        if model_args.long_input_bert_type == LongInputBertType.LONG:
            model.base_model = LongBert.resize_position_embeddings(model.base_model,
                                                                   max_length=data_args.max_seq_len,
                                                                   device=training_args.device)

        if model_args.train_type == TrainType.ADAPTERS:
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
                        model.load_adapter(adapter_args.load_adapter, config=adapter_config, load_as=task_name)
                    # otherwise, add a fresh adapter
                    else:
                        model.add_adapter(task_name, config=adapter_config)
                # optionally load a pre-trained language adapter
                if adapter_args.load_lang_adapter and adapter_args.load_lang_adapter not in ['False', 'None']:
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
                        model_name=model_args.model_name
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

        if model_args.train_type == TrainType.BITFIT:
            # https://arxiv.org/abs/2106.10199, https://arxiv.org/abs/2109.00904
            for p in model.named_parameters():
                if "bias" in p[0]:
                    p[1].requires_grad = True
                else:
                    p[1].requires_grad = False

        logger.info(model)

        return model

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        # IMPORTANT: Can lead to problem with HierarchicalBert
        padding = "longest"

    # TODO do sentence splitting beforehand in the SCRC dataset creation
    if data_args.segmentation_type == SegmentationType.SENTENCE:
        sentencizers = {lang: get_sentencizer(lang) for lang in model_args.train_languages}

    def preprocess_function(batch):
        pad_id = tokenizer.pad_token_id
        if model_args.long_input_bert_type == LongInputBertType.HIERARCHICAL:
            batch['segments'] = []
            if data_args.segmentation_type == SegmentationType.BLOCK:
                tokenized = tokenizer(batch["text"], padding=padding, truncation=True,
                                      max_length=data_args.max_segments * data_args.max_seg_len,
                                      add_special_tokens=False)  # prevent it from adding the cls and sep tokens twice
                for ids in tokenized['input_ids']:
                    # convert ids to tokens and then back to strings
                    id_blocks = [ids[i:i + data_args.max_seg_len] for i in range(0, len(ids), data_args.max_seg_len) if
                                 ids[i] != pad_id]  # remove blocks containing only ids
                    id_blocks[-1] = [id for id in id_blocks[-1] if
                                     id != pad_id]  # remove remaining pad_tokens_ids from the last block
                    token_blocks = [tokenizer.convert_ids_to_tokens(ids) for ids in id_blocks]
                    string_blocks = [tokenizer.convert_tokens_to_string(tokens) for tokens in token_blocks]
                    batch['segments'].append(string_blocks)
            elif data_args.segmentation_type == SegmentationType.SENTENCE:
                # TODO get paragraph information here because sentence splitting is difficult with legal text:
                #  https://aclanthology.org/W19-2204.pdf, https://www.scitepress.org/Papers/2021/102463/102463.pdf
                # For the moment just do it so we can test the new bert variant
                sents_list = []
                if len(model_args.train_languages) == 1:
                    nlp = sentencizers[model_args.train_languages[0]]
                    for doc in nlp.pipe(batch['text'], batch_size=len(batch['text'])):
                        sents_list.append([sent.text for sent in doc.sents])
                else:  # if the languages are mixed we need to load it from the case
                    for case in batch['text']:
                        nlp = sentencizers[case['language']]
                        sents_list.append(get_spacy_sents(case, nlp))
                for sents in sents_list:
                    sentences = combine_small_sentences(sents, data_args.min_seg_len)
                    batch['segments'].append(sentences)

            # Tokenize the texts
            tokenized = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
            for case in batch['segments']:
                case_encodings = tokenizer(case[:data_args.max_segments], padding=padding, truncation=True,
                                           max_length=data_args.max_seg_len, return_token_type_ids=True)
                tokenized['input_ids'].append(append_zero_segments(case_encodings['input_ids'], pad_id))
                tokenized['attention_mask'].append(append_zero_segments(case_encodings['attention_mask'], 0))
                tokenized['token_type_ids'].append(append_zero_segments(case_encodings['token_type_ids'], 0))
            del batch['segments']
        else:
            # Tokenize the texts
            tokenized = tokenizer(batch["text"], padding=padding, truncation=True,
                                  max_length=data_args.max_seq_len, return_token_type_ids=True)

        # Map labels to IDs
        if data_args.problem_type == ProblemType.MULTI_LABEL_CLASSIFICATION:
            tokenized["label"] = [mlb.transform([eval(labels)])[0] for labels in batch["label"]]
        if data_args.problem_type == ProblemType.SINGLE_LABEL_CLASSIFICATION:
            if label_dict["label2id"] is not None and "label" in batch:
                tokenized["label"] = [label_dict["label2id"][l] for l in batch["label"]]
        return tokenized

    def append_zero_segments(case_encodings, pad_token_id):
        """appends a list of zero segments to the encodings to make up for missing segments"""
        return case_encodings + [[pad_token_id] * data_args.max_seg_len] * (
                data_args.max_segments - len(case_encodings))

    def preprocess_dataset(dataset):
        return dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=[col for col in dataset.column_names if not col == "id"],  # keep id for example-wise logging
        )

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = preprocess_dataset(train_dataset)
        # Log a random sample from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            # make sure the tokenizer didn't do anything stupid
            if model_args.long_input_bert_type == LongInputBertType.HIERARCHICAL:
                assert len(train_dataset[index]['input_ids'][0]) == data_args.max_seg_len
            else:
                assert len(train_dataset[index]['input_ids']) == data_args.max_seq_len

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = preprocess_dataset(eval_dataset)

    if training_args.do_predict:
        for lang in model_args.test_languages:
            if data_args.max_predict_samples is not None:
                predict_datasets[lang] = predict_datasets[lang].select(range(data_args.max_predict_samples))
            predict_datasets[lang] = preprocess_dataset(predict_datasets[lang])

    if data_args.test_on_sub_datasets:
        for lang in sub_datasets.keys():
            for experiment, parts in sub_datasets[lang].items():
                for part, dataset in parts.items():
                    sub_datasets[lang][experiment][part] = preprocess_dataset(dataset)

    def labels_to_bools(labels):
        return [tl == 1 for tl in labels]

    def preds_to_bools(preds):
        return [pl > model_args.prediction_threshold for pl in preds]

    def process_results(preds, labels):
        preds = preds[0] if isinstance(preds, tuple) else preds
        probs = softmax(preds)
        if data_args.problem_type == ProblemType.MULTI_LABEL_CLASSIFICATION:
            # for multi_label_classification we need boolean arrays for each example
            preds, labels = preds_to_bools(preds), labels_to_bools(labels)
        if data_args.problem_type == ProblemType.SINGLE_LABEL_CLASSIFICATION:
            preds = np.argmax(preds, axis=1)
        return preds, labels, probs

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        labels = p.label_ids
        preds = p.predictions
        preds, labels, probs = process_results(preds, labels)

        positive_probs = probs[:, 1]  # get only the probs of the positive class (only in binary classification!)
        average_precision = average_precision_score(labels, positive_probs)
        roc_auc = roc_auc_score(labels, positive_probs)
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        # macro averaging is a better evaluation metric for imbalanced label distributions
        precision, recall, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
        return OrderedDict({
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'balanced_accuracy': balanced_accuracy,
            'average_precision': average_precision,
            'roc_auc': roc_auc,
            'mcc': mcc,
        })

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer_class = Trainer
    if training_args.do_train and model_args.label_imbalance_method == LabelImbalanceMethod.CLASS_WEIGHTS:
        lbls = [item['label'] for item in train_dataset]
        # compute class weights based on label distribution
        class_weight = compute_class_weight('balanced', classes=np.unique(lbls), y=lbls)
        class_weight = torch.tensor(class_weight, dtype=torch.float32, device=training_args.device)  # create tensor

        class CustomTrainer(trainer_class):
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

        trainer_class = CustomTrainer

    # NOTE: This is not optimized for multiclass classification
    if training_args.do_train and model_args.label_imbalance_method in [LabelImbalanceMethod.OVERSAMPLING,
                                                                        LabelImbalanceMethod.UNDERSAMPLING]:
        label_datasets = dict()
        minority_len, majority_len = len(train_dataset), 0
        for label_id in label_dict['id2label'].keys():
            label_datasets[label_id] = train_dataset.filter(lambda item: item['label'] == label_id,
                                                            load_from_cache_file=not data_args.overwrite_cache)
            if len(label_datasets[label_id]) < minority_len:
                minority_len = len(label_datasets[label_id])
                minority_id = label_id
            if len(label_datasets[label_id]) > majority_len:
                majority_len = len(label_datasets[label_id])
                majority_id = label_id

        if model_args.label_imbalance_method == LabelImbalanceMethod.OVERSAMPLING:
            logger.info("Oversampling the minority class")
            datasets = [train_dataset]
            num_full_minority_sets = int(majority_len / minority_len)
            for i in range(num_full_minority_sets - 1):  # -1 because one is already included in the training dataset
                datasets.append(label_datasets[minority_id])

            remaining_minority_samples = majority_len % minority_len
            random_ids = np.random.choice(minority_len, remaining_minority_samples, replace=False)
            datasets.append(label_datasets[minority_id].select(random_ids))
            train_dataset = concatenate_datasets(datasets)

        if model_args.label_imbalance_method == LabelImbalanceMethod.UNDERSAMPLING:
            logger.info("Undersampling the majority class")
            random_ids = np.random.choice(majority_len, minority_len, replace=False)
            # just select only the number of minority samples from the majority class
            label_datasets[majority_id] = label_datasets[majority_id].select(random_ids)
            train_dataset = concatenate_datasets(list(label_datasets.values()))

    class CheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kwargs):
            if args.save_strategy == "epoch":
                checkpoint_number = state.epoch
            elif args.save_strategy == "steps":
                checkpoint_number = state.global_step
            else:
                return
            save_model(model, f"{args.output_dir}/checkpoint-{checkpoint_number}")

    callbacks = [EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience,
                                       early_stopping_threshold=model_args.early_stopping_threshold),
                 CheckpointCallback()]
    if "wandb" in training_args.report_to:
        callbacks.append(CustomWandbCallback(experiment_params))
    # Initialize our Trainer
    trainer = trainer_class(
        model=model_init() if not data_args.tune_hyperparams else None,
        model_init=model_init if data_args.tune_hyperparams else None,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
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
            # Choose among many libraries: https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
            # search_alg=HyperOptSearch(),
            # Choose among schedulers: https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
            # scheduler=AsyncHyperBand()
        )
        logger.info(best_trial)
        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)

    def predict(predict_dataset, prefix="test"):
        preds, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix=prefix)
        remove_metrics(metrics, prefix)

        preds, labels, probs = process_results(preds, labels)
        return preds, labels, probs, metrics

    def write_report_section(writer, title, content):
        writer.write(f"{title}\n")
        writer.write("=" * 75 + "\n\n")
        writer.write(content)
        writer.write("\n" * 3)

    def pred2label(pred):
        if data_args.problem_type == ProblemType.MULTI_LABEL_CLASSIFICATION:
            return mlb.inverse_transform(np.array([pred]))[0]
        if data_args.problem_type == ProblemType.SINGLE_LABEL_CLASSIFICATION:
            return label_dict["id2label"][pred]

    def write_reports(base_dir, ids, preds, labels, probs, wandb_prefix, split):
        assert len(ids) == len(preds) == len(labels) == len(probs)
        if trainer.is_world_process_zero():
            correct_confidences, incorrect_confidences = [], []
            # write predictions to csv
            result = {"id": [], "prediction": [], "label": [], "is_correct": [], "confidence": [], "error": []}
            for index, pred in enumerate(preds):
                confidence = probs[index][pred]
                is_correct = pred == labels[index]
                error = 1 - confidence if is_correct else confidence
                if is_correct:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)
                result['id'].append(ids[index])
                result['prediction'].append(pred2label(pred))
                result['label'].append(pred2label(labels[index]))
                result['is_correct'].append(is_correct)
                result['confidence'].append(confidence)
                result['error'].append(error)
            pd.DataFrame.from_dict(result).to_csv(f'{base_dir}/predictions_{split}.csv')

            # IMPORTANT: These confidences are misleading!
            # TODO Use calibration to get better confidences: https://towardsdatascience.com/neural-network-calibration-using-pytorch-c44b7221a61
            # write confidences to csv
            confidences = {"correct": {"mean": np.mean(correct_confidences), "std": np.std(correct_confidences)},
                           "incorrect": {"mean": np.mean(incorrect_confidences), "std": np.std(incorrect_confidences)}}
            pd.DataFrame.from_dict(confidences, orient='index').to_csv(f'{base_dir}/confidences.csv')

            if "wandb" in training_args.report_to:
                wandb.log(OrderedDict({
                    f"{wandb_prefix}correct_mean": confidences['correct']['mean'],
                    f"{wandb_prefix}correct_std": confidences['correct']['std'],
                    f"{wandb_prefix}incorrect_mean": confidences['incorrect']['mean'],
                    f"{wandb_prefix}incorrect_std": confidences['incorrect']['std'],
                }))

            # write report file
            with open(f'{base_dir}/prediction_report.txt', "w") as writer:
                if data_args.problem_type == ProblemType.MULTI_LABEL_CLASSIFICATION:
                    title = "Multilabel Confusion Matrix"
                    matrices = multilabel_confusion_matrix(labels, preds)
                if data_args.problem_type == ProblemType.SINGLE_LABEL_CLASSIFICATION:
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

    def remove_metrics(metrics, split):
        # remove unnecessary values to make overview nicer in wandb
        metrics.pop(f"{split}_loss")
        metrics.pop(f"{split}_runtime")
        metrics.pop(f"{split}_steps_per_second")
        metrics.pop(f"{split}_samples_per_second")

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

        save_model(trainer.model, training_args.output_dir)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if data_args.log_all_predictions:
            # This can be used to get detailed insight into specific predictions
            preds, labels, probs, metrics = predict(train_dataset)
            write_reports(training_args.output_dir, train_dataset["id"], preds, labels, probs, "train", "train")

    # load model ourselves because save_pretrained/load_pretrained might not work well for our hacked models
    # load_model(trainer.model, training_args.output_dir)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        remove_metrics(metrics, 'eval')

        max_eval_samples = data_args.max_eval_samples if \
            data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if data_args.log_all_predictions:
            # This can be used to get detailed insight into specific predictions
            preds, labels, probs, metrics = predict(eval_dataset)
            write_reports(training_args.output_dir, eval_dataset["id"], preds, labels, probs, "eval", "eval")

    base_output_dir = Path(training_args.output_dir)  # save it here because we overwrite it
    if training_args.do_predict and not data_args.tune_hyperparams:
        logger.info("*** Predict ***")
        for lang in model_args.test_languages:
            logger.info(f"Prediction for {lang}")
            training_args.output_dir = base_output_dir / lang
            training_args.output_dir.mkdir(parents=True, exist_ok=True)  # create directory

            predict_dataset = predict_datasets[lang]
            preds, labels, probs, metrics = predict(predict_dataset)

            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            )
            metrics["test_samples"] = min(max_predict_samples, len(predict_dataset))

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

            # rename metrics so that they appear in separate section in wandb and filter out unnecessary ones
            prefix = f"test/{lang}/"
            if "wandb" in training_args.report_to:
                metrics = {k.replace("test_", prefix): v for k, v in metrics.items()}
                # if "mem" not in k and k != "test_samples"}
                wandb.log(metrics)  # log test metrics to wandb

            write_reports(training_args.output_dir, predict_dataset["id"], preds, labels, probs, prefix, "test")

    if data_args.test_on_sub_datasets:
        logger.info("*** Sub-Datasets ***")
        for lang in sub_datasets.keys():
            logger.info(f"Sub-Datasets Prediction for {lang}")
            for experiment, parts in sub_datasets[lang].items():
                for part, dataset in parts.items():
                    if dataset.num_rows >= 50:  # below a minimum number the results are too noisy
                        prefix = f"{lang}/{experiment}/{part}/"
                        training_args.output_dir = Path(training_args.output_dir) / prefix
                        training_args.output_dir.mkdir(parents=True, exist_ok=True)
                        preds, labels, probs, metrics = predict(dataset)
                        if "wandb" in training_args.report_to:
                            metrics = {k.replace("test_", prefix): v for k, v in metrics.items()}
                            metrics[f'{prefix}support'] = dataset.num_rows
                            wandb.log(metrics)  # log test metrics to wandb
                        write_reports(training_args.output_dir, dataset["id"], preds, labels, probs, prefix, "test")

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": finetuning_task}
        if data_args.task_name is not None:
            kwargs["language"] = model_args.test_languages
            kwargs["dataset_tags"] = "sjp"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"SJP {data_args.task_name.upper()}"

        trainer.push_to_hub(**kwargs)

        trainer.model.push_adapter_to_hub(
            "my-awesome-adapter",
            "awesome_adapter",
            adapterhub_tag="text_classification/legal_judgment_prediction",
            datasets_tag="swiss_judgment_prediction"
        )

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{base_output_dir}/*/') if '/checkpoint' in filepath]
    logger.info("Cleaning up checkpoints")
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
