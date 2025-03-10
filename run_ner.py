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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
#code from Minbyul et al., 2021: https://github.com/minstar/PMI/blob/main/run_ner.py , with modifications

import logging
import os
import sys

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
import wandb
from utils_ner import NerDataset, Split, get_bio_labels, get_labels
from modeling import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from data_preprocessing import remove_unrelevant_lines_from_sentence,resolve_pronouns_in_text,lemmatize_sentence

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tagset: str = field(
        metadata={"help": "Path to Tagset"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    random_bias: bool = field(
        default=False, metadata={"help": "biased model as a random logit value"}
    )
    freq_bias: bool = field(
        default=False, metadata={"help": "biased model as a normalized word frequency of each class logit value"}
    )
    pmi_bias: bool = field(
        default=True, metadata={"help": "biased model as a normalized pointwise mutual information value"}
    )
    mixin_bias: bool = field(
        default=True, metadata={"help": "if True biased model on a Learned Mixin framework else Biased Product framework is operating"}
    )
    penalty: bool = field(
        default=False, metadata={"help": "penalty term for Learned Mixin + H framework"},
    )
    lambda_val: float = field(
        default=0.03, metadata={"help": "lambda value of temperature scaling on long-named and complex structure of entities"}
    )
    length_adaptive: bool = field(
        default=True, metadata={"help": "adatively using temperature scaling on biased term"}
    )
    testing: bool = field(
        default=False, metadata={"help": "only run testing"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    wandb_name: str = field(
        default=None, metadata={'help': "Name of Wandb runs"},
    )
    is_pmi: bool = field(
        default=False, metadata={'help': "To use Pointwise Mututal Information in debiasing"}
    )
    smooth: int = field(
        default=1,
        metadata={
            "help": "smoothing hyparameter to make word frequency more discriminative"
        },
    )
    is_subword: bool = field(
        default=False, metadata={'help': "To use subword statistics in biased model"}
    )
    class_alpha: float = field(
        default=1.0,
        metadata={
            "help": "coefficient of class distribution"
        },
    )
    word_alpha: float = field(
        default=1.0,
        metadata={
            "help": "coefficient of word distribution"
        },
    )
    sentence_to_predict: str = field(
        default="",
        metadata={
            "help": "sentence to predict"
        },
    )

def count_elements_in_nested_list(nested_list):
    total_count = 0
    for sublist in nested_list:
        total_count += len(sublist)
    return total_count

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    wandb.init(project="your project name", name=data_args.wandb_name) # need check

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

    # Prepare CONLL-2003 task
    entity_name = data_args.data_dir.split('/')[-1]
    if entity_name in ["CoNLL2003NER", "OntoNotes5.0", "WNUT2017"]:
        labels = get_labels(data_args.labels)
    else:
        labels = get_bio_labels(data_args.labels)

    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    label_map[-100] = 'O'
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    
    model = NER.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path), #False, model saved as bin file
        num_labels=num_labels,
        config=config,
        cache_dir=model_args.cache_dir, #default=None
        random_bias=model_args.random_bias, #default=false
        freq_bias=model_args.freq_bias, #default=False
        pmi_bias=model_args.pmi_bias, #default=True
        mixin_bias=model_args.mixin_bias, #default=True
        penalty=model_args.penalty, #default=False
        lambda_val=model_args.lambda_val, #default=0.03
        length_adaptive=model_args.length_adaptive, #default=True
        tagset=model_args.tagset
    )

    '''
    model_to_save = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model_to_save.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    import pdb; pdb.set_trace()
    '''
    if model_args.testing == False:
        # Get datasets
        train_dataset = (
            NerDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length, #128
                overwrite_cache=data_args.overwrite_cache, #False
                mode=Split.train,
                is_pmi=data_args.is_pmi, #True
                smooth_param=data_args.smooth, #100
                class_alpha=data_args.class_alpha, #1.0
                word_alpha=data_args.word_alpha, #1.0
                use_subword=data_args.is_subword, #True
            )
            if training_args.do_train #True
            else None
        )
        eval_dataset = (
            NerDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length, #128
                overwrite_cache=data_args.overwrite_cache, #False
                mode=Split.dev,
                is_pmi=data_args.is_pmi, #True
                smooth_param=data_args.smooth, #1.0
                use_subword=data_args.is_subword, #True
            )
            if training_args.do_eval #True
            else None
        )
    
    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions[0], axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def align_predictions_predict(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions[0], axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(seq_len):
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)

        flattened_preds_list = [item for sublist in preds_list for item in sublist]
        flattened_out_label_list = [item for sublist in out_label_list for item in sublist]

        classification_report_file = os.path.join(data_args.data_dir, "classification_report.txt")

        with open(classification_report_file, "w") as output_file:
            sys.stdout = output_file
            print(classification_report(flattened_out_label_list, flattened_preds_list, output_dict=True))
            sys.stdout = sys.__stdout__
        
        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
    results = {}
    if model_args.testing == False:
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        # Training
        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
            trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)

        # Evaluation
        
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            result = trainer.evaluate()
            
            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

                results.update(result)
    else: 
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics
        )

    # def remove_irrelevant_tokens(token_list):
    #     for token in token_list:
    #         if token == '[CLS]' or token == ''
    
    
    # Predict
    if training_args.do_predict:
        if len(data_args.sentence_to_predict) > 0:
            sentence = data_args.sentence_to_predict
            sentence = lemmatize_sentence(resolve_pronouns_in_text(remove_unrelevant_lines_from_sentence(sentence)))
            test_dataset = NerDataset(
                data_dir=data_args.data_dir, #data_dir cannot be None
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.predict,
                is_pmi=data_args.is_pmi,
                smooth_param=data_args.smooth,
                use_subword=data_args.is_subword,
                sentence_to_predict=sentence,
            )
        else:
            test_dataset = NerDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.test,
                is_pmi=data_args.is_pmi,
                smooth_param=data_args.smooth,
                use_subword=data_args.is_subword,
            )

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, out_label_list = align_predictions_predict(predictions, label_ids)
        flattened_preds_list = [item for sublist in preds_list for item in sublist]
        flattened_out_label_list = [item for sublist in out_label_list for item in sublist]
        if len(data_args.sentence_to_predict) > 0:
            feedback_criteria = []
            temp_criteria = ""
            for i in range(len(preds_list)):
                sentence_from_tokens = tokenizer.convert_ids_to_tokens(test_dataset[i].input_ids) #filter irrelevant tokens
                for j in range(len(preds_list[i])):
                    if preds_list[i][j] != 'O':
                        temp_criteria += sentence_from_tokens[j] + " "
                    else:
                        if len(temp_criteria) != 0:
                            if ('[CLS]' in temp_criteria) or ('[SEP]' in temp_criteria) or ('[PAD]' in temp_criteria):
                                temp_criteria = ""
                            else:
                                feedback_criteria.append(temp_criteria)
                                temp_criteria = ""
            print(feedback_criteria)
        else:
            classification_report_file = os.path.join(training_args.output_dir, "test-classification_report.txt")
            with open(classification_report_file, "w") as output_file:
                sys.stdout = output_file
                print(classification_report(flattened_out_label_list, flattened_preds_list))
                sys.stdout = sys.__stdout__

            output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            if trainer.is_world_process_zero():
                with open(output_test_predictions_file, "w", encoding = "utf-8") as writer:
                    with open(os.path.join(data_args.data_dir, "test.txt"), "r", encoding = "utf-8") as f:
                        example_id = 0
                        for line in f:
                            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                                writer.write(line)
                                if not preds_list[example_id]:
                                    example_id += 1
                            elif preds_list[example_id]:
                                entity_label = preds_list[example_id].pop(0)
                                if entity_name == 'WNUT2017':
                                    output_line = line.split()[0] + "\t" + line.split()[1] + "\t" + entity_label + "\n"
                                else:
                                    output_line = line.split()[0] + " " + entity_label + "\n"
                                writer.write(output_line)
                            else:
                                logger.warning(
                                    "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
                                )
            

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()