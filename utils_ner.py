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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """
#code from Minbyul et al., 2021: https://github.com/minstar/PMI/blob/main/run_ner.py , with modifications

import logging
import os
import json
import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock

from tqdm import tqdm
from transformers import PreTrainedTokenizer, is_torch_available
from data_preprocessing import lemmatize_sentence
from utils import get_pos_and_words


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]
    pos: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    pos_ids: Optional[List[int]] = None
    bias_tensor: Optional[List[int]] = None
    data_type: Optional[List[int]] = None
    temp_ids: Optional[List[int]] = None

class Split(Enum):
    train = "train"
    dev = "devel"
    test = "test"
    predict = "predict"

if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class NerDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            is_pmi=False,
            smooth_param=1,
            class_alpha=1,
            word_alpha=1,
            use_subword=True,
            eval_data_name='',
            sentence_to_predict=''
        ):

            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    examples = read_examples_from_file(data_dir, mode, eval_data_name=eval_data_name, sentence_to_predict=sentence_to_predict)
                    # TODO clean up all this to leverage built-in features of tokenizers
                    self.features = convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                        data_name=data_dir,
                        data_type=mode.value,
                        smooth_param=smooth_param,
                        class_alpha=class_alpha,
                        word_alpha=word_alpha,
                        use_subword=use_subword,
                        is_pmi=is_pmi,
                    )
                    # logger.info(f"Saving features into cached file {cached_features_file}")
                    # torch.save(self.features, cached_features_file)
                
        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


def read_examples_from_file(data_dir, mode: Union[Split, str], eval_data_name='',sentence_to_predict='') -> List[InputExample]:
    pos_tagset = []
    with open("tagset.txt", 'r', encoding="utf-8") as file:
        for line in file:
            pos_tagset.append(line.strip())
    pos_to_index = {tag: index for index, tag in enumerate(pos_tagset)}

    split = Split

    if isinstance(mode, split):
        mode = mode.value
    if mode == "predict":
        lemmatized_sentence = lemmatize_sentence(sentence_to_predict)
        list_sentence = []
        while len(lemmatized_sentence) > 64:
            list_sentence.append(lemmatized_sentence[:63])
            lemmatized_sentence = lemmatized_sentence[64:]
        list_sentence.append(lemmatized_sentence)
        guid_index = 1
        examples = []
        for sentence in list_sentence:
            sentence_tuples = get_pos_and_words(sentence)
            words = [item[0] for item in sentence_tuples]
            pos = [pos_to_index[item[1]] for item in sentence_tuples]
            labels = ['O' for item in sentence_tuples] # all labels are set to '_'
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, pos=pos))
            guid_index += 1
        return examples
    else:
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            pos = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, pos=pos))
                        guid_index += 1 #sentence index
                        words = []
                        labels = []
                        pos = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        splits_replace = splits[-1].replace("\n", "")
                        labels.append(splits_replace)
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
                    pos.append(pos_to_index[splits[1]])
                    
            
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, pos=pos))

        return examples # a list of InputExamples, each InputExample represent a sentence

def make_word_class_distribution(tokenizer,examples,output_file,label_list):
    dict = {}
    for label in label_list:
        dict[label] = {}
    for example in examples:
        for i in range(len(example.words)):
            word_tokens = tokenizer.tokenize(example.words[i])
            if len(word_tokens) > 0:
                for word_token in word_tokens:
                    for key in dict.keys():
                        if word_token not in dict[key]:
                            dict[key][word_token] = 0
                    dict[example.labels[i]][word_token] += 1
    with open(output_file, "w") as outfile:
        json.dump(dict, outfile)

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    data_name="",
    data_type="",
    smooth_param=1,
    class_alpha=1,
    word_alpha=1,
    use_subword=True, 
    is_pmi=False,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers
    label_map = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_map)
    features = []
    
    word_class_distribution_file = os.path.join(data_name,"train-subword-class_distribution.json")
    if not os.path.exists(word_class_distribution_file):
        make_word_class_distribution(tokenizer,examples,word_class_distribution_file,label_list)

    with open(word_class_distribution_file, 'r') as fp:
        word_class_distribution = json.load(fp)

    if '' in word_class_distribution:
        word_class_distribution.pop('', None)    
    
    # is_pmi to perform WORD or OURS 
    pmi_data = _get_pmi_data(word_class_distribution, label_map, smooth_param=smooth_param, class_alpha=class_alpha, word_alpha=word_alpha, is_pmi=is_pmi)
    word_class_distribution = pmi_data
    visualize_dict(word_class_distribution)

    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens, label_ids, pos_ids, word_class, temperature = [], [], [], [], []
        word_text, label_text = '', ''
        start_idx, punc_idx = -100, -100
        check_idx = 0
        # tokenization and creating labels here, 'tokens' tensor used to save tokenize words, 'labels' tensor used to save labels
        for word_idx, (word, label, pos) in enumerate(zip(example.words, example.labels, example.pos)):
            word_tokens = tokenizer.tokenize(word)
            
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id and pos id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)) #e.g [0,-100,-100...]
                pos_ids.extend([pos] + [0]*(len(word_tokens) - 1))
                if use_subword:
                    word_class.extend(word_tokens) #the subwords from word
                else:
                    word_class.extend([word] * len(word_tokens))

                # Entity length data generation
                for subword_idx, subword in enumerate(word_tokens):
                    if "B" in label:
                        if check_idx == start_idx + 1 and subword_idx == 0:
                            # case for the only entity, indicate subword is beginning of new entity, for words not at start of sentence
                            start_idx = check_idx
                            word_text, label_text = '', ''
                            word_text += subword + ' '
                            label_text += label
                        else:
                            start_idx = check_idx
                            word_text += subword + ' '
                            label_text += label
                    elif "I" in label and check_idx == start_idx + 1:
                        start_idx = check_idx
                        word_text += subword + ' '
                        label_text += label
                    elif "O" in label:
                        if check_idx == start_idx + 1:
                            start_idx = -100
                            # check entity word and text to find entity length
                            if word_text and label_text: #word and label text are the accumulated words and labels from sentence up to that point
                                temperature.extend([len(word_text.split())] * len(label_text))
                            word_text, label_text = '', ''
                            temperature.extend([1])
                        else:
                            temperature.extend([1])

                    check_idx += 1

        # calculate temperature with length : temp = 1 - 0.02 * length
        # temperature = [1 - sharpening * i if i > 1 else i for _, i in enumerate(entity_length)]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            pos_ids = pos_ids[: (max_seq_length-special_tokens_count)]
            word_class = word_class[: (max_seq_length - special_tokens_count)]
            temperature = temperature[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        pos_ids += [0]
        word_class += [sep_token]
        temperature += [1]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            pos_ids += [0]
            word_class += [sep_token]
            temperature += [1]
            
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            pos_ids += [0]
            segment_ids += [cls_token_segment_id]
            word_class += [cls_token]
            temperature += [1]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            pos_ids =[0] + pos_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            word_class = [cls_token] + word_class
            temperature.insert(0, 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_class_tensor = _get_word_class_distribution(word_class, tokens, word_class_distribution, label_ids, max_seq_length, pad_on_left, label_map, is_pmi=is_pmi)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            pos_ids = ([0] * padding_length) + pos_ids
            temperature = ([1] * padding_length) + temperature
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            pos_ids += [0] * padding_length
            temperature += [1] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(pos_ids) == max_seq_length
        
        try:            
            assert len(temperature) == max_seq_length
        except:
            if len(temperature) < max_seq_length:
                temperature += [1] * (max_seq_length - len(temperature))
            else:
                temperature = temperature[:max_seq_length]

        if 'train' in data_type:
            data_type_ids = [1] * max_seq_length
        else:
            data_type_ids = [0] * max_seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("pos_ids: %s", " ".join([str(x) for x in pos_ids]))
            logger.info("data_type_ids: %s", " ".join([str(x) for x in data_type_ids]))
            logger.info("temp_scaling_ids: %s", " ".join([str(x) for x in temperature]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None
        
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, \
                label_ids=label_ids, bias_tensor=word_class_tensor, data_type=data_type_ids, \
                temp_ids=temperature, pos_ids=pos_ids
            )
        )

    return features

def _get_pmi_data(word_class_distribution, label_map, smooth_param=1, class_alpha=1, word_alpha=1, is_pmi=False):
    pmi_data, class_freq = {}, {}
    word_freq = 0 #total number of words
    for class_key in label_map.keys():
        pmi_data[class_key] = {}
        class_freq[class_key] = 0    
    
    # Pointwise Mutual Information (PMI)
    # add-N smoothing
    for class_key in word_class_distribution.keys(): #word_class_distribution is a dictionary that contains labels as keys, and for each label there is another dictionary with subword as key and occurences as value
        for key, val in word_class_distribution[class_key].items():
            word_class_distribution[class_key][key] += smooth_param #Smooth Param from argument before running, just to prevent division by zero
            #class key is e.g 'O', key is the sub word
            class_freq[class_key] += word_class_distribution[class_key][key]

        # add all class frequency
        word_freq += class_freq[class_key]

    for class_key in word_class_distribution.keys():
        for key, val in word_class_distribution[class_key].items():
            cur_word_freq = 0
            for label_key in word_class_distribution.keys(): #add frequency of subword from all labels
                cur_word_freq += word_class_distribution[label_key][key]

            numerator = word_class_distribution[class_key][key] / cur_word_freq
            class_prob = class_freq[class_key] / word_freq
            word_prob = cur_word_freq / word_freq

            if is_pmi:
                pmi_data[class_key][key] = (np.log(numerator) - class_alpha * np.log(class_prob) - word_alpha * np.log(word_prob))
            else:
                pmi_data[class_key][key] = numerator / word_prob

    return pmi_data #contain for each subword the pmi

def visualize_dict(d, indent=0):
    counter = 0
    for key, value in d.items():
        if counter >= 5:
            return
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            visualize_dict(value, indent + 4)
        else:
            print(" " * indent + f"{key}: {value}")
        counter += 1
        
def _get_word_class_distribution(word_class, tokens, word_class_distribution, label_ids, max_seq_length, pad_on_left, label_map, is_pmi=False):
    assert len(word_class) == len(tokens)
    
    class_bias = []
    num_labels = len(label_map)
    padding_length = max_seq_length - len(word_class)

    class_bias.append([0] * num_labels) # [CLS]
    for word_idx, word in enumerate(word_class[1:-1]):
        if word not in word_class_distribution['O']:
            for class_key in word_class_distribution.keys():
                word_class_distribution[class_key][word] = 1

        all_sum, label_list = 0, []
        for class_key in word_class_distribution.keys():
            all_sum += word_class_distribution[class_key][word]
        
        for class_key, class_idx in label_map.items():# label list contains proportion of subword frequency for each label
            label_list.append(word_class_distribution[class_key][word]/all_sum)

        class_bias.append(label_list)

    class_bias.append([0] * num_labels) # [SEP]
    class_bias += ([([0] * num_labels) for i in range(padding_length)])
    
    class_bias = torch.FloatTensor(class_bias).cpu()
    return class_bias

def get_bio_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
            # labels = [i+'-bio' if i != 'O' else 'O' for i in labels]
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        return ["O", "B", "I"]

def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
            
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]