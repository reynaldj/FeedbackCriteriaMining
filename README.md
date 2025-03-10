# BT - James Reynaldi - Feedback generation using ML and NLP
This project aims to develop 2 models for performing NER task. The 2 models being developed are *Bidirectional Encoder Representations from Transformers*(BERT) and *Conditional Random Field*(CRF).

## Environments

- Python 3.9.0 was used

## Additional Requirements
- Please install the libraries listed in ```requirements.txt```

## Text Processing

Functions for text processing can be found in ```data_preprocessing.py```

- *remove_unrelevant_lines()*
- *resolve_pronouns()*
- *lemmatize_folder()*

## BERT
### MLM
For pre-training BERT model with MLM, the function *finetune_distilbert_on_mlm* in class ```mlm.py``` can be used

### Finetune on NER
The relevant files for finetuning BERT on NER are ```run_ner.py```, ```modeling.py```, ```utils_ner.py``` (code from Minbyul et al., 2021: https://github.com/minstar/PMI/blob/main/run_ner.py , with modifications)

```run_ner.py``` performs the main logic, which parse the arguments, initialize mode, start training and evaluate the model. The arguments used can be found in ```automation.py```.

```modeling.py``` provides a class that extends *BertForTokenClassification* from Huggingface, which can be used to perform the forward and backward propagation. ```modeling.py``` provides its own implementation of the forward method, to include extra features that are not defined in the original BERT model, namely the bias and POS tagging

```utils_ner.py``` extract features and the labels from conll data, convert it into a PyTorch dataset.

## CRF
```ner_crf``` applies the entire logic for training a CRF model, as well as evaluating it. (code inspired from https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html)

## Adaptive Resampling
```NER_Adaptive_Resampling.py``` apply the resampling methods suggested in https://aclanthology.org/2022.naacl-main.156.pdf, namely *sc*, *sCR*, *sCRD* and *nsCRD*, as well as *Balanced Undersampling*(BUS) suggested in https://www.researchgate.net/publication/315697738_Balanced_Undersampling_a_Novel_Sentence-based_Undersampling_Method_to_Improve_Recognition_of_Named_Entities_in_Chemical_and_Biomedical_Text

## Data Augmentation
```augment_script.py```(code from https://github.com/XiaoChen-W/NER_Adaptive_Resampling/tree/main/Data_Augmentation) triggers the data augmentation logic implemented in ```augment.py``` (code from https://github.com/boschresearch/data-augmentation-coling2020)

The augmentation method suggested in https://aclanthology.org/2020.coling-main.343.pdf are implemented to augment the training data. The augmentation methods suggested are *SR*, *LwTR*, *MR*, and *SiS*.

## Training and Testing Data
Two types of tagging are experimented on here, namely "BIO" tagging and regular tagging. Testing and training data for "BIO" tagging can be found in "./BIO" folder, whereas data for regular tagging can be found in "./NON_BIO" folder. In both folders, different training data from different resampling and augmentation method are provided, as well as a folder for the corresponding testing data. 

Results of the model trained on that resampling method with one of the tagging can be found in the training data folder. For CRF model, result is saved on folder "ner_crf_..." in the training folder, whereas for BERT model, result is saved on folder "regularized_with_optimizer". (e.g result of training BERT model on *SR* augmention method with *BIO* tagging can be found in "BIO/sr_augmented_new_2_train_data_conll/regularized_with_optimizer/test-classification_report.txt")
