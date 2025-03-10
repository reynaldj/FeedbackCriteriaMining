#code inspired from https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html

from utils import open_conll
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import nltk
# import sklearn
import scipy.stats
from sklearn.metrics import make_scorer, classification_report
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import metrics
import json
from utils import get_pos_and_words, get_class_freq_from_training_datas
import random
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from data_preprocessing import lemmatize_sentence, remove_unrelevant_lines_from_sentence, resolve_pronouns_in_text
from utils import get_pos_and_words
import string
import joblib

nltk.download('stopwords')
nltk.download('punkt')
german_stop_words = set(stopwords.words('german'))
german_stop_words.update(['?', ')', '...', '!', ':', ';', ',', '.', '(', '[', ']','/','„','“','``'])
german_stop_words = list(german_stop_words)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [tuples[2] for tuples in sent]

def sent2tokens(sent):
    return [tuples[0] for tuples in sent]

def generate_machine_labeled_datas(train_folder,unlabeled_folder_of_unlabeled_sets,labels, output_file):
    lower_threshold = 0.7
    upper_threshold = 0.9
    machine_labeled_datas = []
    folders = os.listdir(unlabeled_folder_of_unlabeled_sets)
    paths_to_sets = []
    training_datas = []
    for filename in os.listdir(train_folder):
        if filename.endswith('.conll'):
            input_path = os.path.join(train_folder, filename)
            training_data = open_conll(input_path)
            training_datas += training_data
    # Iterate through the items and count subfolders
    for item in folders:
        item_path = os.path.join(unlabeled_folder_of_unlabeled_sets, item)        
        # Check if the item is a directory (subfolder)
        if os.path.isdir(item_path):
            paths_to_sets.append(item_path)
    for folder in paths_to_sets:
        unlabeled_datas = prepare_unlabeled_datas(folder)
        
        X_train = [sent2features(s) for s in training_datas]
        y_train = [sent2labels(s) for s in training_datas]
        X_predict = [sent2features(s) for s in unlabeled_datas]
        y_pred,decision_scores = train_ner_and_predict(X_train,y_train,X_predict,folder,True,labels)
        for sentence_tuple, sentence_pred in zip(unlabeled_datas,y_pred):
            for word_tuple, pred in zip(sentence_tuple,sentence_pred):
                word_tuple = list(word_tuple)
                word_tuple[2] = pred
                word_tuple = tuple(word_tuple)
        bootstrap_sample_size = len(unlabeled_datas) // 5
        random.shuffle(unlabeled_datas)
        bootstrap_samples = [unlabeled_datas[i:i+bootstrap_sample_size] for i in range(0, len(unlabeled_datas), bootstrap_sample_size)]
        decision_scores_bags = []
        for bootstrap_sample in bootstrap_samples:
            bagging_training_data = training_datas+bootstrap_sample
            X_train_bag = [sent2features(s) for s in bagging_training_data]
            y_train_bag = [sent2labels(s) for s in bagging_training_data]
            X_predict_bag = [sent2features(s) for s in unlabeled_datas]
            y_pred_bag,decision_scores_bag = train_ner_and_predict(X_train_bag, y_train_bag, X_predict_bag, folder, False, labels)
            decision_scores_bags.append(decision_scores_bag)
        probability_scores = [[] for _ in range(len(unlabeled_datas))]
        for i in range(len(unlabeled_datas)):
            probability_scores[i] = [0] * len(unlabeled_datas[i])
            for j in range(len(unlabeled_datas[i])):
                labels_with_O = ['O'] + labels
                probabilities = {label: 0 for label in labels}
                for label in labels_with_O:
                    total_score = 0
                    for decision_scores_bag in decision_scores_bags:
                        total_score += decision_scores_bag[i][j][label]
                    probability = total_score/len(decision_scores_bags)
                    probabilities[label] = probability
                label_with_max_value = max(probabilities, key=lambda k: probabilities[k])
                unlabeled_datas[i][j] = list(unlabeled_datas[i][j])
                unlabeled_datas[i][j][2] = label_with_max_value
                unlabeled_datas[i][j] = tuple(unlabeled_datas[i][j])
                probability_scores[i][j] = probabilities[label_with_max_value]
        filtered_out_datas = filter_out_machine_labeled_datas(unlabeled_datas,probability_scores,lower_threshold,upper_threshold)
        machine_labeled_datas += filtered_out_datas
        training_datas += filtered_out_datas
    with open(output_file, 'w', encoding='utf-8') as file:
        for i in range(len(machine_labeled_datas)):
            for word, pos, label in machine_labeled_datas[i]:
                line = f"{word} {pos} _ {label}" 
                file.write(line)
                file.write("\n")
            file.write("\n")
            


def train_ner_and_predict(X_train,y_train,X_predict,folder_path,is_initial_model,labels):
    if is_initial_model:
        result_folder_path= os.path.join(folder_path, 'results/init')
    else:
        result_folder_path = os.path.join(folder_path,'results')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
        )
    params_space = {
    'c1': scipy.stats.expon(scale=0.5), #wider range of distribution of values of hyperparameters
    'c2': scipy.stats.expon(scale=0.05), #narrower range
    }
    f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)
     # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)
    results = {
        'best params': rs.best_params_,
        'best CV score': rs.best_score_,
        'model size': rs.best_estimator_.size_ / 1000000
    }
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    training_results_file = os.path.join(result_folder_path, 'training_results.json')
    with open(training_results_file, 'w') as file:
        json.dump(results, file)

    #Checking Parameter Space
    _x = [s['c1'] for s in rs.cv_results_['params']]
    _y = [s['c2'] for s in rs.cv_results_['params']]
    _c = [s for s in rs.cv_results_['mean_test_score']]
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
        min(_c), max(_c)
    ))

    ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])
    hyperparameter_plot_path = os.path.join(result_folder_path, 'hyperparameter_search_plot.png')
    plt.savefig(hyperparameter_plot_path)
    crf = rs.best_estimator_
    y_pred = crf.predict(X_predict)
    decision_scores = crf.predict_marginals(X_predict)
    return y_pred, decision_scores

def make_word_class_distribution(examples, output_file, labels):
    dict = {}
    for key in labels:
        dict[key] = {}
    for sentence in examples:
        for tuples in sentence:
            word = tuples[0]
            for key in dict.keys():
                if word not in dict[key].keys():
                    dict[key][word] = 0
            dict[tuples[2]][word] += 1
    with open(output_file, "w", encoding='utf-8') as outfile:
        json.dump(dict, outfile)

def get_pmi_data(word_class_distribution, label_map, smooth_param=1, class_alpha=1, word_alpha=1, is_pmi=True):
    pmi_data, class_freq = {}, {}
    word_freq = 0 #total number of words
    for class_key in label_map.keys():
        pmi_data[class_key] = {}
        class_freq[class_key] = 0 

    for class_key in word_class_distribution.keys():
        for key, val in word_class_distribution[class_key].items():
            word_class_distribution[class_key][key] += smooth_param #Smooth Param from argument before running, just to prevent division by zero
            #class key is e.g 'O', key is the sub word
            class_freq[class_key] += word_class_distribution[class_key][key]
        
        word_freq += class_freq[class_key]

    for class_key in word_class_distribution.keys():
        for key, val in word_class_distribution[class_key].items():
            cur_word_freq = 0
            for label_key in word_class_distribution.keys():
                cur_word_freq += word_class_distribution[label_key][key]
            
            numerator = word_class_distribution[class_key][key] / cur_word_freq
            class_prob = class_freq[class_key] / word_freq
            word_prob = cur_word_freq / word_freq
            denominator = class_freq[class_key] / word_freq * cur_word_freq / word_freq

            if is_pmi:
                pmi_data[class_key][key] = (np.log(numerator) - class_alpha * np.log(class_prob) - word_alpha * np.log(word_prob))
            else:
                pmi_data[class_key][key] = numerator / word_prob

    return pmi_data

def get_word_class_distribution(dataset, word_class_distribution,label_map):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            word = dataset[i][j][0]
            if word not in word_class_distribution['O']:
                for class_key in word_class_distribution.keys():
                    word_class_distribution[class_key][word] = 1  
            
            all_sum, label_list = 0, []
            for class_key in word_class_distribution.keys():
                all_sum += word_class_distribution[class_key][word]

            for class_key, class_idx in label_map.items():
                # training_set[i][j] = training_set[i][j] + (tf_train.A[i][word_index[0][0]],)
                dataset[i][j] = dataset[i][j] + (word_class_distribution[class_key][word]/all_sum,)

    return dataset

def train_ner_and_eval(train_folder,folder_path,labels,test_folder=None,is_smote=False,is_conll=True, algo="pa"):
    sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
    sorted_labels.insert(0,'O')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    training_datas = []
    for filename in os.listdir(train_folder):
        if is_conll == True:
            if filename.endswith('.conll'):
                input_path = os.path.join(train_folder, filename)
                training_data = open_conll(input_path)
                training_datas += training_data
        else:
            if filename == "train.txt":
                input_path = os.path.join(train_folder, filename)
                training_data = open_conll(input_path)
                training_datas += training_data
    if test_folder==None:
        training_set, evaluation_set = train_test_split(training_datas, test_size=0.2, random_state=42)
    else:
        training_set=training_datas
        evaluation_set = [] 
        for filename in os.listdir(test_folder):
            if filename.endswith('.conll'):
                input_path = os.path.join(test_folder, filename)
                evaluation_data = open_conll(input_path)
                evaluation_set += evaluation_data
    label_to_id = {}
    for i,label in enumerate(sorted_labels):
        label_to_id[label] = i

    X_final_train_lists = [sent2features(s) for s in training_set]
    y_final_train_lists = [sent2labels(s) for s in training_set]
    X_final_eval_lists = [sent2features(s) for s in evaluation_set]
    y_final_eval_lists = [sent2labels(s) for s in evaluation_set]
    freq_dict = None

    result_folder_path = os.path.join(folder_path, 'result_crf_training_and_eval')
    crf = sklearn_crfsuite.CRF(
        algorithm=algo,
        max_iterations=100,
        all_possible_transitions=True,
        all_possible_states=True
        )
    # hyperparameter search for lbfgs
    if algo=="lbfgs":
        params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
        # # 'averaging': [True, False],
        # # 'calibration_eta': scipy.stats.expon(scale=1.0),
        # # 'calibration_rate': scipy.stats.expon(scale=1.0),
        # 'epsilon': scipy.stats.expon(scale=1e-5),
        # # 'variance': scipy.stats.expon(scale=1.0),
        # 'delta': scipy.stats.expon(scale=1e-5),
        # 'linesearch': ['MoreThuente','Backtracking','StrongBackTracking']
        }

    #hyperparameter search for l2sgd
    elif algo=="l2sgd":
        params_space = {
        # 'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
        # 'averaging': [True, False],
        'calibration_eta': scipy.stats.expon(scale=1.0),
        'calibration_rate': scipy.stats.expon(scale=1.0),
        'calibration_samples': [500,1000,2000],
        'calibration_candidates': [5,10,20],
        'calibration_max_trials': [10,20,40],
        # 'epsilon': scipy.stats.expon(scale=1e-5),
        # 'variance': scipy.stats.expon(scale=1.0),
        'delta': scipy.stats.expon(scale=1e-5),
        # 'linesearch': ['MoreThuente','Backtracking','StrongBackTracking']
        }

    #hyperparameter search for ap
    elif algo=="ap":
        params_space = {
        # 'c1': scipy.stats.expon(scale=0.5),
        # 'c2': scipy.stats.expon(scale=0.05),
        # 'averaging': [True, False],
        # 'calibration_eta': scipy.stats.expon(scale=1.0),
        # 'calibration_rate': scipy.stats.expon(scale=1.0),
        # 'calibration_samples': [500,1000,2000],
        # 'calibration_candidates': [5,10,20],
        # 'calibration_max_trials': [10,20,40],
        'epsilon': scipy.stats.expon(scale=1e-5),
        # 'variance': scipy.stats.expon(scale=1.0),
        # 'delta': scipy.stats.expon(scale=1e-5),
        # 'linesearch': ['MoreThuente','Backtracking','StrongBackTracking']
        # 'pa_type': [0,1,2],
        # 'c': scipy.stats.expon(scale=1.0),
        # 'error_sensitive': [True,False],
        # 'averaging': [True,False],

        }

    #hyperparameter search for pa
    elif algo=="pa":
        params_space = {
        # 'c1': scipy.stats.expon(scale=0.5),
        # 'c2': scipy.stats.expon(scale=0.05),
        # 'averaging': [True, False],
        # 'calibration_eta': scipy.stats.expon(scale=1.0),
        # 'calibration_rate': scipy.stats.expon(scale=1.0),
        # 'calibration_samples': [500,1000,2000],
        # 'calibration_candidates': [5,10,20],
        # 'calibration_max_trials': [10,20,40],
        # 'epsilon': scipy.stats.expon(scale=1e-5),
        # 'variance': scipy.stats.expon(scale=1.0),
        # 'delta': scipy.stats.expon(scale=1e-5),
        # 'linesearch': ['MoreThuente','Backtracking','StrongBackTracking']
        'pa_type': [0,1,2],
        'c': scipy.stats.expon(scale=1.0),
        'error_sensitive': [True,False],
        'averaging': [True,False],
        }

    elif algo=="arow":
        params_space = {
        # 'c1': scipy.stats.expon(scale=0.5),
        # 'c2': scipy.stats.expon(scale=0.05),
        # 'averaging': [True, False],
        # 'calibration_eta': scipy.stats.expon(scale=1.0),
        # 'calibration_rate': scipy.stats.expon(scale=1.0),
        # 'calibration_samples': [500,1000,2000],
        # 'calibration_candidates': [5,10,20],
        # 'calibration_max_trials': [10,20,40],
        'epsilon': scipy.stats.expon(scale=1e-5),
        'variance': scipy.stats.expon(scale=1.0),
        'gamma': scipy.stats.expon(scale=1.0),
        # 'delta': scipy.stats.expon(scale=1e-5),
        # 'linesearch': ['MoreThuente','Backtracking','StrongBackTracking']
        # 'pa_type': [0,1,2],
        # 'c': scipy.stats.expon(scale=1.0),
        # 'error_sensitive': [True,False],
        # 'averaging': [True,False],
        }

    f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)
     # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=150,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer,
                            error_score='raise',
                            random_state=42)

    rs.fit(X_final_train_lists, y_final_train_lists)
    results = {
        'best params': rs.best_params_,
        'best CV score': rs.best_score_,
        'model size': rs.best_estimator_.size_ / 1000000
    }
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    training_results_file = os.path.join(result_folder_path, 'training_results.json')
    with open(training_results_file, 'w') as file:
        json.dump(results, file)
        file.write('\n')
        if freq_dict != None:
            for key, value in freq_dict.items():
                file.write("{}:{}\n".format(key,value))

    crf = rs.best_estimator_
    model_filename = os.path.join(result_folder_path,'crf.pkl')
    joblib.dump(crf, model_filename)
    y_pred = crf.predict(X_final_eval_lists)

    classification_report_file = os.path.join(result_folder_path,'classification_report.txt')
    with open(classification_report_file, "w") as output_file:
        # Write the classification report to the file
        
        flat_true = [item for sublist in y_final_eval_lists for item in sublist]
        flat_pred = [item for sublist in y_pred for item in sublist]

        report = classification_report(flat_true, flat_pred)
        output_file.write(report)

def load_model_and_predict(model_path,sentence,labels):
    sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
    sorted_labels.insert(0,'O')
    sentence = remove_unrelevant_lines_from_sentence(sentence)
    sentence = resolve_pronouns_in_text(sentence)
    lemmatized_sentence_tuples = get_pos_and_words(lemmatize_sentence(sentence))
    X = sent2features(lemmatized_sentence_tuples)

    loaded_crf = joblib.load(model_path)
    decision_scores = loaded_crf.predict_marginals([X])
    probability_scores = [0 for _ in range(len(X))]
    temp_y_pred = ['' for _ in range(len(X))]
    for i in range(len(X)):
        probabilities = {label: 0 for label in sorted_labels}
        for label in sorted_labels:
            probabilities[label] = decision_scores[0][i][label]
        label_with_max_value = max(probabilities, key=lambda k: probabilities[k])
        probability_scores[i] = probabilities[label_with_max_value]
        temp_y_pred[i] = label_with_max_value   
    
    feedback_criteria = []
    temp_criteria = ""
    for i in range(len(temp_y_pred)):
        if temp_y_pred[i] != 'O':
            temp_criteria += lemmatized_sentence_tuples[i][0] + " "
        else:
            if len(temp_criteria) != 0:
                feedback_criteria.append(temp_criteria)
            temp_criteria = ""
    print(feedback_criteria)


def load_model_and_test(model_path,test_folder,labels, result_folder):
    # lower_threshold = 0.51
    lower_threshold_positive = 0.51
    lower_threshold_outside = 0.51
    sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
    sorted_labels.insert(0,'O')
    evaluation_set = [] 
    for filename in os.listdir(test_folder):
        if filename.endswith('.conll'):
            input_path = os.path.join(test_folder, filename)
            evaluation_data = open_conll(input_path)
            evaluation_set += evaluation_data
    X_eval = [sent2features(s,sorted_labels) for s in evaluation_set]
    y_eval = [sent2labels(s) for s in evaluation_set]


    loaded_crf = joblib.load(model_path)
    decision_scores = loaded_crf.predict_marginals(X_eval)

    probability_scores = [[] for _ in range(len(X_eval))]
    temp_y_pred = [[] for _ in range(len(X_eval))]
    for i in range(len(X_eval)):
        probability_scores[i] = [0] * len(X_eval[i])
        temp_y_pred[i] = [''] * len(X_eval[i])
        for j in range(len(probability_scores[i])):
            probabilities = {label: 0 for label in sorted_labels}
            for label in sorted_labels:
                probabilities[label] = decision_scores[i][j][label]
            label_with_max_value = max(probabilities, key=lambda k: probabilities[k])
            probability_scores[i][j] = probabilities[label_with_max_value]
            temp_y_pred[i][j] = label_with_max_value   
    new_y_pred = []
    new_y_eval = []    
    lowest_positive_prob = 1.0
    lowest_outside_prob = 1.0
    for i in range(len(probability_scores)):
        # all_higher_than_lower_thres = all(score > lower_threshold for score in probability_scores[i])
        positives_higher_than_lower_thres = True
        for j in range(len(probability_scores[i])):
            if temp_y_pred[i][j]!='O' and probability_scores[i][j]<lowest_positive_prob:
                lowest_positive_prob = probability_scores[i][j]
            if temp_y_pred[i][j]=='O' and probability_scores[i][j]<lowest_outside_prob:
                lowest_outside_prob = probability_scores[i][j]
                
            if temp_y_pred[i][j]!='O' and probability_scores[i][j]<lower_threshold_positive:
                positives_higher_than_lower_thres = False
            if temp_y_pred[i][j]=='O' and probability_scores[i][j]<lower_threshold_outside:
                positives_higher_than_lower_thres = False
        # if(all_higher_than_lower_thres):
        #     new_y_pred.append(temp_y_pred[i])
        #     new_y_eval.append(y_eval[i])
        if(positives_higher_than_lower_thres):
            new_y_pred.append(temp_y_pred[i])
            new_y_eval.append(y_eval[i])
    results_file = os.path.join(result_folder, 'selected_classification_report.txt')
    with open(results_file, "w", encoding='utf-8') as output_file:
        report_dict = calculate_classification_report_for_each_labels(sorted_labels,new_y_eval,new_y_pred)
        json.dump(report_dict,output_file)

def prepare_unlabeled_datas(folder_path):
    unlabeled_datas = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            input_path = os.path.join(folder_path, filename)
            with open(input_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    line_tuples = get_pos_and_words(line)
                    unlabeled_datas.append(line_tuples)
    return unlabeled_datas

def filter_out_machine_labeled_datas(machine_labeled_datas, probability_scores, lower_threshold, upper_threshold):
    filtered_out_datas = []
    for i in range(len(machine_labeled_datas)):
        all_higher_than_lower_thres = all(score > lower_threshold for score in probability_scores[i])
        at_least_one_higher_than_upper = any(score > upper_threshold for score in probability_scores[i])
        if(all_higher_than_lower_thres and at_least_one_higher_than_upper):
            filtered_out_datas.append(machine_labeled_datas[i])
    return filtered_out_datas

def generate_features(numbers):
    for i, num in enumerate(numbers):
        yield {"feature{}".format(i): num}

def calculate_classification_report_for_each_labels(labels,y_eval,y_pred):
    results_dict = dict.fromkeys(labels)
    for label in labels:
        results_dict[label] = {
            "precision": 0,  # You can set initial values if needed
            "recall": 0,
            "f1-score": 0,
            "support": 0
        }
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        support = 0
        for i in range(len(y_eval)):
            assert len(y_eval[i]) == len(y_pred[i])
            for j in range(len(y_eval[i])):
                if y_pred[i][j]!=label and y_eval[i][j]!=label:
                    tn += 1
                elif y_pred[i][j]==label and y_eval[i][j]==label:
                    tp +=1
                elif y_pred[i][j]==label and y_eval[i][j]!=label:
                    fn += 1
                else:
                    fp += 1
                if y_eval[i][j]==label:
                    support += 1
        results_dict[label]["precision"] = tp/(tp+fp)
        results_dict[label]["recall"] = tp/(tp+fn)
        results_dict[label]["f1"] = (2*results_dict[label]["recall"]*results_dict[label]["precision"])/(results_dict[label]["recall"]+results_dict[label]["precision"])
        results_dict[label]["support"] = support
    return results_dict

def get_class_freqs(y_bal,labels):
    class_dict = {label: 0 for label in labels}
    for tag in y_bal:
        class_dict[labels[tag]] +=1
    return class_dict

def custom_tokenizer(text):
    # Implement your custom tokenization logic here
    tokens = text.split()  # For example, split by whitespace
    return tokens

def clean_and_normalize_data(data):
    new_data = []
    characters_to_remove = string.punctuation + string.printable.replace(' ', '')[62:] + '1234567890'
    translator = str.maketrans('', '', characters_to_remove)
    for i in range(len(data)):
        temp_sentence = []
        for j in range(len(data[i])):
            clean_word = data[i][j][0].translate(translator)
            clean_word = clean_word.strip()
            if clean_word:
                temp_sentence.append((clean_word,) + data[i][j][1:])
            else:
                continue
        new_data.append(temp_sentence)
    return new_data


if __name__ == "__main__":
    #train_ner_and_eval("NON_BIO/sr_augmented_new_2_train_data_conll_non_bio","NON_BIO/sr_augmented_new_2_train_data_conll_non_bio/ner_crf_optimized_pa_corrected",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False)
    # load_model_and_test("NON_BIO/sr_augmented_new_2_train_data_conll_non_bio/ner_crf_optimized_pa2/result_crf_training_and_eval/crf.pkl","NON_BIO/new_2_test_data_conll_non_bio",['Feedback_Criteria'],"NON_BIO/sr_augmented_new_2_train_data_conll_non_bio/ner_crf_optimized_pa2/result_crf_training_and_eval")
    # train_ner_and_eval("BIO/new_2_train_data_conll","BIO/new_2_train_data_conll/ner_crf",['B-Feedback_Criteria','I-Feedback_Criteria'],"BIO/new_2_test_data_conll",is_smote=False)
    # train_ner_and_eval("BIO/sr_augmented_new_2_train_data_conll","BIO/sr_augmented_new_2_train_data_conll/ner_crf_optimized_pa",['B-Feedback_Criteria','I-Feedback_Criteria'],"BIO/new_2_test_data_conll",is_smote=False)

    ## NER_CRF NON_BIO Chosen
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_optimized_ap_corrected",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo="ap")
    # train_ner_and_eval("NON_BIO/sr_augmented_new_2_train_data_conll_non_bio","NON_BIO/sr_augmented_new_2_train_data_conll_non_bio/ner_crf_optimized_ap_corrected",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo="ap")
    # train_ner_and_eval("NON_BIO/bus_new_2_train_data_conll_non_bio","NON_BIO/bus_new_2_train_data_conll_non_bio/ner_crf_optimized_ap_corrected",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo="ap")
    # train_ner_and_eval("NON_BIO/nsCRD_new_2_train_data_conll_non_bio","NON_BIO/nsCRD_new_2_train_data_conll_non_bio/ner_crf_optimized_ap_corrected",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo="ap")

    ## NER_CRF BIO Chosen
    # train_ner_and_eval("BIO/new_2_train_data_conll" ,"BIO/new_2_train_data_conll/ner_crf_optimized_pa_corrected",['B-Feedback_Criteria','I-Feedback_Criteria'],'BIO/new_2_test_data_conll',is_smote=False)
    # train_ner_and_eval("BIO/sr_augmented_new_2_train_data_conll" ,"BIO/sr_new_2_train_data_conll/ner_crf_optimized_pa_corrected",['B-Feedback_Criteria','I-Feedback_Criteria'],'BIO/new_2_test_data_conll',is_smote=False)
    # train_ner_and_eval("BIO/bus_new_2_train_data_conll" ,"BIO/bus_new_2_train_data_conll/ner_crf_optimized_pa_corrected",['B-Feedback_Criteria','I-Feedback_Criteria'],'BIO/new_2_test_data_conll',is_smote=False)
    # train_ner_and_eval("BIO/nsCRD_new_2_train_data_conll" ,"BIO/nsCRD_new_2_train_data_conll/ner_crf_optimized_pa_corrected",['B-Feedback_Criteria','I-Feedback_Criteria'],'BIO/new_2_test_data_conll',is_smote=False)

    ## NER_CRF NON_BIO algo
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_optimized_lbfgs",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo='lbfgs')
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_optimized_l2sgd",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo='l2sgd')
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_optimized_ap",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo='ap')
    # # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_optimized_pa_corrected_macro",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo='pa')
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_optimized_arow",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo='arow')
    
    # NER_CRF BIO
    # train_ner_and_eval("BIO/new_2_train_data_conll","BIO/new_2_train_data_conll/ner_crf_optimized_lbfgs",['B-Feedback_Criteria','I-Feedback_Criteria'],"BIO/new_2_test_data_conll",is_smote=False,algo='lbfgs')
    # train_ner_and_eval("BIO/new_2_train_data_conll","BIO/new_2_train_data_conll/ner_crf_optimized_l2sgd",['B-Feedback_Criteria','I-Feedback_Criteria'],"BIO/new_2_test_data_conll",is_smote=False,algo='l2sgd')
    # train_ner_and_eval("BIO/new_2_train_data_conll","BIO/new_2_train_data_conll/ner_crf_optimized_ap",['B-Feedback_Criteria','I-Feedback_Criteria'],"BIO/new_2_test_data_conll",is_smote=False,algo='ap')
    # # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_optimized_pa_corrected_macro",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False,algo='pa')
    # train_ner_and_eval("BIO/new_2_train_data_conll","BIO/new_2_train_data_conll/ner_crf_optimized_arow",['B-Feedback_Criteria','I-Feedback_Criteria'],"BIO/new_2_test_data_conll",is_smote=False,algo='arow')

    load_model_and_predict("NON_BIO/new_2_train_data_conll_non_bio/ner_crf_optimized_ap_corrected/result_crf_training_and_eval/crf.pkl","= = = = = Teacher_feedback = = = = = insgesamt : ich halten ihr Analyse für gut gelingen , allerdings sein mir der Punkt der Sprachproduktion noch etwas untergehen , gerade bei der Aufgabe , der schon ein Diskursfunktion enthalten müssen auch überlegen werden , wie ein Lösung hier aussehen können ( = der Sprachprodukt ) und dann auch überlegen werden , welche sprachlich Mittel darin ein Hürde darstellen und irgendwie spät bei der Unterrichtsplanung als Sprachhilfe einbauen werden sollen . der Punkt der Sprachproduktion haben sie nicht vollständig ausführen .",['Feedback_Criteria'])