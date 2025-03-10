from data_preprocessing import separate_students_feedbacks_texts, filter_out_same_lines_with_checks, remove_duplicate_lines,remove_unrelevant_lines, extract_teacher_model_answer,count_texts_in_folder,filter_out_same_lines,resolve_pronouns_in_text,resolve_pronouns,separate_feedbacks_into_sentences,separate_into_sentences,copy_feedback_only, sentence_splitter_somajo, sentence_splitter_bert, separate_feedbacks_into_sentences_somajo, lemmatize_folder, lemmatize_file,lemmatize_file_with_output, extract_non_outside_training_sentences_to_file,filter_outside_training_sentences, split_conll_data
import pandas as pd
from utils import read_feedback_texts, tokenize_text, json_to_excel, join_sentences_in_folder,combine_lines_in_folder, identify_keywords_rake, open_conll, convert_bio_tags_to_one_tag,get_class_freq_from_training_datas,get_key_by_value, join_conll_files_in_folder, copy_into_folders, get_number_of_words_in_conll
import os
from ner_crf import generate_machine_labeled_datas, train_ner_and_eval
from datasets import load_dataset
from NER_Adaptive_Resampling import NER_Adaptive_Resampling
from py_openthesaurus import OpenThesaurusDb
from augment import generate_sentences_by_synonym_replacement
from augment_script import Augment_data
import nltk
from nltk.corpus import stopwords


nltk.download("stopwords")
stop_words = set(stopwords.words("german"))


def split_sentences_material(folder_path,output_folder):
    """
    Split sentences from all files found in folder_path and write it to output_folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only text files
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            sentences =  sentence_splitter_somajo(input_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                for sentence in sentences:
                    f.write(str(sentence) + '\n')

def resample_datas(input_folder_path, output_folder_path, method):
    """
    Resample data with one of the method sc, sCR, sCRD, nsCRD
    Parameter method must be one of the listed above
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.conll'):
            input_path = os.path.join(input_folder_path, filename)
            output_path = os.path.join(output_folder_path, filename)
            ner_Resampling = NER_Adaptive_Resampling(input_path,output_path)
            ner_Resampling.resamp(method)

def resample_datas_bus(input_folder_path, output_folder_path):
    """
    Resample data with BUS
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.conll'):
            input_path = os.path.join(input_folder_path, filename)
            output_path = os.path.join(output_folder_path, filename)
            ner_Resampling = NER_Adaptive_Resampling(input_path,output_path)
            ner_Resampling.BUS()

def augment_datas(input_folder_path, output_folder_path,method, is_conll=True):
    """
    Augment data using one of the method SR, LwTR, MR, SiS
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for filename in os.listdir(input_folder_path):
        if is_conll ==True:
            if filename.endswith('.conll'):
                input_path = os.path.join(input_folder_path, filename)
                output_path = os.path.join(output_folder_path, filename)
                Augment_data(input_path,output_path,method)
        else:
            if filename == "train.txt":
                input_path = os.path.join(input_folder_path, filename)
                output_path = os.path.join(output_folder_path, filename)
                Augment_data(input_path,output_path,method)


def get_class_freq_from_multiple_training_datas(label_to_id,train_paths_list,output_file,method_list):
    """
    Get class frequency from all conl datas listed in train_paths_list
    """
    with open(output_file,"a", encoding="utf-8") as output_file:

        for train_path,method in zip(train_paths_list,method_list):
            output_file.write(method+"\n")
            training_datas = []
            for filename in os.listdir(train_path):
                if filename.endswith('.conll'):
                    input_path = os.path.join(train_path, filename)
                    training_data = open_conll(input_path)
                    training_datas+= training_data
            class_freq=get_class_freq_from_training_datas(label_to_id,training_datas)
            for i in range(len(class_freq)):
                output_file.write(str(get_key_by_value(label_to_id,i)) +": "+ str(class_freq[i]) +"\n")
            output_file.write("\n")
    output_file.close()

def filter_training_data(train_folder_path, output_folder_path):
    """
    Filter training data in a conll file to only contain sentences without sentences only containing outside tokens
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for filename in os.listdir(train_folder_path):
        if filename.endswith('.conll'):
            input_path = os.path.join(train_folder_path, filename)
            training_data = open_conll(input_path)
            separated_training_data = filter_outside_training_sentences(training_data)
            filtered_training_data = []
            for sentence in separated_training_data:
                filtered_sentence = []
                for tuple in sentence:
                    if tuple[2]!="O" or tuple[0].lower() not in stop_words:
                        filtered_sentence.append(tuple)
                filtered_training_data.append(filtered_sentence)
            output_path = os.path.join(output_folder_path, filename)
            with open(output_path,"w",encoding="utf-8") as output_file:
                for sentence in filtered_training_data:
                    for tuple in sentence:
                        output_file.write(str(tuple[0])+ " " + str(tuple[1]) + " " + str(tuple[2]) + "\n")
                    output_file.write("\n")
            output_file.close()

def split_train_test_devel_data(input_folders):
    """
    Split the train.txt, test.txt, devel.txt files in all folders listed in input_folders so that one sentence has maximum 64 tokens
    """
    for folder in input_folders:
        split_conll_data(os.path.join(folder,"train.txt"),os.path.join(folder,"train.txt"))
        split_conll_data(os.path.join(folder,"test.txt"),os.path.join(folder,"test.txt"))
        split_conll_data(os.path.join(folder,"devel.txt"),os.path.join(folder,"devel.txt"))

def remove_new_line(folder,file_names):
    """
    Remove new line characters at end of files
    """
    for file_name in file_names:
        input_file_path = os.path.join(folder,file_name)
        output_file_path = os.path.join(folder,file_name)

        # Read the input file
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            file_content = input_file.read()

        # Check if the last character is a newline character
        if file_content.endswith("\n"):
            # Remove the last character (newline)
            file_content = file_content[:-1]

        # Write the modified content back to the output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(file_content)

def remove_new_lines_from_folders(folders,file_names):
    """
    Remove new lines characters from all files listed in folders
    """
    for folder in folders:
        remove_new_line(folder,file_names)

def create_tagset(files):
    """
    Create a POS tagset based on the conll data found in files
    """
    tagset = []
    for file in files:
        data = open_conll(file)
        for sentence in data:
            for tuple in sentence:
                if tuple[1] not in tagset:
                    tagset.append(tuple[1])
    with open("tagset.txt", "w", encoding="utf-8") as output_file:
        for word in tagset:
        # Write each word to a separate line in the file
            output_file.write(word + "\n")



if __name__ == "__main__":
    #### Extract student and teacher feedback from excel file
    # for x in range(1,6):
    #     excelfile = f'Aufgabe{x}/Students&FeedbackTexts{x}.xlsx'
    #     separate_students_feedbacks_texts(excelfile, x)

    #### Preparation for Materials to be used to train MLM ####

    #### Text Cleaning
    # for i in range(1,6):
    #     remove_unrelevant_lines(f"Aufgabe{i}/feedbacktexts",i)
    #extract_teacher_model_answer("Aufgabe1")
    # remove_duplicate_lines("Aufgabe1/filteredfeedbacktexts/feedbacktexts1.txt")
    # count = count_texts_in_folder("Aufgabe1/teacherModelAnswer")
    # for i in range(1,count+1):
    #     filter_out_same_lines(f"Aufgabe1/teacherModelAnswer/modelAnswer{i}.txt",f"Aufgabe1/filteredfeedbacktexts/feedbacktexts{i}.txt")
    # filter_out_same_lines(f"Aufgabe2/teacherModelAnswer/modelAnswer4.txt",f"Aufgabe2/filteredfeedbacktexts/feedbacktexts4.txt")
    # filter_out_same_lines_with_checks("Aufgabe1")
    # filter_out_same_lines_with_checks("Aufgabe2")

    #### Coreference Resolution
    # for i in range(1,6):
    #     resolve_pronouns_all_feedbacks(f"Aufgabe{i}/filteredfeedbacktexts")
    #separate_feedbacks_into_sentences(1)
    # for i in range(1,6):
    #     separate_feedbacks_into_sentences(i)
    # for i in range(5,16):
    #     copy_feedback_only(f"Aufgabe1/studenttexts/studenttexts{i}.txt",f"Aufgabe1/filteredfeedbacktexts/feedbacktexts{i}.txt",i,1)
    # for i in range(1,6):
    #     separate_feedbacks_into_sentences_somajo(i)
    
    
    # join_sentences_in_folder("Materials","joined_Materials")
    # combine_lines_in_folder("joined_Materials","one_line_Materials")
    # split_sentences_material("one_line_Materials","somajo_splitted_Materials")
    # lemmatize_folder("somajo_splitted_Materials","lemmatized_Materials")
    
    #### Preparation for Materials to be used to train MLM ####

    #### Preparations for materials to be tagged ####

    # for i in range(1,3):
    #     remove_unrelevant_lines(f"Aufgabe{i}/teacherModelAnswer",f"Aufgabe{i}/teacherModelAnswer")
    #     resolve_pronouns(f"Aufgabe{i}/teacherModelAnswer",f"Aufgabe{i}/teacherModelAnswer")
    #     lemmatize_folder(f"Aufgabe{i}/teacherModelAnswer",f"Aufgabe{i}/lemmatizedTeacherModelAnswer")
    # for i in range(1,6):
    #     lemmatize_file_with_output(f"Aufgabe{i}/Aufgabe{i}-Description.txt",f"Aufgabe{i}/Aufgabe{i}-Description_lemmatised.txt")
        # lemmatize_file_with_output("Misc-Feedback.txt","Misc-Feedback-lemmatized.txt")

    #### Preparations for materials to be tagged ####

    # generate_machine_labeled_datas("train_data_conll", "lemmatized_Materials_unlabeled",['B-Feedback_Criteria','I-Feedback_Criteria'],'train_data_conll/machine_labeled_datas.conll')
    
    # convert_bio_tags_to_one_tag("train_data_conll_original","train_data_conll_original_non_bio")
    # convert_bio_tags_to_one_tag("train_data_conll","train_data_conll_non_bio")
    # convert_bio_tags_to_one_tag("test_data_conll","test_data_conll_non_bio")

    # label_to_id = {'O':0, 'B-Feedback_Criteria':1, 'I-Feedback_Criteria':2}

    #Data Resampling BIO
    # resample_datas_bus("BIO/new_2_train_data_conll","BIO/bus_new_2_train_data_conll")
    # resample_datas("BIO/new_2_train_data_conll","BIO/sc_new_2_train_data_conll",'sc')
    # resample_datas("BIO/new_2_train_data_conll","BIO/nsCRD_new_2_train_data_conll",'nsCRD')

    #Data Resampling NON_BIO
    # resample_datas_bus("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/bus_new_2_train_data_conll_non_bio")
    # resample_datas("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/nsCRD_new_2_train_data_conll_non_bio","nsCRD")
    # resample_datas("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/sc_new_2_train_data_conll_non_bio","sc")

    # augment_datas("train_data_conll","sr_augmented_train_data_conll",'SR')
    # augment_datas("train_data_conll","mr_augmented_train_data_conll",'MR')
    # augment_datas("train_data_conll","lwtr_augmented_train_data_conll",'LwTR')
    # augment_datas("train_data_conll","sis_augmented_train_data_conll",'SiS')
    # label_to_id = {'O':0, 'B-Feedback_Criteria':1, 'I-Feedback_Criteria':2}

    # augment_datas("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/sr_augmented_new_2_train_data_conll_non_bio",'SR')
    # augment_datas("BIO/new_2_train_data_conll","BIO/sr_augmented_new_2_train_data_conll","SR")
    label_to_id = {'O':0, 'B-Feedback_Criteria':1, 'I-Feedback_Criteria':2}

    # train_ner_and_eval("NON_BIO/sc_train_data_conll","NON_BIO/sc_train_data_conll/ner_crf",['Feedback_Criteria'],"NON_BIO/test_data_conll_non_bio")
    # train_ner_and_eval("NON_BIO/sCR_train_data_conll","NON_BIO/sCR_train_data_conll/ner_crf",['Feedback_Criteria'],"NON_BIO/test_data_conll_non_bio")
    # train_ner_and_eval("NON_BIO/sCRD_train_data_conll","NON_BIO/sCRD_train_data_conll/ner_crf",['Feedback_Criteria'],"NON_BIO/test_data_conll_non_bio")
    # train_ner_and_eval("NON_BIO/nsCRD_train_data_conll","NON_BIO/nsCRD_train_data_conll/ner_crf",['Feedback_Criteria'],"NON_BIO/test_data_conll_non_bio")

    # train_ner_and_eval("NON_BIO/sr_augmented_train_data_conll","NON_BIO/sr_augmented_train_data_conll/ner_crf",['Feedback_Criteria'],"NON_BIO/test_data_conll_non_bio")
    # train_ner_and_eval("NON_BIO/sis_augmented_train_data_conll","NON_BIO/sis_augmented_train_data_conll/ner_crf",['Feedback_Criteria'],"NON_BIO/test_data_conll_non_bio")
    # train_ner_and_eval("NON_BIO/lwtr_augmented_train_data_conll","NON_BIO/lwtr_augmented_train_data_conll/ner_crf",['Feedback_Criteria'],"NON_BIO/test_data_conll_non_bio")

    # train_ner_and_eval("NON_BIO/new_train_data_conll_non_bio","NON_BIO/new_train_data_conll_non_bio/ner_crf",['Feedback_Criteria'],"NON_BIO/new_test_data_conll_non_bio")
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_smote",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=True)
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False)
    # train_ner_and_eval("NON_BIO/sr_augmented_new_2_train_data_conll_non_bio","NON_BIO/sr_augmented_new_2_train_data_conll_non_bio/ner_crf",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False)
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_tf_tfidf",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False)
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_clean",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False)
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_tf_tfidf_same_vectorizer",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False)
    # train_ner_and_eval("NON_BIO/new_2_train_data_conll_non_bio","NON_BIO/new_2_train_data_conll_non_bio/ner_crf_with_pmi",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False)
    # train_ner_and_eval("NON_BIO/sr_augmented_new_2_train_data_conll_non_bio","NON_BIO/sr_augmented_new_2_train_data_conll_non_bio/ner_crf_with_pmi",['Feedback_Criteria'],"NON_BIO/new_2_test_data_conll_non_bio",is_smote=False)

    #PMI BIO
    # train_paths_list = ["BIO/sr_augmented_new_2_train_data_conll","BIO/sc_new_2_train_data_conll","BIO/nsCRD_new_2_train_data_conll","BIO/bus_new_2_train_data_conll","BIO/new_2_train_data_conll"]
    # # join_conll_files_in_folder("BIO/new_2_train_data_conll","train.txt")
    # # join_conll_files_in_folder("BIO/new_2_test_data_conll","test.txt")
    # # join_conll_files_in_folder("BIO/new_2_test_data_conll","devel.txt")
    # for folder_name in train_paths_list:
    #     join_conll_files_in_folder(folder_name,"train.txt")
    # copy_into_folders(train_paths_list,["BIO/new_2_test_data_conll/test.txt","BIO/new_2_test_data_conll/devel.txt"],["test.txt","devel.txt"])
    # split_train_test_devel_data(train_paths_list)
    # file_names = ["train.txt","devel.txt","test.txt"]
    # remove_new_lines_from_folders(train_paths_list,file_names)

    #PMI NON_BIO
    # train_paths_list = ["NON_BIO/sr_augmented_new_2_train_data_conll_non_bio","NON_BIO/sc_new_2_train_data_conll_non_bio","NON_BIO/nsCRD_new_2_train_data_conll_non_bio","NON_BIO/bus_new_2_train_data_conll_non_bio"]
    # train_paths_list = ["NON_BIO/new_2_train_data_conll_non_bio"]
    # for folder_name in train_paths_list:
    #     join_conll_files_in_folder(folder_name,"train.txt")
    # # join_conll_files_in_folder("NON_BIO/new_2_test_data_conll_non_bio","test.txt")
    # # join_conll_files_in_folder("NON_BIO/new_2_test_data_conll_non_bio","devel.txt")
    # copy_into_folders(train_paths_list,["NON_BIO/new_2_test_data_conll_non_bio/test.txt","NON_BIO/new_2_test_data_conll_non_bio/devel.txt"],["test.txt","devel.txt"])
    # split_train_test_devel_data(train_paths_list)
    # file_names = ["train.txt","devel.txt","test.txt"]
    # remove_new_lines_from_folders(train_paths_list,file_names)
    
    # create_tagset(["NON_BIO/new_2_train_data_conll_non_bio/train.txt","NON_BIO/new_2_test_data_conll_non_bio/test.txt"])
    