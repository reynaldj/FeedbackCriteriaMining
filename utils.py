import os
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import json
import random
from HanTa import HanoverTagger as ht
import re
from datasets import Dataset
import shutil
# from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('german'))
stop_words.update(['?', ')', '...', '!', ':', ';', ',', '.', '(', '[', ']','/','„','“','``'])
tagger = ht.HanoverTagger('morphmodel_ger.pgz')

def read_feedback_texts(folder_path):
    """
    Read all the feedback file and save it in an array
    """
    feedback_texts = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            feedback_texts.append(file.read())
    return feedback_texts

def json_to_excel(dir,file):
    """
    Convert json file to excel
    """
    with open(dir+file, 'r') as f:
        data = json.load(f)

    # Initialize lists to store data
    labels = []
    texts = []
    entity_texts = []

    # Iterate through each entry in the data
    for entry in data:
        label = entry["label"]
        text = entry["text"]
        
        for entity in entry["ents"]:
            entity_id, start, end, confidence = entity
            entity_text = text[start:end]
            
            labels.append(label)
            texts.append(text)
            entity_texts.append(entity_text)

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'Label': labels,
        'Text': texts,
        'Entity Text': entity_texts
    })
    df.to_excel(dir+"output.xlsx", index=False)

def join_sentences_in_folder(folder_path,output_folder):
    """
    Join sentences that are separated when in pdf format because of new line
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only text files
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            
            with open(input_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
            combined_text = input_text.replace('-\n', '')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)

def combine_lines(text):
    """
    Combine lines that are separated by newline in a text
    """
    lines = text.split('\n')
    combined_lines = ' '.join(lines)
    return combined_lines

def combine_lines_in_folder(folder_path,output_folder):
    """
    For all text files in a folder, combine all lines into one line
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only text files
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            
            with open(input_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
            combined_text = combine_lines(input_text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)

def shuffle_sentences(folder_path):
    """
    Shuffle the order of sentences in a file
    """
    all_sentences = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only text files
            input_path = os.path.join(folder_path, filename)
            with open(input_path, 'r', encoding='utf-8') as f:
                sentences = f.readlines()
                all_sentences.extend(sentences)
    random.shuffle(all_sentences)
    return all_sentences

def open_conll(file_path):
    """
    Open a conll file and save all the words as list of tuples
    1st tuple is the token, 2nd is pos, 3rd is label
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    training_data = []
    line_tuples = []
    for line in lines:
        if line.isspace():
            sentence_list = [item[0] for item in line_tuples]
            sentence = " ".join(sentence_list)
            pos_list = get_pos(sentence)
            words_and_pos = get_pos_and_words(sentence)
            stack = 0
            sentence_tuples = []
            line_tuples_counter = 0
            # for i, (token, _, ner) in enumerate(line_tuples):
            #     if token == words_and_pos[i][0]:
            #         pos = pos_list[i]
            #         # line_tuples[i] = (token, pos, ner)
            #         sentence_tuples.append((token, pos, ner))
            #     elif len(token) < len(words_and_pos[i][0]):
            #         counter = 0
            #         temp_token = token
            #         while temp_token != words_and_pos[i][0]:
            #             counter += 1
            #             temp_token += line_tuples[i+counter][0]
            #             stack += 1
            #         sentence_tuples.append((words_and_pos[i][0],))
            for i,(_,_,_) in enumerate(words_and_pos):
                if stack > 0:
                    stack -= 1
                    continue
                if line_tuples[line_tuples_counter][0] == words_and_pos[i][0]:
                    sentence_tuples.append((words_and_pos[i][0],words_and_pos[i][1],line_tuples[line_tuples_counter][2]))
                elif len(line_tuples[line_tuples_counter][0]) < len(words_and_pos[i][0]):
                    temp_token = line_tuples[line_tuples_counter][0]
                    while temp_token != words_and_pos[i][0]:
                        line_tuples_counter += 1
                        temp_token += line_tuples[line_tuples_counter][0]
                    sentence_tuples.append((words_and_pos[i][0],words_and_pos[i][1],line_tuples[line_tuples_counter][2]))
                elif len(words_and_pos[i][0]) < len(line_tuples[line_tuples_counter][0]):
                    temp_token = words_and_pos[i][0]
                    while temp_token != line_tuples[line_tuples_counter][0]:
                        stack += 1
                        temp_token += words_and_pos[i+stack][0]
                    for j in range(stack+1):
                        sentence_tuples.append((words_and_pos[i+j][0],words_and_pos[i+j][1],line_tuples[line_tuples_counter][2]))
                line_tuples_counter += 1
            assert line_tuples_counter == len(line_tuples)    
            assert len(sentence_tuples) == len(words_and_pos)                 
            temp = sentence_tuples
            training_data.append(temp)
            line_tuples.clear()
        elif line.startswith("-DOCSTART-"):
            continue
        else:
            tokens = line.split()
            token = tokens[0]
            ner_tag = tokens[-1]
            line_tuples.append((token, '_', ner_tag))
    return training_data
                
    


def get_pos(sentence):
    """
    Get POS tag of all words in a sentence
    """
    words = nltk.word_tokenize(sentence)
    lemmata = tagger.tag_sent(words)
    return [item[2] for item in lemmata]

def get_pos_and_words(sentence):
    """
    Return tuples, 1st tuple is the word itself, 2nd is POS tag, 3rd is just a place holder
    """
    words = nltk.word_tokenize(sentence)
    lemmata = tagger.tag_sent(words)
    return [(item[0], item[2], '_') for item in lemmata]

def get_key_by_value(dictionary, target_value):
    """
    Get key in a dictionary given its corresponding value
    """
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None

def get_class_freq_from_training_datas(label_to_id,training_datas):
    """
    Get the class frequencies (number of words that belong to that class) in a training data
    """
    class_freq = [0] * len(label_to_id.items())
    for training_data in training_datas:
        for tuple in training_data:
            index = label_to_id[tuple[2]]
            class_freq[index]  += 1
    return class_freq

def convert_bio_tags_to_one_tag(input_folder_path, output_folder_path):
    """
    Convert BIO tag to regular tag e.g 'B-Feedback_Criteria' and 'I-Feedback_Criteria' will be considerd same entity to become 'Feedback_Criteria'
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for filename in os.listdir(input_folder_path):
            if filename.endswith('.conll'):
                input_file_path = os.path.join(input_folder_path,filename)
                output_file_path = os.path.join(output_folder_path,filename)
                with open(input_file_path,"r", encoding="utf-8") as input_file, open(output_file_path, "w", encoding="utf-8") as output_file:
                    for line in input_file:
                        # Remove leading and trailing whitespaces
                        line = line.strip()

                        # Check if the line contains the desired prefixes
                        line = re.sub(r'B-Feedback_Criteria\b', 'Feedback_Criteria', line)
                        line = re.sub(r'I-Feedback_Criteria\b', 'Feedback_Criteria', line)

                        # Write the modified line to the output file
                        output_file.write(line + "\n")
                input_file.close()
                output_file.close()

def convert_to_datasets_format(training_datas,label_to_id):
    """
    Convert tuples of training data (word,pos_tag,label) returned by open_conll to dataset format
    """
    formatted_data = {
        'tokens': [],
        'pos_tags': [],
        'ner_tags': [],
        }
    for sentence in training_datas:
        tokens, pos_tags, ner_tags = zip(*sentence)
        ner_tags_ids = [label_to_id[label] for label in ner_tags]
        formatted_data['tokens'].append(tokens)
        formatted_data['pos_tags'].append(pos_tags)
        formatted_data["ner_tags"].append(ner_tags_ids)
    return Dataset.from_dict(formatted_data)

def get_sentence_from_conll_tuples(sentence_tuples):
    """
    Get the actual sentence from the tuples of (word,pos_tag,label)
    """
    sentence=""
    for word_tuple in sentence_tuples:
        word = word_tuple[0]
        sentence += word + " "
    return sentence.strip()

def join_conll_files_in_folder(folder_path,file_name):
    """
    Join all conll files in a folder into one file
    """
    conll_files = [f for f in os.listdir(folder_path) if f.endswith(".conll")]
    output_file_path = os.path.join(folder_path, file_name)
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for conll_file in conll_files:
            conll_file_path = os.path.join(folder_path, conll_file)
            with open(conll_file_path, "r", encoding="utf-8") as input_file:
                # Read the content of the CoNLL file and write it to the output file
                content = input_file.read()
                output_file.write(content)
                output_file.write("\n\n")
            
def copy_into_folders(destination_folders,file_sources,file_names):
    """
    Copy all files in files found in file_sources into destination_folders with the name as destination_folder + file_names[i]
    """
    for destination_folder in destination_folders:
        # Define the destination file paths
        for i in range(len(file_sources)):
            shutil.copy(file_sources[i], os.path.join(destination_folder,file_names[i]))

def get_number_of_words_in_conll(data_path):
    """
    Get number of words found in a conll file
    """
    total_words = 0
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == '' or line.startswith('-DOCSTART-'):
                continue  # Skip empty lines or document start lines
            columns = line.split('\t')
            if len(columns) > 0:
                total_words += 1
    return total_words

def concatenate_files(source_file, files_to_concatenate, output_file):
    """
    concatenate file from files_to_concatenate into source_file and save it as output_file
    """
    source_data = open_conll(source_file)
    for file in files_to_concatenate:
        data = open_conll(file)
        source_data += data
    with open(output_file, 'w', encoding="utf-8") as of:
        for sentence in source_data:
            for tuples in sentence:
                of.write(tuples[0] + ' ' + tuples[1] + ' ' + tuples[2] + '\n')
            of.write('\n')
    

    

if __name__ == "__main__":
    # labels = ["O","B-Feedback_Criteria","I-Feedback_Criteria"]
    labels = ["O","Feedback_Criteria"]
    # concatenate_files("NON_BIO/train_data_conll_non_bio/train.txt",["NON_BIO/new_2_train_data_conll_non_bio/lemmatised_aufgabeDescription_02.conll","NON_BIO/new_2_train_data_conll_non_bio/lemmatised_feedback_02.conll","NON_BIO/new_2_train_data_conll_non_bio/lemmatised_teacherModelAnswer_02.conll"],"NON_BIO/new_2_train_data_conll_non_bio/train.txt")
    # Sample training and test data
    # original_tuple = (1, 2)

    # # Element to add
    # new_element = 3
    # new_element2 = 4

    # # Modify the original tuple to include the new element
    # modified_tuple = original_tuple + (new_element,)

    # # Output the modified tuple
    # print(modified_tuple)
    # sentence = "POS-Tagger bestimmen den Worttyp anhand des Kontexts"
    # pos_and_words = get_pos_and_words(sentence)
    # print(pos_and_words)
    print(get_pos("geäußert"))
