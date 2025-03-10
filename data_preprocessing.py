from nltk.metrics import distance
import openpyxl
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
import spacy
from spacy.tokens import doc
from transformers import AutoTokenizer, AutoModelForTokenClassification
# import crosslingual_coreference
import nltk
from somajo import SoMaJo
from HanTa import HanoverTagger as ht
from PyMultiDictionary import MultiDictionary
from utils import get_sentence_from_conll_tuples, open_conll


tokenizer = SoMaJo("de_CMC", split_camel_case=True)
tagger = ht.HanoverTagger('morphmodel_ger.pgz')
dictionary = MultiDictionary()
# tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
unrelevant_sentences = ["Liebe Gruppe, vielen Dank für Ihre Abgabe!","Ich erstelle immer gern zu Beginn eine Kurzanalyse, die Sie als minimale Musterlösung ansehen können ;)","_________________________________________","Nun zu Ihrer Analyse: Meine Anmerkungen finden Sie als ✔️oder grünen Text 😉", "Ich hoffe, das hilft Ihnen!","Bei Fragen gern melden lächelnd","Liebe Gruppe, ","Danke! Schönes Material ","Nicht Bestanden","Bewertung: richtig gut erledigt","Danke!","Kein Problem!","In Ihrer Einreichung sind dann meine Anmerkungen mit ✔️versehen und ggf. mit grünem Text kommentiert. ", '', "Abgabe von :","180758 , remdekir@hu-berlin.de , Gruppe 3 ,"]
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("de_core_news_lg")
nlp.add_pipe("xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": -1,"model_name": "info_xlm"}) #xx_coref is the coreference model
def check_text_similar(text1, text2):
    """
    Check if two sentencese are same, used to remove duplicate sentences in text
    """
    similarity = 1 - distance.jaccard_distance(set(text1.split()), set(text2.split()))

    if similarity == 1:
        return True
    else:
        return False

def separate_students_feedbacks_texts(excelfile,x):
    """
    Extract student and teacher feedback text from an excel file
    """
    workbook = openpyxl.load_workbook(excelfile)
    sheet = workbook.active

    studentTexts_folder = f'Aufgabe{x}/studenttexts'
    if not os.path.exists(studentTexts_folder):
        os.makedirs(studentTexts_folder)
    feedbackTexts_folder = f'Aufgabe{x}/feedbacktexts'
    if not os.path.exists(feedbackTexts_folder):
        os.makedirs(feedbackTexts_folder)

    for index, cell in enumerate(sheet['A'], start=1):
        text = cell.value
        if text:
            file_path = f"{studentTexts_folder}/studenttexts{index}.txt"
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)

    for index, cell in enumerate(sheet['B'], start=1):
        text = cell.value
        if text:
            file_path = f"{feedbackTexts_folder}/feedbacktexts{index}.txt"
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)

def filter_out_same_lines_with_checks(path_aufgabe):
    """
    Filter out sentences from student answer that is restated in the teacher feedback
    In teacher feedback, some student answer sentences are restated with the addition of ✔️\n at the end
    """
    count_texts = count_texts_in_folder(path_aufgabe + "/feedbacktexts")
    output_folder_path = path_aufgabe + "/filteredfeedbacktexts"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for x in range(1,count_texts+1):
        feedback_path = f"/filteredfeedbacktexts/feedbacktexts{x}.txt"
        full_feedback_path = path_aufgabe + feedback_path
        studenttext_path = f"/studenttexts/studenttexts{x}.txt"
        full_studenttext_path = path_aufgabe + studenttext_path

        with open(full_feedback_path, 'r', encoding="utf-8") as file:
            feedback_lines = file.readlines()
            feedback_lines = [line.rstrip() for line in feedback_lines]
            feedback_lines = [line + '\n' for line in feedback_lines]
        with open(full_studenttext_path, 'r', encoding="utf-8") as file:
            studenttext_lines_raw = file.readlines()
            studenttext_lines_raw= [line.rstrip() for line in studenttext_lines_raw]
            studenttext_lines_raw= [line + '\n' for line in studenttext_lines_raw]

        filtered_lines_raw = [line for line in feedback_lines if line not in studenttext_lines_raw]
        # print(filtered_lines_raw)
        filtered_lines = [line for line in filtered_lines_raw if not line.endswith("✔️\n")]

        output_file_name = f"/feedbacktexts{x}.txt"
        full_output_path = output_folder_path + output_file_name
        with open(full_output_path,'w',encoding='utf-8') as file:
            file.writelines(filtered_lines)




def count_texts_in_folder(folder_path):
    """
    Count how many files in folder
    """
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count

def remove_duplicate_lines(text_path):
    """
    Remove lines duplicated in a text
    """
    with open(text_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Remove duplicate lines
    lines = list(set(lines))
    os.remove(text_path)

    # Write the updated lines to a new file
    with open(text_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

def remove_unrelevant_lines(folder_path, output_folder):
    """
    Remove lines that are irrelevant.
    Irrelevant lines are lines that have 85% similarity to those stated in unrelevant_sentences defined above
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only text files
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            with open(input_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            threshold = 85

            filtered_lines = []
            for line in lines:
                sentences = line_splitter_somajo(line)
                filtered_sentences = []
                for sentence in sentences:
                    similarity = process.extractOne(str(sentence), unrelevant_sentences,scorer=fuzz.token_set_ratio)

                    if similarity[1] < threshold:
                        filtered_sentences.append(sentence)
                filtered_sentences = [sentence if isinstance(sentence, str) else sentence.text for sentence in filtered_sentences]
                combined_line = ' '.join(filtered_sentences)
                filtered_lines.append(combined_line)

            with open(output_path, "w", encoding="utf-8") as file:
                for line in lines:
                    file.write(str(line))

def remove_unrelevant_lines_from_sentence(sentence):
    """
    Remove sentences that comes from a splitted line that are irrelevant.
    Irrelevant lines are lines that have 85% similarity to those stated in unrelevant_sentences defined above
    """
    threshold = 85
    sentences = line_splitter_somajo(sentence)
    filtered_sentences = []
    for sentence in sentences:
        similarity = process.extractOne(str(sentence), unrelevant_sentences,scorer=fuzz.token_set_ratio)

        if similarity[1] < threshold:
            filtered_sentences.append(sentence)
    filtered_sentences = [sentence if isinstance(sentence, str) else sentence.text for sentence in filtered_sentences]
    combined_line = ' '.join(filtered_sentences)
    return combined_line

def extract_teacher_model_answer(aufgabe_path):
    """
    Extract teacher model answer (example answer from teacher) in teacher feedback text
    """
    feedback_text_folder = aufgabe_path + "/feedbacktexts"
    output_folder_path = aufgabe_path+"/teacherModelAnswer"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    count = count_texts_in_folder(feedback_text_folder)
    pattern = r"________+"
    for i in range(1,count+1):
        feedback_text = feedback_text_folder +f"/feedbacktexts{i}.txt"
        output_text = output_folder_path + f"/modelAnswer{i}.txt"
        with open(feedback_text,'r',encoding="utf-8") as file:
            lines = file.readlines()

        line_index = None
        for j, line in enumerate(lines):
            if re.search(pattern, line):
                line_index = j
                break

        if line_index is not None:
            model_answer = lines[:line_index]

            with open(output_text,'w',encoding='utf-8') as file:
                file.writelines(model_answer)
        else:
            print("Pattern not found")

def filter_out_same_lines(text_path1, text_path2): 
    """
    filter out lines in text_path2 that are same with lines in text_path1
    """
    with open(text_path1, 'r', encoding="utf-8") as file:
            text1_lines = file.readlines()
            text1_lines = [line.rstrip() for line in text1_lines]
            text1_lines = [line + '\n' for line in text1_lines]
    with open(text_path2, 'r', encoding="utf-8") as file:
            text2_lines = file.readlines()
            text2_lines = [line.rstrip() for line in text2_lines]
            text2_lines = [line + '\n' for line in text2_lines]
    filtered_lines = [line for line in text2_lines if line not in text1_lines]
    with open(text_path2,'w',encoding='utf-8') as file:
            file.writelines(filtered_lines)

def resolve_pronouns_in_text(text):
    """
    perform coreference resolution to resolve pronouns in text
    """
    if text.strip() and len(text)>10:    
        doc = nlp(text)
        print(doc._.coref_clusters)
        i=0
        for item in doc._.coref_clusters:
            print(f"Cluster {i}")
            for span in item:
                start, end = span
                print(doc[start:end+1])
            i = i+1
        return doc._.resolved_text

def resolve_pronouns(folder_path,output_folder):
    """
    perform coreference resolution to resolve pronouns in for all files in folder_path
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only text files
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            with open(input_path,"r",encoding='utf-8') as file:
                feedbacktext_lines = file.readlines()
            resolved_lines = []
            for line in feedbacktext_lines:
                if line != "\n":
                    newline = resolve_pronouns_in_text(line)
                    resolved_lines.append(newline)
            resolved_lines = [sentence if isinstance(sentence, str) else "" if sentence is None else sentence.text for sentence in resolved_lines]
            with open(output_path,"w",encoding='utf-8') as file:
                for line in resolved_lines:
                    file.write(str(line))

def copy_feedback_only(studentext_path, feedback_path,file_index,x):
    """
    Copy only teacher feedback from feedback text
    In some feedback text, student answers are stated
    """
    with open(studentext_path, 'r', encoding='utf-8') as student_file:
        student_texts = student_file.readlines()

    # Read the teacher feedback file
    with open(feedback_path, 'r', encoding='utf-8') as feedback_file:
        feedback_texts = feedback_file.readlines()

    sentences = []
    for i in range(len(feedback_texts)):
        start_position=-1
        student_text_index = -1
        for j in range(len(student_texts)):
            # Find the starting position of the feedback in the student texts
            if(student_texts[j]!= '\n' and feedback_texts[i].find(student_texts[j])!=-1):
                start_position=feedback_texts[i].find(student_texts[j])
                student_text_index=j
                break
        if start_position!=-1:    
            # Extract the feedback
            feedback = feedback_texts[i][start_position + len(student_texts[student_text_index]):].strip()
            sentences.append(feedback)

    # Write the feedback to a new file
    with open(f"Aufgabe{x}/onlyfeedbacktexts/onlyfeedback{file_index}.txt", 'w', encoding='utf-8') as feedback_output:
        for sentence in sentences:
                feedback_output.write(str(sentence) + '\n')
    
def sentence_splitter_somajo(data_path):
    """
    Split text from a file to sentence using somajo
    """
    sentences = tokenizer.tokenize_text_file(data_path, paragraph_separator="single_newlines")
    extracted_sentences = []
    for sentence in sentences:
        sentence_list = []
        for token in sentence:
            sentence_list.append(token.text)
        extracted_sentence = ' '.join(sentence_list)
        extracted_sentences.append(extracted_sentence)
    return extracted_sentences

def line_splitter_somajo(text):
    """
    Split a line to a sentence
    """
    paragraph = [text]
    sentences = tokenizer.tokenize_text(paragraph)
    extracted_sentences = []
    for sentence in sentences:
        sentence_list = []
        for token in sentence:
            sentence_list.append(token.text)
        extracted_sentence = ' '.join(sentence_list)
        extracted_sentences.append(extracted_sentence)
    return extracted_sentences

def filter_outside_training_sentences(sentences_tuples): 
    """
    filter out sentences that only contain outside tags
    """
    filtered_training_datas = [[]]
    counter=0
    for sentence_tuples in sentences_tuples:
        sentence = get_sentence_from_conll_tuples(sentence_tuples)
        splitted_sentences = tokenizer.tokenize_text([sentence])
        sum_len = 0
        b_index_sentences = []
        extracted_sentences=[]
        actual_sentences=[]
        if counter==62:
            print()
        for splitted_sentence in splitted_sentences:
            b_index_sentences.append(sum_len)
            sentence_list = []
            stack_count = 0
            for token in splitted_sentence:
                if stack_count <= 0:
                    if(token.text == sentence_tuples[sum_len][0]):
                        sum_len += 1
                    else:
                        untokenized_word = sentence_tuples[sum_len][0]
                        tokenized_words = tokenizer.tokenize_text([str(untokenized_word)])
                        for sentence in tokenized_words:
                            if(len(sentence)>1): #case tokenizer further tokenized than conll data
                                stack_count = len(sentence)-1
                            else:
                                word = sentence_tuples[sum_len][0]
                                while word != token.text: #case conll data further tokenized than tokenizer
                                    sum_len += 1
                                    word += sentence_tuples[sum_len][0]
                        sum_len += 1
                else:
                    stack_count -= 1
                sentence_list.append(token.text)
            extracted_sentence = ' '.join(sentence_list)
            list_of_tuples = sentence_tuples[sum_len:sum_len+len(sentence_list)]
            first_elements = []

            # Iterate through the list of tuples and extract the first element from each tuple
            for t in list_of_tuples:
                first_elements.append(t[0])
            actual_sentence =' '.join(first_elements)    
            extracted_sentences.append(extracted_sentence)
            actual_sentences.append(actual_sentence)
            # if tokens[-1] != sentence_tuples[sum_len+(len(tokens)-1)]:
            b_index_sentences.append(sum_len)
        assert sum_len == len(sentence_tuples) 
        b_index_sentences.append(sum_len)   
        for i in range(len(b_index_sentences)):
            if i == len(b_index_sentences)-1:
                break
            found_non_outside = False
            for j in range(b_index_sentences[i+1]-b_index_sentences[i]):
                if sentence_tuples[b_index_sentences[i]+j][2] == 'B-Feedback_Criteria':
                    found_non_outside = True
                    break
            if found_non_outside == True:
                filtered_training_datas.append(sentence_tuples[b_index_sentences[i]:b_index_sentences[i] + (b_index_sentences[i+1]-b_index_sentences[i])])   
        counter+=1        
    return filtered_training_datas

def split_conll_data(input_path,output_path):
    """
    Split conll data so that one sentence has maximum 64 tokens
    """
    splitted_training_datas = []
    train_data = open_conll(input_path)
    for sentence_tuples in train_data:
        sentence = get_sentence_from_conll_tuples(sentence_tuples)
        splitted_sentences = tokenizer.tokenize_text([sentence]) 
        sum_len = 0
        b_index_sentences = []
        extracted_sentences=[]
        actual_sentences=[]
        for splitted_sentence in splitted_sentences:
            b_index_sentences.append(sum_len)
            sentence_list = []
            stack_count = 0
            for token in splitted_sentence:
                if stack_count <= 0:
                    if(token.text == sentence_tuples[sum_len][0]):
                        sum_len += 1
                    else:
                        untokenized_word = sentence_tuples[sum_len][0]
                        tokenized_words = tokenizer.tokenize_text([str(untokenized_word)])
                        for sentence in tokenized_words:
                            if(len(sentence)>1): #case tokenizer further tokenized than conll data
                                stack_count = len(sentence)-1
                            else:
                                word = sentence_tuples[sum_len][0]
                                while word != token.text: #case conll data further tokenized than tokenizer
                                    sum_len += 1
                                    word += sentence_tuples[sum_len][0]
                        sum_len += 1
                else:
                    stack_count -= 1
                sentence_list.append(token.text)
            extracted_sentence = ' '.join(sentence_list)
            list_of_tuples = sentence_tuples[sum_len:sum_len+len(sentence_list)]
            first_elements = []
             # Iterate through the list of tuples and extract the first element from each tuple
            for t in list_of_tuples:
                first_elements.append(t[0])
            actual_sentence =' '.join(first_elements)    
            extracted_sentences.append(extracted_sentence)
            actual_sentences.append(actual_sentence)
            # if tokens[-1] != sentence_tuples[sum_len+(len(tokens)-1)]:
            # b_index_sentences.append(sum_len)
        assert sum_len == len(sentence_tuples)
        b_index_sentences.append(sum_len)   
        for i in range(len(b_index_sentences)):
            if i == len(b_index_sentences)-1:
                break
            splitted_sentence_tuples = sentence_tuples[b_index_sentences[i]:b_index_sentences[i] + (b_index_sentences[i+1]-b_index_sentences[i])]
            
            if len(splitted_sentence_tuples) > 64:
                group_size = 64

                # Initialize an empty list to store the groups
                groups = []

                # Calculate the number of groups required
                num_groups = len(splitted_sentence_tuples) // group_size + (len(splitted_sentence_tuples) % group_size != 0)

                # Iterate through the list and split into groups
                for i in range(num_groups):
                    start_idx = i * group_size
                    end_idx = (i + 1) * group_size
                    group = splitted_sentence_tuples[start_idx:end_idx]
                    splitted_training_datas.append(group)
            else:
                splitted_training_datas.append(splitted_sentence_tuples) #add splitted sentence
    with open(output_path,"w",encoding="utf-8") as output_file:
        for i in range(len(splitted_training_datas)):
            for j in range(len(splitted_training_datas[i])):
                output_file.write(splitted_training_datas[i][j][0] + " " + splitted_training_datas[i][j][1] + " " + splitted_training_datas[i][j][2])
                output_file.write("\n")
            if i == (len(splitted_training_datas)-1) and j == len(splitted_training_datas[i])-1:
                break
            else:
                output_file.write("\n")


def extract_non_outside_training_sentences_to_file(input_folder_path, output_folder_path):
    """
    Extract only sentences that has minimum 1 non-outside token
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.conll'):
            input_path = os.path.join(input_folder_path, filename)
            data = open_conll(input_path)
            separated_training_data = filter_outside_training_sentences(data)              
            separated_training_data = [sentence for sentence in separated_training_data if sentence]
            output_file = os.path.join(output_folder_path, filename)
            with open(output_file,'w', encoding='utf-8') as output_file:
                for sentence in separated_training_data:
                    for tag in sentence:
                        string = str(tag[0]) + ' ' + str(tag[1]) + " _ " + str(tag[2])
                        output_file.write(str(string) + '\n')
                    output_file.write('\n')
            output_file.close()    


def separate_feedbacks_into_sentences_somajo(x): #x is Aufgabe number
    """
    Separate feedback text into sentence using somajo
    """
    feedback_folder = f"Aufgabe{x}/filteredfeedbacktexts"
    onlyFeedback_folder = f"Aufgabe{x}/onlyfeedbacktexts"
    count_files = count_texts_in_folder(feedback_folder)
    output_folder = f"Aufgabe{x}/somajosentencestexts"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(1,count_files+1):
        if x == 1 or (x == 2 and (i == 1 or i == 17)):
            sentences = sentence_splitter_somajo(onlyFeedback_folder+f"/onlyfeedback{i}.txt")
        else:
            sentences = sentence_splitter_somajo(feedback_folder+f"/feedbacktexts{i}.txt")
        with open(output_folder+f"/somajosentencetext{i}.txt", 'w', encoding='utf-8') as output_file:
            for sentence in sentences:
                output_file.write(str(sentence) + '\n')

def lemmatize_sentence(text):
    """
    lemmatize all words in a text, for german compound verb, the lemmatized form will be concatenation of prefix and the base word
    e.g anlaufen, anstehen
    """
    words = nltk.word_tokenize(text)
    lemmata = tagger.tag_sent(words)
    separable_prefix = None
    beginning_sent = 0
    word_list = []
    for index, (word,lemma,pos) in enumerate(lemmata):
        if pos == 'PTKVZ': # PTKVZ is a preposition, this is used to catch german compound words
            separable_prefix = lemma
            for i in range(beginning_sent, index):
                in_word, in_pos = word_list[i]
                if in_pos == 'VV(FIN)':
                    combine_word = separable_prefix + in_word
                    if dictionary.meaning('de',combine_word)[1] != '':
                        word_list[i] = (combine_word,in_pos)
            word_list.append((lemma,pos))
            beginning_sent = index+1
        elif pos == '$.':
            word_list.append((lemma,pos))
            beginning_sent = index+1
        else:
            word_list.append((lemma,pos))

    words = [t[0] for t in word_list]
    sentence = " ".join(words)
    return sentence


def lemmatize_line(line):
    """
    Lemmatize a line
    """
    sentences = line_splitter_somajo(line)
    lemmatized_sentences = []
    for sentence in sentences:
        lemmatized_sentences.append(lemmatize_sentence(sentence))
    lemmatized_line = " ".join(lemmatized_sentences)
    return lemmatized_line

def lemmatize_file(file_path):
    """
    Lemmatize the entire text in a file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lemmatized_lines = []
    for line in lines:
        lemmatized_lines.append(lemmatize_line(line))
    return lemmatized_lines

def lemmatize_file_with_output(file_path, output_path):
    """
    Lemmatize the file and save the lemmatized text in an output file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lemmatized_lines = []
    for line in lines:
        lemmatized_lines.append(lemmatize_line(line))
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lemmatized_lines:
            f.write(str(line) + '\n')

def lemmatize_folder(folder_path,output_folder):
    """
    Lemmatize all texts of all files found in a folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only text files
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            lines = lemmatize_file(input_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(str(line) + '\n')

if __name__ == "__main__":
#     text = (
#     "Do not forget about Momofuku Ando! He created instant noodles in Osaka. At"
#     " that location, Nissin was founded. Many students survived by eating these"
#     " noodles, but they don't even know him."
# )
#     resolved_text = resolve_pronouns_in_text(text)
#     print(resolved_text)
    # sentence = "Ich mag es, wenn Sätze lemmatisiert werden"
    filter_out_same_lines_with_checks("Aufgabe1")