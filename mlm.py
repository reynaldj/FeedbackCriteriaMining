#code inspired from https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c

from transformers import BertForMaskedLM, BertTokenizer, AdamW
import torch
from utils import shuffle_sentences
import math
from tqdm import tqdm
import math

wwm_probability = 0.2

class MeditationTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def finetune_distilbert_on_mlm(data_folder):
    model_checkpoint = "distilbert-base-german-cased"
    model = BertForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    vocab_size = len(tokenizer.get_vocab())
    print("Vocabulary size:", vocab_size)
    data_sentences = shuffle_sentences(data_folder)

    tokenized_inputs = tokenizer(data_sentences, return_tensors="pt", max_length=512, truncation=True, padding='max_length')

    tokenized_inputs['labels'] = tokenized_inputs.input_ids.detach().clone()

    random_tensor = torch.rand(tokenized_inputs.input_ids.shape)
    mask_array = (random_tensor < 0.15) * (tokenized_inputs.input_ids != 102) * (tokenized_inputs.input_ids != 103) * (tokenized_inputs.input_ids != 0)

    masked_indices = []

    # Masking
    for i in range(mask_array.shape[0]):
        masked_indices.append(torch.flatten(mask_array[i].nonzero()).tolist())

    for i in range(mask_array.shape[0]):
        tokenized_inputs.input_ids[i, masked_indices[i]] = 104

    meditation_dataset = MeditationTextDataset(tokenized_inputs)
    data_loader = torch.utils.data.DataLoader(meditation_dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    epochs = 2

    for epoch in range(epochs):
        progress_bar = tqdm(data_loader, leave=True)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()  # Backward propagation
            optimizer.step()

            progress_bar.set_description(f'Epoch {epoch}')
            progress_bar.set_postfix(loss=loss.item())

    model.save_pretrained("distilBERT_Sprachbildung")

def evaluate_model_on_mlm(model_path, test_data_folder):
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("distilbert-base-german-cased")
    test_sentences = shuffle_sentences(test_data_folder)
    tokenized_test_inputs = tokenizer(test_sentences, return_tensors="pt", max_length=512, truncation=True, padding='max_length')

    random_tensor = torch.rand(tokenized_test_inputs.input_ids.shape)
    test_mask_array = (random_tensor < 0.15) * (tokenized_test_inputs.input_ids != 102) * (tokenized_test_inputs.input_ids != 103) * (tokenized_test_inputs.input_ids != 0)

    test_masked_indices = []

    for i in range(test_mask_array.shape[0]):
        test_masked_indices.append(torch.flatten(test_mask_array[i].nonzero()).tolist())

    for i in range(test_mask_array.shape[0]):
        tokenized_test_inputs.input_ids[i, test_masked_indices[i]] = 104
    
    test_dataset = MeditationTextDataset(tokenized_test_inputs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.eval()
    total_test_loss = 0.0
    num_test_batches = 0
    correct_predictions = 0
    total_predictions = 0
    total_tokens = 0  
    test_progress_bar = tqdm(test_loader, leave=True)
    for batch in test_progress_bar:
        with torch.no_grad():  # Disable gradient tracking for inference
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass to compute test loss
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()  # numel() gives the total number of elements
            total_tokens += input_ids.numel()

            # Accumulate the test loss
            total_test_loss += loss.item()
            num_test_batches += 1

    # Calculate the average test loss
    average_test_loss = total_test_loss / num_test_batches
    accuracy = correct_predictions / total_predictions

    perplexity = math.exp(average_test_loss)

    test_results = {
        'average_test_loss': average_test_loss,
        'total_test_loss': total_test_loss,
        'num_test_batches': num_test_batches,
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }
    results_file = 'test_results.json'

    # Save the results to a JSON file
    import json
    with open(results_file, 'w') as file:
        json.dump(test_results, file)

    # Print or use the average_test_loss as needed
    print(f'Average Test Loss: {average_test_loss}')
    print(f'Total Test Loss: {total_test_loss}')

if __name__ == "__main__":
    finetune_distilbert_on_mlm("somajo_splitted_Materials")
    # evaluate_model_on_mlm("distilBERT-Sprachbildung",data_folder)