from utils.utils import get_flatten
from datasets import Dataset
from transformers import AutoTokenizer
import torch
from math import floor
from torch.nn.utils.rnn import pad_sequence
from random import randint

class Tokenized_loader:
    def __init__(self, sent_emb_model_path, config):
        self.tokenizer_model = AutoTokenizer.from_pretrained(sent_emb_model_path)
        self.config=config
    
    
    def set_for_rep(self, preprocessed_data, is_question):
        split_contents_flatten, indices = get_flatten(preprocessed_data['split_contents'])
        
        if self.config['erase_dup']:
            split_contents_flatten = list(set(split_contents_flatten))
        flatten_data = Dataset.from_dict({'text':split_contents_flatten})
        
        def tokenize_row(row):
            return self.tokenizer_model(row['text'], truncation=True, max_length=self.config['max_length'], padding='max_length')
        
        tokenized_data = flatten_data.map(tokenize_row, batch_size=50)
        
        tokenized_data = tokenized_data.remove_columns(['text'])
        tokenized_data = tokenized_data.with_format('torch')
        
        all_outputs = zeros((len(tokenized_data), self.config['hidden_size']))
        return all_outputs, contents_len, tokenized_data, indices, split_contents_flatten 
    
    
    def set_for_rep_addtoken(self, preprocessed_data):
        max_length = self.config['max_length']
        split_contents_flatten, indices = get_flatten(preprocessed_data['split_contents'])
        
        if self.config['erase_dup']:
            split_contents_flatten = list(set(split_contents_flatten))
        flatten_data = Dataset.from_dict({'text':split_contents_flatten})
        
        def tokenize_row(row):
            return self.tokenizer_model(row['text'], truncation=True, max_length=max_length)
        tokenized_data = flatten_data.map(tokenize_row, batch_size=1)
        
        all_outputs = torch.zeros((len(tokenized_data), self.config['hidden_size']))
        return all_outputs, tokenized_data, indices, split_contents_flatten 

if __name__ == "__main__":
    from datasets import load_from_disk
    test_a = load_from_disk('funcs/test_a')
    test_q = load_from_disk('funcs/test_q')
    
    from preprocessors import Preprocessor
    config = {'erase_dup':True,'max_length':64,'hidden_size':1024}
    preprocessor = Preprocessor(debug=True, config=config)

    q_preprocessed= preprocessor.preprocess(test_q)
    a_preprocessed = preprocessor.preprocess(test_a)

    tokenized_loader = Tokenized_loader(sent_emb_model_path='sorryhyun/sentence-embedding-klue-large', config=config)
    tokenized_loader.set_for_rep_addtoken(q_preprocessed, is_question=True)