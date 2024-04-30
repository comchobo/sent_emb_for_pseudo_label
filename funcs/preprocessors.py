import json
import re
from datasets import Dataset
from kiwipiepy import Kiwi
from tqdm import tqdm
from datasets import load_from_disk
from unicodedata import normalize

def preprocess_minimal(row):
    extracted_data = [normalize('NFC', x) for x in row['contents']]
    extracted_data = [x if capture_weirdo.search(x) is not None else 'skip_this'
                      for x in extracted_data]
    return {'preprocessed_contents': extracted_data}


class Preprocessor:
    def __init__(self, config, debug=False):
        self.debug = debug
        if self.debug:
            self.kiwi = Kiwi(num_workers=1, model_type='sbg')
            self.num_proc = 1
        else :
            self.kiwi = Kiwi(num_workers=0, model_type='sbg')
            self.num_proc = 8
        self.config=config

    def split_sentence(self, paragraph):
        splitted_contents = self.kiwi.split_into_sents(paragraph)
        contents = [contents for contents in tqdm(splitted_contents)]
        split_contents_list = []
        for content in contents:
            temp = [x for x in content if not len(x) < 2] # embargo
            split_contents_list.append(temp)
        return split_contents_list

    def base_preprocess(self, data_to_preprocess): # embargo
        return data_to_preprocess

    def preprocess(self, data_to_preprocess): # embargo
        
        extracted_data = self.base_preprocess(data_to_preprocess)
        
        temp_contents = []
        if self.config['preprocess_for_split']:
            for content in extracted_data['preprocessed_contents']:
                temp_contents.append(content)
        else:
            temp_contents = extracted_data['preprocessed_contents']
                
        split_contents_list = self.split_sentence(temp_contents)
        extracted_data = extracted_data.add_column(name="split_contents", column=split_contents_list)
        return extracted_data
    
if __name__ == "__main__":
    pre = Preprocessor()
    from datasets import load_from_disk
    test_a = load_from_disk('funcs/test_a')
    test_q = load_from_disk('funcs/test_q')
    