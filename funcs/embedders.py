from utils.utils import get_flatten
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from datasets import Dataset

class Embedder:
    def __init__(self, sent_emb_model_path, device, config):
        self.tokenizer_model = AutoTokenizer.from_pretrained(sent_emb_model_path)
        self.model = AutoModel.from_pretrained(sent_emb_model_path).to(device)
        self.device = device
        self.config=config

    def run_rep_cls_pool(self, all_outputs, dataloader):
        start_idx=0
        with torch.no_grad():
            for inputs in tqdm(dataloader):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                representations, _ = self.model(**inputs, return_dict=False)
                end_idx = start_idx + representations.shape[0]
                all_outputs[start_idx:end_idx] = representations[:,0,:]
                start_idx = end_idx
                
            return all_outputs
                
    def run_rep_mean_pool(self, all_outputs, dataloader):
        start_idx=0
        with torch.no_grad():
            for inputs in tqdm(dataloader):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                representations, _ = self.model(**inputs, return_dict=False)
                attention_mask = inputs["attention_mask"]
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(representations.size()).to(representations.dtype)
                )
                summed = torch.sum(representations * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                end_idx = start_idx + representations.shape[0]
                all_outputs[start_idx:end_idx] = (summed / sum_mask)
                start_idx = end_idx
                
            return all_outputs
    
    def clean_memory(self):
        del self.model, self.all_outputs
        torch.cuda.empty_cache()