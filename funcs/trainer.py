from datasets import load_from_disk, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate, json
import numpy as np
import os, copy, gc, torch, shutil, json


class BERT_Trainer:
    def __init__(self, pretrained_model_path, output_path, config, \
                 debug=True, num_labels=-1, num_proc=1):
        self.pretrained_model_path = pretrained_model_path
        self.output_path = output_path
        self.model = AutoModelForSequenceClassification.from_pretrained(\
            self.pretrained_model_path , num_labels=num_labels+1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path)
        self.debug = debug
        self.config = config
        self.num_proc = num_proc
        self.num_labels= num_labels

    def train(self, train_dataset, dev_dataset, mode, batch_size, lr=1e-4):
        def do_tokenize(row):
            return self.tokenizer(row['text'], max_length=self.config['max_length'], truncation=True)
        
        tokenized_train_dataset = train_dataset.map(do_tokenize, batch_size=100, batched=True, num_proc=self.num_proc)
        tokenized_dev_dataset = dev_dataset.map(do_tokenize, batch_size=100, batched=True, num_proc=self.num_proc)
        collator = DataCollatorWithPadding(self.tokenizer)
        
        f1_score = evaluate.load('f1')
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return f1_score.compute(predictions=predictions, references=labels, average='micro')
        
        training_args = TrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=5,
            per_device_train_batch_size=48,
            per_device_eval_batch_size=96,
            warmup_steps=100,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            save_total_limit=1,
            learning_rate=lr,
            metric_for_best_model='eval_f1')

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_dev_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        # Train the model
        self.trainer.train()
        self.trainer.save_model(f'{self.output_path}_saved')


    def train_with_optimization(self, train_dataset, dev_dataset, mode, batch_size, ):
        def do_tokenize(row):
            return self.tokenizer(row['text'], max_length=self.config['max_length'], truncation=True)
        
        tokenized_train_dataset = train_dataset.map(do_tokenize, batch_size=100, batched=True, num_proc=self.num_proc)
        tokenized_dev_dataset = dev_dataset.map(do_tokenize, batch_size=100, batched=True, num_proc=self.num_proc)
        collator = DataCollatorWithPadding(self.tokenizer)
        
        f1_score = evaluate.load('f1')
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return f1_score.compute(predictions=predictions, references=labels, average='micro')
        
        model_save_path = f'{self.output_path}/trials'

        try:
            os.mkdir(f'{model_save_path}')
        except FileExistsError:
            pass

        metric_accumul = []
        lr_list = {0:1e-5, 1:2e-5, 2:5e-5, 3:1e-4}
        os.environ['WANDB_DISABLED'] = 'true'
        
        for trial_key in lr_list.keys():
            training_args = TrainingArguments(
                output_dir=f'{model_save_path}/HP_{trial_key}',
                evaluation_strategy="epoch",
                save_strategy='epoch',
                num_train_epochs=5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=int(batch_size*2),
                warmup_steps=10,
                weight_decay=0.01,
                load_best_model_at_end=True,
                save_total_limit=1,
                metric_for_best_model='eval_f1',
                learning_rate=lr_list[trial_key],
                tf32=True if 'lambda' in mode else False,
            )
        
            trainer = Trainer(
                model=copy.deepcopy(self.model),
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_dev_dataset,
                tokenizer=self.tokenizer,
                data_collator=collator,
                compute_metrics=compute_metrics
            )

            trainer.train()
            trainer.save_model(f'{model_save_path}/HP_{trial_key}_saved')
            metric_accumul.append(trainer.state.best_metric)
            del trainer

        save_target = metric_accumul.index(max(metric_accumul))
        print(f'{model_save_path}/HP_{save_target}_saved',' will be saved')
        try:
            shutil.copytree(f'{model_save_path}/HP_{save_target}_saved', f'{self.output_path}/optimized')
        except FileExistsError:
            shutil.rmtree( f'{self.output_path}/optimized', ignore_errors=True)
            shutil.copytree(f'{model_save_path}/HP_{save_target}_saved', f'{self.output_path}/optimized')
        shutil.rmtree(model_save_path, ignore_errors=True)
        
        gc.collect()
        torch.cuda.empty_cache()

    def test(self, test_dataset):
        def do_tokenize(row):
            return self.tokenizer(row['text'],  max_length=self.config['max_length'], truncation=True, padding='max_length')
        tokenized_test_data = test_dataset.map(
            do_tokenize, batch_size=100, batched=True, num_proc=self.num_proc)
        
        model = AutoModelForSequenceClassification.from_pretrained(f'{self.output_path}/optimized')
        
        f1_score = evaluate.load('f1')
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return f1_score.compute(predictions=predictions, references=labels, average='micro')
        
        trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        
        return trainer.predict(tokenized_test_data)

