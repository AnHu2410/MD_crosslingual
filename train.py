"""This file contains the neural trainer functions."""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers.adapters import AutoAdapterModel
from transformers.adapters.composition import Stack
from transformers import AdapterConfig, AdapterTrainer
from transformers import set_seed
import numpy as np


set_seed(42)


class TrainerMbert(object):
    def __init__(self, dataframe_train, dataframe_test, target_file):
        self.checkpoint = "bert-base-multilingual-cased"
        self.dataframe_train = dataframe_train
        self.dataframe_test = dataframe_test
        self.input_train = self.dataframe_train[["sentence", "masked_sen", "label"]]
        self.input_test = self.dataframe_test[["sentence", "masked_sen", "label"]]
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.target = target_file
        self.training_args = TrainingArguments(output_dir="results")

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text_a"], examples["text_b"], padding=True)

    def preprocess_data(self, df):
        # rename columns so they are recognizable for BERT:
        df.columns = ["text_a", "text_b", "labels"]

        # turn dataframe into huggingface dataset:
        dataset = Dataset.from_pandas(df)

        # tokenize dataset:
        tokenized_dataset = dataset.map(self.preprocess_function, batched=True)

        return tokenized_dataset

    def train(self):
        # preprocess training data:
        tokenized_dataset = self.preprocess_data(self.input_train)
        print(tokenized_dataset)

        # use data collator for faster training:
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # set up trainer:
        model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=2)
        trainer = Trainer(model=model, args=self.training_args, train_dataset=tokenized_dataset,
                          tokenizer=self.tokenizer,
                          data_collator=data_collator,
                          )
        # train:
        trainer.train()

        # save fine-tuned model, so it can be used for second fine-tuning or for inference:
        trainer.save_model(self.target)

    def predict(self):
        # preprocess evaluation data:
        tokenized_dataset = self.preprocess_data(self.input_test)

        # set up trainer with fine-tuned weights:
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(self.target, num_labels=2)
        trainer = Trainer(model=model, args=self.training_args, tokenizer=self.tokenizer,
                          data_collator=data_collator)

        # generate predictions from trainer:
        predictions = trainer.predict(tokenized_dataset)
        preds = np.argmax(predictions.predictions, axis=-1).tolist()  # logits to predictions
        self.dataframe_test["predictions"] = preds
        return self.dataframe_test

    def mad_x(self, language, path_task_adapter):
        # preprocess evaluation data:
        tokenized_dataset_target = self.preprocess_data(self.input_test)

        # initialize model:
        model = AutoAdapterModel.from_pretrained(self.checkpoint)

        # load pretrained language adapters for model:
        lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
        model.load_adapter("en/wiki@ukp", config=lang_adapter_config)
        if language == "ru":
            model.load_adapter("ru/wiki@ukp", config=lang_adapter_config)
        elif language == "ge":
            model.load_adapter("de/wiki@ukp", config=lang_adapter_config)
        elif language == "la":
            model.load_adapter("la/wiki@ukp", config=lang_adapter_config)

        # load pretrained task adapter for model:
        adapter_path = path_task_adapter
        adapter_name = model.load_adapter(adapter_path)

        # combine pretrained task adapter with pretrained language
        # adapter for language of unseen data:
        if language == "ru":
            model.active_adapters = Stack("ru", adapter_name)
        elif language == "ge":
            model.active_adapters = Stack("de", adapter_name)
        elif language == "la":
            model.active_adapters = Stack("la", adapter_name)

        # perform zero-shot evaluation:
        args = TrainingArguments(learning_rate=1e-4, num_train_epochs=10,
                                 per_device_train_batch_size=32,
                                 per_device_eval_batch_size=32,
                                 logging_steps=100,
                                 output_dir="./training_output",
                                 overwrite_output_dir=True,
                                 # The next line is important to ensure
                                 # the dataset labels are properly passed to the model
                                 # remove_unused_columns = False
                                 )
        ad_trainer = AdapterTrainer(model=model, args=args, tokenizer=self.tokenizer)
        predictions = ad_trainer.predict(tokenized_dataset_target)
        preds = np.argmax(predictions.predictions, axis=-1, keepdims=True)
        print(preds)
        self.dataframe_test["predictions"] = preds
        return self.dataframe_test
