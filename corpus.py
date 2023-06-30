import pandas as pd
from evaluate import Evaluator

class Corpus(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.as_dataframe = ""
        self.predictions = ""

    def read(self):
        self.as_dataframe = pd.read_table(self.data_path, sep="\t")
        
    def evaluate(self):
        labels = self.predictions["label"].tolist()
        preds = self.predictions["predictions"].tolist()
        evaluator = Evaluator(labels, preds)
        evaluator.get_scores()
        return evaluator
    
    def write_file_with_preds(self, target_file):
        self.predictions.to_csv(target_file, sep="\t")
