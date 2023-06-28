import pandas as pd

class Corpus(object):
    def __init__(self, data_path, language, experiment):
        self.data_path = data_path
        self.language = language
        self.as_dataframe = ""
        self.predictions = ""
        self.augmented_dataset = False

    def read_corpus(self):
        self.as_dataframe = pd.read_table(file, sep="\t")
        
    def make_predictions(self):
        pass
    
    def write_preds_2_file(self):
        pass
    
    
       
