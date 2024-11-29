import torchFastText
import unittest
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split


class tftTest(unittest.TestCase):
      
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        test_data_path = 'test_data/dbpedia_sample_10000.csv'
        df_sample = pd.read_csv(test_data_path,header=None)
        self.texts = df_sample[2].tolist()
        labels = df_sample[0].tolist()
        length = len("__class__")
        self.labels_num = [int(x[length:]) for x in labels]
        num_buckets = 20
        embedding_dim = 100
        num_classes = len(set(self.labels_num))
        min_count=1
        min_n=3
        max_n=6
        len_word_ngrams=10
        sparse=False 
        self.torchfasttext = torchFastText.torchFastText(num_buckets=num_buckets,embedding_dim=embedding_dim,num_classes=num_classes,min_count=min_count,min_n=min_n,max_n=max_n,len_word_ngrams=len_word_ngrams,sparse=sparse)


    def test_init(self):
        num_buckets = 20
        embedding_dim = 100
        num_classes = len(set(self.labels_num))
        min_count=1
        min_n=3
        max_n=6
        len_word_ngrams=10
        sparse=False 
        self.torchfasttext = torchFastText.torchFastText(num_buckets=num_buckets,embedding_dim=embedding_dim,num_classes=num_classes,min_count=min_count,min_n=min_n,max_n=max_n,len_word_ngrams=len_word_ngrams,sparse=sparse)
        self.assertTrue(True)
        
    def test_train_no_categorical_variables(self):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.texts, self.labels_num, test_size=0.2, stratify=self.labels_num
        )
        self.torchfasttext.train(np.asarray(train_texts)[:,None], np.asarray(train_labels)[:,None], np.asarray(val_texts)[:,None],np.asarray(val_labels)[:,None], num_epochs=5, batch_size=64)
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()