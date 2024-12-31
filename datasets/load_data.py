import os
import pandas as pd
import json

def load_test_data(datadir, dataset="sst2"):
	if dataset == "sst2":
		path = os.path.join(datadir,"SST2")
		test_df = pd.read_csv(os.path.join(path,"test.tsv"),sep='\t',header=0)
		test_df.columns = ['sentence','label']
		return test_df
	else:
		raise ValueError('Invalid dataset name passed!')

def load_train_data(datadir, dataset="sst2", synthetic=False):
	if dataset == "sst2":
		path = os.path.join(datadir,"SST2")
		train_df = pd.read_csv(os.path.join(path,"train.tsv"),sep='\t',header=0)
		train_df.columns = ['sentence','label']
		return train_df
	else:
		raise ValueError('Invalid dataset name passed!')