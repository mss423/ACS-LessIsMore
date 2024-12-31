from load_data import *
import pandas as pd

datadir = "/Users/maxspringer/Documents/GitHub/ACS-LessIsMore/datasets"
savedir = "/Users/maxspringer/Documents/GitHub/ACS-LessIsMore/results"
dataset = 'sst2'

# Load data_df, train and test
train_df = load_train_data(datadir, dataset)
test_df  = load_test_data(datadir, dataset)

# Run subsampling on train
Ks = [len(train_df)//10]
train_subsample_idx = acs_sample(train_df, Ks)
results_df = pd.DataFrame(columns=['K','acc','f1'])

# Train models
for K in Ks:
	k_samples = train_subsample_idx[K]
	train, dev = []
	test_result = train_bert()
	acc, f1 = Metric(test_df.label, test_result.prediction)
	pd.concat([results_df, pd.DataFrame({'K': K, 'acc': [acc], 'f1': [f1]})], ignore_index=True)

# Save results
results_df.to_csv(os.path.join(savedir, f"{dataset}_results.csv"), index=False)