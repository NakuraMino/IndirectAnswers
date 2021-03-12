import pandas as pd 
import time 
from sklearn.model_selection import train_test_split


# setting file paths
file_path = './data/circa-data.tsv'
all_data_path = './data/unmatched-circa-data.tsv'
train_data_path = './data/unmatched-circa-data-train.tsv'
dev_data_path = './data/unmatched-circa-data-dev.tsv'
test_data_path = './data/unmatched-circa-data-test.tsv'

# loading data
data = pd.read_csv(file_path, sep='\t')
time.sleep(2) # hack to make data load

# pick the first scenario from shuffled data
data = data.sample(frac=1)  # shuffle data
scenario = data.iloc[0]['context']
print('Scenario to leave out:', scenario)

# filter out scenario and save as test data
scenario_data = data[data.context == scenario]
scenario_data.to_csv(test_data_path, sep='\t', index=None)

# split rest of data for train and dev
majority_data = data[data.context != scenario]
train, dev = train_test_split(majority_data, test_size=0.1)  # 90% training, 10% dev
train.to_csv(train_data_path, sep="\t", index=None)
dev.to_csv(dev_data_path, sep="\t", index=None)
