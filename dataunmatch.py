import pandas as pd 
import time 
from sklearn.model_selection import train_test_split


# setting file paths
file_path = './data/CIRCA/circa-data.tsv'
unmatched_path = './data/CIRCA/unmatched/scenario'

train_data_name = '/circa-data-train.tsv'
dev_data_name = '/circa-data-dev.tsv'
test_data_name = '/circa-data-test.tsv'

# loading data
data = pd.read_csv(file_path, sep='\t')
time.sleep(2) # hack to make data load

# save each train/dev/test datasets in a separate folders for each scenario 
scenarios = data.context.unique()
print(scenarios)

# uncomment below to recreate datasets
'''
for i in range(len(scenarios)):
    s = scenarios[i]
    print(s)

    # filter out scenario and save as test data
    scenario_data = data[data.context == s]
    scenario_data.to_csv(unmatched_path + str(i) + test_data_name, sep='\t', index=None)

    # split rest of data for train and dev
    majority_data = data[data.context != s]
    train, dev = train_test_split(majority_data, test_size=0.1)  # 90% training, 10% dev
    train.to_csv(unmatched_path + str(i) + train_data_name, sep="\t", index=None)
    dev.to_csv(unmatched_path + str(i) + dev_data_name, sep="\t", index=None)
'''
