import pandas as pd 
import time 
from sklearn.model_selection import train_test_split

# setting file paths
file_path = './data/circa-data.tsv'
train_data_path = './data/circa-data-train.tsv'
dev_data_path = './data/circa-data-dev.tsv'
test_data_path = './data/circa-data-test.tsv'

# loading data
data = pd.read_csv(file_path, sep='\t')
time.sleep(2) # hack to make data load 

# splitting data
train, test = train_test_split(data, test_size=0.4)
dev, test = train_test_split(test, test_size=0.5)

# saving data
train.to_csv(train_data_path, sep="\t")
dev.to_csv(dev_data_path, sep="\t")
test.to_csv(test_data_path, sep="\t")