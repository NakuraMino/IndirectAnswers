import pandas as pd 
import time 
from sklearn.model_selection import train_test_split

file_path = './data/circa-data.tsv'
train_data_path = './data/circa-data-train.tsv'
dev_data_path = './data/circa-data-dev.tsv'
test_data_path = './data/circa-data-test.tsv'

data = pd.read_csv(file_path, sep='\t')
time.sleep(2)
rows = data.shape[0]

train, test = train_test_split(data, test_size=0.4)
dev, test = train_test_split(test, test_size=0.5)

train.to_csv(train_data_path, sep="\t")
dev.to_csv(dev_data_path, sep="\t")
test.to_csv(test_data_path, sep="\t")