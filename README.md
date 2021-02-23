# IndirectAnswers

This repository contains the reproduction of the paper, ["I'd rather just go to bed": Understanding Indirect Answers](https://arxiv.org/pdf/2010.03450.pdf). 

`dataloader.py`: contains the dataloader classes for each of the datasets (Circa, BOOLQ, MNLI, DIS). Also contains utility functions to load the dataloader. 

`dataplit.py`: script to split the data in train, dev, and test data. 

`finetuning.py`: the main script to run training and validation on a different model for each of the datasets. 

`testing.py`: evaluates our model by running our models on test data. 

`data/`: folder that contains all the datasets (Circa, BOOLQ, MNLI, DIS).

`bash_script/`: contains the scripts to quickly run the models. 

To run our bash scripts in the bash script folder, run the command below. Note that the script and code is still in progress. 

```
./bash_scripts/name_of_bash_script.sh
```
