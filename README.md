# IndirectAnswers

This repository contains the reproduction of the paper, ["I'd rather just go to bed": Understanding Indirect Answers](https://arxiv.org/pdf/2010.03450.pdf). 

## Files Overview
`dataloader.py`: contains the dataloader classes for each of the datasets (Circa, BOOLQ, MNLI, DIS). Also contains utility functions to load the dataloader. 

`datasplit.py`: script to split the Circa dataset into train, dev, and test in the matched setting.

`dataunmatch.py`: script to split the Circa dataset into train, dev, and test data in the unmatched setting.

`finetuning.py`: the main script to run training and validation on a different model for each of the datasets. 

`testing.py`: evaluates our model by running our models on test data. 

`data/`: folder that contains all the datasets (Circa, BOOLQ, MNLI, DIS).

`bash_script/`: contains the scripts to quickly run the models.

## Data download instructions
All datasets are publicly available and can be accessed online. The Circa and BoolQ Dataset are both available on Google Research Datasets’ Github Repository. MNLI is available on the author’s website, while the DIS dataset is also available on the author’s github. Store each downloaded dataset in the data/ folder of the repository if it is not already present. 
https://github.com/google-research-datasets/circa

https://github.com/google-research-datasets/boolean-questions

https://cims.nyu.edu/~sbowman/multinli/

https://github.com/windweller/DisExtract

## Preprocessing Code
To preprocess, code, run these commands:
```
python datasplit.py
python dataunmatch.py
```

## Training
Run the bash scripts that begin with "finetune". The models will save to a directory called "models".

To run our bash scripts in the bash script folder, run the command below. Note that the script and code is still in progress. 

```
./bash_scripts/name_of_bash_script.sh
```

## Testing
Run the bash scripts that begin with "test".
