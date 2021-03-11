from argparse import ArgumentParser
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score

import dataloader


def evaluate(model, dataloader, device):
    actual = []
    pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard1'], batch['token_type_ids']

            input_ids = input_ids.to(device)
            atten = atten.to(device)
            labels = labels.to(device).squeeze()
            token_type_id = token_type_id.to(device)

            outputs = model(input_ids=input_ids, attention_mask=atten, token_type_ids=token_type_id, labels=labels)
            pred.extend([val.item() for val in outputs])
            actual.extend([val.item() for val in labels])
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average='macro')

    return accuracy, f1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--test_data', type=Path, required=True)
    parser.add_argument('--batch_size', default=32, type=int, help="total batch size")
    # parser.add_argument('--model_type', type=str, required=True, help="choose a valid pretrained model")
    # parser.add_argument('--num_labels', type=int, required=True, help="choose the number of labels for the experiment")
    args = parser.parse_args()
    
    # config = BertConfig.from_pretrained(args.model_type, num_labels=args.num_labels)
    # model = BertForSequenceClassification.from_pretrained(args.model_type, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    test_dataloader = dataloader.getCircaDataloader(args.test_data, batch_size=args.batch_size, num_workers=4, use_tokenizer=False, tokenizer=tokenizer)
    model = BertForSequenceClassification()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracy, f1 = evaluate(model, dataloader, device)
    print(f"Accuracy: {accuracy}")
    print(f"F1 score: {f1}")