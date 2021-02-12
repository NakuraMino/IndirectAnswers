from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import logging
import torch

# Import the dataset to use here

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

# Runs validation to find the best model
def validate(args, model, tokenizer, data, device, epoch, model_losses):
    logging.info("***** Running development *****")

    dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=args.train_batch_size,
    )

    dev_loss = 0.0
    nb_dev_step = 0

    model.eval()
    for batch in tqdm(dataloader, desc="Checking dev model accuracy..."):
        with torch.no_grad():
            inputs = batch[0].to(device)
            atten = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=inputs, attention_mask=atten, labels=labels)
            tmp_dev_loss, _ = outputs[:2]

            dev_loss += tmp_dev_loss.item()
            nb_dev_step += 1

    loss = dev_loss / nb_dev_step
    model_losses.append(loss)
    writer.add_scalar('dev_loss', loss, epoch)

    # Saving a trained model
    logging.info("** ** * Saving validated model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model

    path = str(epoch)
    full_path = os.path.join(args.output_dir, path)
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
    os.mkdir(full_path)
    output_model_file = os.path.join(args.output_dir, path, "pytorch_model.bin")
    output_config_file = os.path.join(args.output_dir, path, "config.json")

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, path))


def main():
	parser = ArgumentParser()
	parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs to train for")
	parser.add_argument('--model_type', type=str, required=True, help="choose a valid pretrained model")
    parser.add_argument('--batch_size', default=32, type=int, help="total batch size")
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="initial learning rate for Adam")
    parser.add_argument('--grad_clip', type=float, default=0.25, help="Grad clipping value")

	args = parser.parse_args()

	model = AutoTokenizer.from_pretrained(args.model_type)
	tokenizer = AutoModel.from_pretrained(args.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

	logging.info("****** Running training *****")
    logging.info(f"  Num examples = {len(database)}")
    logging.info(f"  Num epochs = {args.epochs}")
    logging.info(f"  Batch size = {args.batch_size}")
    logging.info(f"  Learning rate = {args.learning_rate}")


	train_iterator = trange(0, args.epochs, desc="Epoch")
    for epoch, _ in enumerate(train_iterator):
        epoch_dataset = dataset_train                                                                   
        train_dataloader = DataLoader(
          	epoch_dataset, 
            batch_size=args.train_batch_size, 
            shuffle=False,
        )
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")                                       
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(epoch_iterator):                                                       
        	model.train()

        	input_ids, atten, labels = batch
            input_ids = input_ids.to(device)                                                                
            atten = atten.to(device)                                                                        
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=atten, labels=labels)

            loss = outputs[0]                                                                                                                                                                               
            loss.backward()

            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            model.zero_grad()

            global_step += 1
            epoch_iterator.set_description(f"Loss: {total_loss / (global_step + 1)}")

        validate(args, model, tokenizer, dataset_dev, device, epoch, model_losses)


if __name__ == '__main__':
	main()