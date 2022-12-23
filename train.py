import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from loader import DatasetLoader
from config import *
from tqdm import tqdm
import logging


stream = logging.StreamHandler()
recorder = logging.FileHandler('output/log.txt')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO, handlers = [stream, recorder])
logger = logging.getLogger()


# define the tokenizer, model and device
# if your computer has CUDA device, use the next line instead and disable the line next to the next line
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels = labels_counts).to(device)


# the function used to collate text with its label
def text_label_linker(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer.batch_encode_plus(texts, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels


def main():
    # introduce random factors to make sure the result can be reprodecued
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # data
    dataset = DatasetLoader(train_set)
    total_counts = len(dataset)
    # split the dataset into trainset and validset
    train_counts = int(total_counts * proportion)
    valid_counts = total_counts - train_counts
    trainset, validset = random_split(dataset, [train_counts, valid_counts], torch.Generator().manual_seed(seed))
    train_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn = text_label_linker)
    valid_data_loader = DataLoader(validset, batch_size=batch_size, collate_fn = text_label_linker)

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        # different learning rate，bert's: bert_lr，layer of classifier's: cls_lr
        # introduce weight_decay
        {
            "params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": decay_factor,
            "lr": bert_lr
        },
        {
            "params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' not in n],
            "weight_decay": decay_factor,
            "lr": cls_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' not in n],
            "weight_decay": 0.0,
            "lr": cls_lr
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = bert_lr)

    # train
    best_score = -1.0
    for epoch in range(epochs):
        train_loss = []
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_data_loader, desc='[--train--]', ncols=100, nrows = 100)
        for inputs, labels in pbar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # forward
            output = model(**inputs, labels=labels)
            loss = output.loss
            # backward
            loss.backward()

            train_loss.append(loss.item())
            # display epoch and loss
            pbar.set_postfix({'epoch': epoch + 1, 'loss': '{:.4f}'.format(loss.item()),
                              'epoch loss': '{:.4f}'.format(np.mean(train_loss))})

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

        # validation
        model.eval()
        with torch.no_grad():
            true_ids = []
            pred_ids = []
            pbar_valid = tqdm(valid_data_loader, desc='[--validation--]', ncols = 100, nrows = 100)
            for inputs, labels in pbar_valid:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                output = model(**inputs, labels = labels)
                loss = output.loss
                logits = output.logits

                logits_np = logits.detach().cpu().numpy()
                batch_id = np.argmax(logits_np, axis=-1)
                pred_ids.extend(batch_id.tolist())
                true_ids.extend(labels.detach().cpu().numpy().tolist())

                pbar_valid.set_postfix({'epoch': epoch + 1, 'loss': '{:.4f}'.format(loss.item())})
        # compute accuracy
        acc = accuracy_score(true_ids, pred_ids)
        # generate confusion matrix
        cm_matrix = confusion_matrix(true_ids, pred_ids)
        # logging
        logger.info('--' * 10)
        logger.info('epoch: {} '.format(epoch + 1) + 'acc: {:.4f}'.format(acc))
        logger.info('\nconfusion matrix: \n' + str(cm_matrix))

        # save the best model
        if best_score < acc:
            best_score = acc
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)


if __name__ == '__main__':
    main()