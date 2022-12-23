import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from config import *


# if your computer has CUDA device, use the next line instead and disable the line next to the next line
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels = 2).to(device)


def predict(input):
    # predict
    text = input.replace(' ', '')
    inputs = tokenizer.batch_encode_plus([text], return_tensors='pt')
    inputs = {k: v.to(device) for k,v in inputs.items()}
    logits = model(**inputs).logits
    pred_id = np.argmax(logits.detach().cpu().numpy(), axis=-1)[0]
    return pred_id


if __name__ == '__main__':
    # unblue_1
    # text = '無論 何時何地 無論 這 世界 怎麼 轉所能 做 也 僅僅只是 活在 當下 做好 自己 能 做 不能 改變 事 想 再 多 也 徒勞'
    # blue_1
    # text = '羨慕 可以 活在 當下 人 如數家珍 地 細數 生活 我 每天 醒來 只 感到痛苦 為什麼'
    # blue_2
    text = "嗚 嗚嗚 嗚嗚 嗚嗚,從 綠島 回來 後,每天 起床 好 害怕,要 面對 抉擇"
    pred_id = predict(text)
    if pred_id == 0:
        print('unblue')
    else:
        print('blue')
