import pandas as pd
import numpy as np
import random
import torch
import os
import numpy as np
from tqdm import tqdm

from nltk.corpus import stopwords
from model import *
from helper import clean_text, convert_image

from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss

MAX_SEQ_LENGTH = 56

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, input_mask, segment_ids, label_id):
    # def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        # self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

# def collate_tokenize(data):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     text_batch = [element["text"] for element in data]
#     tokenized = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt')
#     return tokenized

def get_appropriate_dataset(data):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    features = convert_to_features(data, MAX_SEQ_LENGTH, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    # all_acoustic = torch.tensor(
    #     [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        # all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset

def data_loader():
    data_train_ = pd.read_excel('data_train.xlsx')
    data_train_.drop('Unnamed: 0', axis=1, inplace=True)
    data_dev_ = pd.read_excel('data.xlsx')
    data_dev_.drop('Unnamed: 0', axis=1, inplace=True)

    data_train = []
    data_dev = []

    for idx, item in data_train_.iterrows():
        data_train.append(((clean_text(item['text']), convert_image(item['image_context'], len(clean_text(item['text']))), clean_text(item['answers']), clean_text(item['question'])), idx))
        if idx == 100:
            break

    for idx, item in data_dev_.iterrows():
        data_dev.append(((clean_text(item['text']), convert_image(item['image_context'], len(clean_text(item['text']))), clean_text(item['answers']), clean_text(item['question'])), idx))    
        if idx == 100:
            break

    num_train_optimization_steps = (
        int(
            len(data_train) / 24 /
            1
        )
        * 20
    )

    train_data = get_appropriate_dataset(data_train)
    dev_data = get_appropriate_dataset(data_dev)

    train_dataloader = DataLoader(
        train_data, batch_size=24, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_data, batch_size=64, shuffle=True
    )

    return (
        train_dataloader,
        dev_dataloader,
        num_train_optimization_steps,
    )

def convert_to_features(examples, max_seq_length, tokenizer):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    features = []

    for (idx_ex, example) in enumerate(examples):
        (words_ex, visual_ex, ans_ex), label_id = example

        tokens_words = []
        # inversions = []
        for idx, word in enumerate(words_ex):
            tokenized_word = tokenizer.tokenize(word)
            tokens_words.extend(tokenized_word)
            # inversions.extend([idx] * len(tokenized_word))

        tokens_ans = []
        for idx, ans in enumerate(ans_ex):
            tokenized_ans = tokenizer.tokenize(ans)
            tokens_ans.extend(tokenized_ans)
            # inversions.extend([idx] * len(tokenized_ans))

        # print(f'len words {len(tokens_words)} len ans {len(tokens_ans)}')

        if len(tokens_words) + len(tokens_ans) + 4 > MAX_SEQ_LENGTH:
            tokens_words = tokens_words[:(MAX_SEQ_LENGTH - len(tokens_ans)) - 4]
        # print('after')
        # print(f'len words {len(tokens_words)} len ans {len(tokens_ans)}')

        # assert len(tokens_words) == len(inversions)

        ###
        # aligned_visual = []
        # for inv_idx in inversions:
        #     try:
        #         aligned_visual.append(visual_ex[inv_idx, :])
        #     except:
        #         print('out of bound')
        ###

        # visual = np.array(aligned_visual)
        visual = np.array(visual_ex)
        
        input_ids, visual, input_mask, segment_ids = prepare_bert_input(tokens_words, tokens_ans, visual, tokenizer)
        # input_ids, visual, input_mask, segment_ids = prepare_bert_input(tokens_words, visual, tokenizer)

        # Check input length
        assert len(input_ids) == MAX_SEQ_LENGTH
        assert len(input_mask) == MAX_SEQ_LENGTH
        assert len(segment_ids) == MAX_SEQ_LENGTH
        # assert acoustic.shape[0] == args.max_seq_length
        # print(f'visual shape {visual.shape} visual[0] shape {visual.shape[0]} max {MAX_SEQ_LENGTH}')
        assert visual.shape[0] == MAX_SEQ_LENGTH

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                label_id=label_id
            )
        )
    return features

# def prepare_bert_input(tokens, visual, acoustic, tokenizer):
def prepare_bert_input(tokens_words, tokens_ans, visual, tokenizer):
# def prepare_bert_input(tokens_words, visual, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    MASK = tokenizer.mask_token
    tokens = [CLS] + tokens_words + [SEP] + tokens_ans + [SEP] + [MASK]
    # tokens = [CLS] + tokens_words + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    # acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    # acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    # print(f'visual before {visual.shape}')
    visual = np.concatenate((visual_zero, visual, visual_zero))
    # print(f'visual_zero dimension {visual_zero.shape} and visual dimension {visual.shape}')

    input_ids = tokenizer.convert_tokens_to_ids(tokens); ###
    segment_ids1 = [0] * (len(tokens_words) + 2); ###
    segment_ids2 = [1] * (len(tokens_ans) + 2)
    segment_ids = segment_ids1 + segment_ids2
    input_mask = [1] * len(input_ids); ###

    pad_length = (MAX_SEQ_LENGTH - len(input_ids)); ###
    # print(f'pad length {pad_length} len(input_ids) {len(input_ids)}')
    # pad_length = args.max_seq_length - len(input_ids)

    # acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    # acoustic = np.concatenate((acoustic, acoustic_padding))
    pad_length_visual = MAX_SEQ_LENGTH - visual.shape[0]

    visual_padding = np.zeros((pad_length_visual, VISUAL_DIM))
    # print(f'visual before {visual.shape}')
    visual = np.concatenate((visual, visual_padding))
    # print(f'visual padding {visual_padding.shape} visual_dim {VISUAL_DIM} visual {visual.shape}')

    # loss_ids = input_ids.copy()[:-1]
    # loss_tensor = torch.tensor([loss_ids]).to(DEVICE)
    # keys_text = tokenizer.decode(input_ids)
    

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, input_mask, segment_ids
    # return input_ids, visual, acoustic, input_mask, segment_ids

def make_labels():
    print()

def prep_for_training(num_train_steps):
    multimodal_config = MultimodalConfig(
        beta_shift=1.0, dropout_prob=0.5
    )

    model = MAG_BertForQuestionGeneration.from_pretrained("bert-base-uncased", multimodal_config)
    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * num_train_steps,
        num_training_steps=num_train_steps,
    )
    return model, optimizer, scheduler



def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        # input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        input_ids, visual, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        # acoustic = torch.squeeze(acoustic, 1)
        # print(type(segment_ids))
        # print(segment_ids.shape)
        # print(type(label_ids))
        # print(label_ids.shape)
        # print(label_ids)
        outputs = model(
            input_ids,
            visual,
            # acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=input_ids
        )

        loss = outputs.loss
        # print(outputs)

        # logits = outputs[0]
        # print(outputs[0])
        # print(len(outputs[0]))
        # print(outputs[0].shape)
        # print(type(logits))
        # print(logits.shape)
        # print(logits.view(-1))
        # print(logits.view(-1).shape)
        # print(type(label_ids))
        # print(label_ids.view(-1))
        # print(label_ids.view(-1).shape)
        # loss_fct = MSELoss()
        # test = torch.reshape(logits, (1, 24))
        # loss = loss_fct(test.view(-1), label_ids.view(-1))

        if 1 > 1:
            loss = loss / 1

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % 1 == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps

def train(
    model,
    train_dataloader,
    validation_dataloader,
    # test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_accuracies = []

    for epoch_i in range(int(3)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        # test_acc, test_mae, test_corr, test_f_score = test_score_model(
        #     model, test_data_loader
        # )

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc:{}".format(
                epoch_i, train_loss, valid_loss, "test_acc"
            )
        )

        valid_losses.append(valid_loss)
        # test_accuracies.append(test_acc)

def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            # acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                # acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=input_ids,
            )
            # print(outputs.loss)
            # logits = outputs[0]
            loss = outputs.loss
            # loss_fct = MSELoss()
            # loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if 1 > 1:
                loss = loss / 1

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps

def predict(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    preds = []
    mask_token = tokenizer.mask_token

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, input_mask, segment_ids, label_ids = batch
            # print(input_ids)
            # temp = tokenizer.decode(input_ids[0])
            # context_tokenized = tokenizer.encode(temp)
            context_tokenized = tokenizer.encode(tokenizer.decode(input_ids[0]))
            pred_list = []

            for _ in range(MAX_SEQ_LENGTH):
                pred_str_ids = tokenizer.convert_tokens_to_ids(pred_list + [mask_token])
                # print(type(context_tokenized))
                # print(type(pred_str_ids))
                predict_token = context_tokenized + pred_str_ids
                # print(predict_token)
                # print(len(predict_token) >= MAX_SEQ_LENGTH)
                # print(len(predict_token))
                if len(predict_token) >= MAX_SEQ_LENGTH:
                    break
                predict_token = torch.tensor(predict_token)
                predictions = model(predict_token, visual)
                # print(predictions)
                # print("test")
                predict_idx = torch.argmax(predictions[0][0][-1]).item()
                predicted_token = tokenizer.convert_ids_to_tokens([predict_idx])
                # print(predicted_token)
                # print(type(predicted_token))
                # print(predicted_token.shape)
                if "[SEP]" in predicted_token:
                    break
                pred_list.append(predicted_token)
                # print(predicted_token[0])
            
            token_ids = tokenizer.convert_tokens_to_ids(pred_list)
            preds.append(tokenizer.decode(token_ids))
            # print(tokenizer.decode(token_ids))
        
        with open("./test.txt", "w", encoding="UTF-8") as f:
            for name in preds:
                f.write(name + "\n")
        print(preds)



def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    print('done')
    (
        train_data_loader,
        dev_data_loader,
        num_train_optimization_steps
    ) = data_loader()

    model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        optimizer,
        scheduler
    )

    predict(model, train_data_loader)

if __name__ == "__main__":
    main()