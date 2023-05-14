import os

from torch.nn import CrossEntropyLoss
from PIL import Image, ImageDraw, ImageFont
import pytesseract as pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import LayoutLMTokenizer
from layoutlm.data.funsd import FunsdDataset, InputFeatures
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import LayoutLMForTokenClassification
import torch
from transformers import AdamW
from tqdm import tqdm
import pandas as pd
from torch import nn




def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


labels = get_labels("content/data/labels.txt")
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index

args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': 'content/data',
        'model_name_or_path': 'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',}

# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

args = AttrDict(args)

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

# the LayoutLM authors already defined a specific FunsdDataset, so we are going to use this here
train_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="train")
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=2)

eval_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="test")
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset,
                             sampler=eval_sampler,
                            batch_size=2)

batch = next(iter(train_dataloader))
input_ids = batch[0][0]
tokenizer.decode(input_ids)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 5
t_total = len(train_dataloader) * num_train_epochs # total number of training steps

#put the model in training mode
model.train()
for epoch in range(num_train_epochs):
  for batch in tqdm(train_dataloader, desc="Training"):
      input_ids = batch[0].to(device)
      bbox = batch[4].to(device)
      attention_mask = batch[1].to(device)
      token_type_ids = batch[2].to(device)
      labels = batch[3].to(device)
      # forward pass
      outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                      labels=labels)
      loss = outputs.loss

      # print loss every 100 steps
      if global_step % 100 == 0:
        print(f"Loss after {global_step} steps: {loss.item()}")

      # backward pass to get the gradients
      loss.backward()

      #print("Gradients on classification head:")
      #print(model.classifier.weight.grad[6,:].sum())

      # update
      optimizer.step()
      optimizer.zero_grad()
      global_step += 1

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None

# put model in evaluation mode
model.eval()
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    with torch.no_grad():
        input_ids = batch[0].to(device)
        bbox = batch[4].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels)
        # get the loss and logits
        tmp_eval_loss = outputs.loss
        logits = outputs.logits

        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # compute the predictions
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0
            )

# compute average evaluation loss
eval_loss = eval_loss / nb_eval_steps
preds = np.argmax(preds, axis=2)

out_label_list = [[] for _ in range(out_label_ids.shape[0])]
preds_list = [[] for _ in range(out_label_ids.shape[0])]

for i in range(out_label_ids.shape[0]):
    for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
            out_label_list[i].append(label_map[out_label_ids[i][j]])
            preds_list[i].append(label_map[preds[i][j]])

results = {
    "loss": eval_loss,
    "precision": precision_score(out_label_list, preds_list),
    "recall": recall_score(out_label_list, preds_list),
    "f1": f1_score(out_label_list, preds_list),
}

model.save_pretrained("model")

model = LayoutLMForTokenClassification.from_pretrained("model")

#image = Image.open('/content/form_example.jpg')
image = Image.open("8db51e0c267ca6d3ad2c24258eb571f0.jpg")
image = image.convert("RGB")
image

width, height = image.size
w_scale = 1000 / 2481
h_scale = 1000 / 3507

ocr_dfs = pd.read_csv('initialJsons/testing/out.csv')

ocr_list = []
if ocr_dfs.shape[0]>125:
    window = 50
    min_words = 0
    max_words = 125
    while max_words<ocr_dfs.shape[0]:
        ocr_list.append(ocr_dfs.iloc[min_words:max_words])
        min_words = min_words+window
        max_words = max_words+window
    if max_words-window!=ocr_dfs.shape[0]:
        ocr_list.append(ocr_dfs.iloc[min_words:ocr_dfs.shape[0]])
else:
    ocr_list.append(ocr_dfs.iloc[0:ocr_dfs.shape[0]])


for ocr_df in ocr_list:
    words = list(ocr_df.text)
    coordinates = ocr_df[['left', 'top', 'width', 'height', 'text']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
      x, y, w, h, t = tuple(row) # the row comes in (left, top, width, height) format
      actual_box = [x, y, x+w, y+h, t] # we turn it into (left, top, left+widght, top+height) to get the actual box
      actual_boxes.append(actual_box)

    def normalize_box(box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    boxes = []
    for box in actual_boxes:
      boxes.append(normalize_box(box, width, height))


    def convert_example_to_features(image, words, boxes, actual_boxes, tokenizer, args, cls_token_box=[0, 0, 0, 0],
                                    sep_token_box=[1000, 1000, 1000, 1000],
                                    pad_token_box=[0, 0, 0, 0]):
            width, height = image.size

            tokens = []
            token_boxes = []
            actual_bboxes = []  # we use an extra b because actual_boxes is already used
            token_actual_boxes = []
            for word, box, actual_bbox in zip(words, boxes, actual_boxes):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                token_boxes.extend([box] * len(word_tokens))
                actual_bboxes.extend([actual_bbox] * len(word_tokens))
                token_actual_boxes.extend([actual_bbox] * len(word_tokens))

            # Truncation: account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > args.max_seq_length - special_tokens_count:
                    tokens = tokens[: (args.max_seq_length - special_tokens_count)]
                    token_boxes = token_boxes[: (args.max_seq_length - special_tokens_count)]
                    actual_bboxes = actual_bboxes[: (args.max_seq_length - special_tokens_count)]
                    token_actual_boxes = token_actual_boxes[: (args.max_seq_length - special_tokens_count)]
            # add [SEP] token, with corresponding token boxes and actual boxes
            tokens += [tokenizer.sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]
            token_actual_boxes += [[0, 0, width, height]]

            segment_ids = [0] * len(tokens)

            # next: [CLS] token
            tokens = [tokenizer.cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            actual_bboxes = [[0, 0, width, height]] + actual_bboxes
            token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
            segment_ids = [1] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [tokenizer.pad_token_id] * padding_length
            token_boxes += [pad_token_box] * padding_length
            token_actual_boxes += [pad_token_box] * padding_length

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length
            # assert len(label_ids) == args.max_seq_length
            assert len(token_boxes) == args.max_seq_length
            assert len(token_actual_boxes) == args.max_seq_length

            return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes

    input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes = convert_example_to_features(image=image, words=words, boxes=boxes, actual_boxes=actual_boxes, tokenizer=tokenizer, args=args)

    tokenizer.decode(input_ids)

    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    attention_mask = torch.tensor(input_mask, device=device).unsqueeze(0)
    token_type_ids = torch.tensor(segment_ids, device=device).unsqueeze(0)
    bbox = torch.tensor(token_boxes, device=device).unsqueeze(0)
    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
    torch.set_printoptions(threshold=100_000)
    outputs.logits.shape
    outputs.logits.argmax(-1)

    token_predictions = outputs.logits.argmax(-1).squeeze().tolist() # the predictions are at the token level

    word_level_predictions = [] # let's turn them into word level predictions
    final_boxes = []
    for id, token_pred, box in zip(input_ids.squeeze().tolist(), token_predictions, token_actual_boxes):
      if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id,
                                                               tokenizer.sep_token_id,
                                                              tokenizer.pad_token_id]):
        # skip prediction + bounding box

        continue
      else:
        word_level_predictions.append(token_pred)
        final_boxes.append(box)

    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    def iob_to_label(label):
      if label != 'O':
        return label[2:]
      else:
        return "OTHER"

    label2color = {'other':'blue', 'priv_companyid':'green', 'priv_document':'orange', 'priv_doc_finid':'violet', 'priv_vatnum': 'red'}
    current_box = [0,0,0,0]
    for prediction, box in zip(word_level_predictions, final_boxes):
        if box == current_box:
            continue
        current_box = box
        predicted_label = iob_to_label(label_map[prediction]).lower()
        draw.rectangle(box[:4], outline=label2color[predicted_label])
        print(f"{box[4]}---{predicted_label}")
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)


    image

    from PIL import Image, ImageDraw, ImageFont
    image.show()
