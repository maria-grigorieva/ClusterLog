import re

from itertools import product

from torch import argmax
from transformers import BertTokenizerFast, BertForTokenClassification


model = BertForTokenClassification.from_pretrained("./BERT_model/checkpoint-2000")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
tokenizer.add_tokens("(.*?)")

all_labels = ['[PAD]', 'O'] + [f"{prefix}-{label}" for prefix, label in product('BILU', ["Meaningful", "Meaningless"])]
id_to_label = {i: label for i, label in enumerate(all_labels)}

spaces = re.compile(r'\s+')


def extract_relevant_part(sentence: str) -> str:
    token_position = 0
    beginnings, ends = [], []

    tokenized = tokenizer(sentence, return_tensors='pt')
    output = model(**tokenized)
    predictions = argmax(output.logits, dim=2)

    for token, prediction in zip(tokenized.tokens(), predictions[0].numpy()):
        if token in ['[CLS]', '[SEP]']:
            continue
        if token.startswith("##"):
            token = token[2:]

        token_position += sentence[token_position:].find(token)
        predicted_label = id_to_label[prediction]
        if predicted_label[2:] == "Meaningful":
            beginnings.append(token_position)
            ends.append(token_position + len(token))

    if not beginnings or not ends:
        return ""
    return spaces.sub(' ', sentence[beginnings[0]:ends[-1]])
