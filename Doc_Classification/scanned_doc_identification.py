import numpy as np
import pandas as pd

import pytesseract
from PIL import Image#, ImageDraw, ImageFont

import torch
from datasets import Dataset, Features, Sequence, Value, Array2D #ClassLabel
from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification #, AdamW
import logging
from datetime import datetime
logger = logging.getLogger(__name__)

class FormRecognizer_LayoutLM:
    def __init__(self, confidence_threshold=0.9):
        self.confidence_threshold = confidence_threshold

        self.label2idx = {'1098': 0,
                        '1098c': 1,
                        '1098e': 2,
                        '1098t': 3,
                        '1099b': 4,
                        '1099div': 5,
                        '1099int': 6,
                        '1099misc': 7,
                        '1099oid': 8,
                        'benefitcharge': 9,
                        'generic_doc': 10,
                        'Invoices': 11,
                        'K1': 12,
                        'k3': 13,
                        'RL1': 14,
                        'RL2': 15,
                        't3': 16,
                        't4': 17,
                        't4a': 18,
                        't4e': 19,
                        'T4ps': 20,
                        't4rsp': 21,
                        't5': 22,
                        'T5008': 23,
                        'T5013': 24,
                        'w2': 25
                    }
                            
        self.testing_features = Features({
                                        'input_ids': Sequence(feature=Value(dtype='int64')),
                                        'bbox': Array2D(dtype="int64", shape=(512, 4)),
                                        'attention_mask': Sequence(Value(dtype='int64')),
                                        'token_type_ids': Sequence(Value(dtype='int64')),
                                        'image_path': Value(dtype='string'),
                                        'words': Sequence(feature=Value(dtype='string')),
                                    })
        
        #Doenload the Tokenizer & Model in Offline Mode
        # self.tokenizer = LayoutLMTokenizer.from_pretrained("./layoutlm")
        # self.tokenizer = LayoutLMTokenizer.from_pretrained("Layout_LM/layoutlm")
        self.tokenizer = LayoutLMTokenizer.from_pretrained("/classification-model/layoutlm/latest")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = LayoutLMForSequenceClassification.from_pretrained("saved_model_epochs_19/epoch_5")
        # self.model = LayoutLMForSequenceClassification.from_pretrained("Layout_LM/V1-All_samples")
        self.model = LayoutLMForSequenceClassification.from_pretrained("/classification-model/model/latest")
        self.model.to(self.device)

    def normalize_box(self, box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    def apply_ocr(self, example):
        image = Image.open(self.page_image_path)
        width, height = image.size

        ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)

        words = list(ocr_df.text)
        words = [str(w) for w in words]
        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row)
            actual_box = [x, y, x + w, y + h]
            actual_boxes.append(actual_box)

        boxes = []
        for box in actual_boxes:
            boxes.append(self.normalize_box(box, width, height))

        assert len(words) == len(boxes)
        example['words'] = words
        example['bbox'] = boxes
        return example

    def encode_testing_example(self, example, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):
        words = example['words']
        normalized_word_boxes = example['bbox']
        assert len(words) == len(normalized_word_boxes)

        token_boxes = []
        for word, box in zip(words, normalized_word_boxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        special_tokens_count = 2
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True)
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length
        encoding['bbox'] = token_boxes

        assert len(encoding['input_ids']) == max_seq_length
        assert len(encoding['attention_mask']) == max_seq_length
        assert len(encoding['token_type_ids']) == max_seq_length
        assert len(encoding['bbox']) == max_seq_length

        return encoding

    def recognize_form(self):
        query_df = pd.DataFrame({'image_path': [self.page_image_path]})
        query = Dataset.from_pandas(query_df)
        query = query.map(self.apply_ocr)
        query = query.map(lambda example: self.encode_testing_example(example), features=self.testing_features)
        query.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids'])
        query = torch.utils.data.DataLoader(query, batch_size=1, shuffle=True)
        batch = next(iter(query))

        outputs = self.model(
            input_ids=batch["input_ids"].to(self.device),
            bbox=batch["bbox"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device)
        )

        preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
        pred_labels = {label: pred for label, pred in zip(self.label2idx.keys(), preds)}
        
        page_class = "Unidentified"
        max_confidence = max(pred_labels.values())
        if max_confidence >= self.confidence_threshold:
            page_class = max(pred_labels, key=pred_labels.get)

        # result = {
        #     "page_types": [
        #         {
        #             "page_no": self.page_no,
        #             "page_type": self.page_type,
        #             "page_class": page_class,
        #             "conf_score": np.round(max_confidence,3)
        #         }
        #     ]
        # }
        result = {"page_class": page_class, "conf_score": np.round(max_confidence,3)}
        return result
    
