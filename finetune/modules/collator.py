import logging

from transformers import PreTrainedTokenizer

from .arguments import DataArguments, ModelArguments

logger = logging.getLogger(__name__)


class TrainDataCollator:

    def __init__(self, data_args: DataArguments, model_args: ModelArguments,
                 tokenizer: PreTrainedTokenizer):
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = 'right'
        input_texts = [d["input_ids"] for d in batch]

        encoding = self.tokenizer(input_texts,
                                  return_tensors="pt",
                                  max_length=self.model_args.max_input_len,
                                  truncation=True,
                                  padding=True,
                                  return_attention_mask=True)
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        response_texts = [d["labels"] for d in batch]
        target_encoding = self.tokenizer(
            response_texts,
            return_tensors="pt",
            max_length=self.model_args.max_input_len,
            truncation=True,
            padding=True,
            return_attention_mask=True)
        labels = target_encoding.input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)


class TestDataCollator:

    def __init__(self, data_args: DataArguments, model_args: ModelArguments,
                 tokenizer: PreTrainedTokenizer):
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "left"
        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            max_length=self.model_args.max_input_len,
            truncation=True,
            padding=True,
            return_attention_mask=True,
        )

        test_prefix_allowed_tokens = [
            x['test_prefix_allowed_tokens'] for x in batch
        ]
        return (inputs, targets, input_texts, test_prefix_allowed_tokens)
