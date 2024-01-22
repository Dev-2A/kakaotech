import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

class MyDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, seq_length):
        super().__init__()
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token = self.tokenizer.encode("<pad>").ids[0]
        self.unk_token = self.tokenizer.encode("<unk>").ids[0]
        self.bos_token = self.tokenizer.encode("<s>").ids[0]
        self.eos_token = self.tokenizer.encode("</s>").ids[0]
        self.mask_token = self.tokenizer.encode("<mask>").ids[0]
        self.input_ids = []
        buffer = []
        with open(text_path, "r", encoding="utf-8") as f:
            for text in f.readlines():
                buffer.extend(self.tokenizer.encode(text).ids)
                
                #NOTE - eos, bos 토큰을 붙이기 위해 seq_length-2 만큼 자른다.
                while len(buffer) >= (seq_length - 2):
                    input_id = (
                        [self.bos_token] + buffer[: seq_length - 2] + [self.eos_token]
                    )
                    self.input_ids.append(input_id)
                    buffer = buffer[seq_length - 2:]
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx]

#!SECTION collate_fn 구현
# 토큰 시퀀스를 마스킹 하고 레이블을 생성하기 위해 필요
# collate_fn - 메소드를 입력받고, 데이터 샘플을 미니 배치로 만들 때 다양한 길이의 시퀀스를 패딩할 때 주로 사용
# collate_fn을 Callable 1개 클래스로 구현 시 추가 입력 가능 -> 패딩 및 마스킹 포함하여 처리
# collate_fn 메소드 -> Dataset의 요소들의 리스트를 입력으로 받음
# MyDataset에서 토큰 시퀀스를 리턴 --> collate_fn은 토큰 시퀀스들의 리스트를 입력으로 받음
# 이러한 토큰 시퀀스를 학습하기 위해 텐서로 변환 후 길이 패딩, 패딩 마스크, 토큰 마스킹과 레이블을 생성해 최종 데이터세트 완성

#LINK https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L748
def get_mask_tokens(input_ids, tokenizer, mlm_probability=0.15, special_token_cnt=5):
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_token_mask = input_ids < special_token_cnt
    probability_matrix.masked_fill_(special_token_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.encode("<mask>").ids[0]
    
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    
    random_words = torch.randint(
        tokenizer.get_vocab_size(), labels.shape, dtype=torch.long
    )
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels

def get_padding_mask(input_id):
    return torch.zeros(input_id.shape).bool()

class MLMCollator:
    def __init__(self, tokenizer=None, special_token_cnt=5):
        self.tokenizer = tokenizer
        self.special_token_cnt = special_token_cnt
    
    def __call__(self, batch):
        input_ids = [torch.tensor(x) for x in batch]
        padding_mask = [get_padding_mask(x) for x in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_ids, labels = get_mask_tokens(
            input_ids, self.tokenizer, special_token_cnt=self.special_token_cnt
        )
        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=True)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "padding_mask": padding_mask
        }