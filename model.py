#STUB MLM, CLM = 모델을 사전학습하는 메커니즘. 논문 Attention Is All You Need에 나온 트랜스포머 모델을 학습.
#STUB 트랜스포머 모델은 크게 인코더와 디코더로 이루어져 있으며, 해당 모델은 인코더를 학습

#STUB 트랜스포머 인코더의 구성 요소
#STUB 1) Embedding Layer
#STUB 2) Transformer Encoder - Multi Head Self-Attention Layer / Feed-Forward Layer

import torch
from torch import nn
import torch.nn.functional as F

#!SECTION Transformer Encoder = MultiHeadAttention + FeedForward
# Pytorch에서 구현한 torch.nn.TransformerEncoderLayer : MultiHeadAttention과 FeedForward를 연결한 구현체
# Pytorch에서 구현한 torch.nn.TransformerEncoder : 위 레이어의 수를 입력으로 받아 인코더를 구성하는 구현체

#ANCHOR Config Class
class Config:
    def __init__(
        self,
        vocab_size = 10000, # 어휘 크기
        hidden_size=512, # 임베딩과 hidden states의 차원
        num_hidden_layers=4, # 인코더 레이어 수
        num_attention_heads=4,
        intermediate_size=2048, # 피드 포워드의 중간 차원. 보통 hidden_size * 4로 설정
        max_position_embeddings=128, # 시퀀스의 최대 길이
        layer_norm_eps=1e-12, # LayerNorm의 epsilon값
        hidden_dropout_prob=0.1, # 드롭아웃 확률
        initializer_range=0.02, # 모델 가중치의 초기화 값 구간
        is_causal=False, # CLM 학습인지 체크 True라면 Attention Mask 추가
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.is_causal = is_causal

#ANCHOR Encoder Class
# Embedding, torch.nn.TransformerEncoderLayer, torch.nn.TransformerEncoder를 선언하고 초기화
# Padding Mask : 시퀀스의 길이가 서로 다를 때, 길이를 맞추기 위해 패딩 토큰의 위치를 알려주는 마스킹 (src_key_padding_mask)
# Attention Mask : 미래(혹은 다음) 단어를 참조하지 못하도록 마스킹 (mask)
class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embeddings = Embedding(config)
        layers = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=F.gelu,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=layers, num_layers=config.num_hidden_layers
        )
        
    def forward(
        self,
        input_ids,
        attn_mask = None,
        padding_mask = None,
    ):
        if self.config.is_casual and attn_mask is None:
            size = input_ids.shape[1]
            device = input_ids.device
            attn_mask = torch.triu(
                torch.ones(size, size) * float("-inf"), diagonal=1
            ).to(device)
            
        x = self.embeddings(input_ids)
        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        return x
    
#!SECTION Embedding Layer
#ANCHOR 토큰 시퀀스와 토큰 시퀀스의 인덱스를 각각 nn.Embedding()을 거쳐 더하는 방법으로 임베딩 레이어 구성 --> 위치정보 포함
class Embedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        position_ids = (
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0).to(input_ids.device)
        )
        position_embeddings = self.position_embeddings(position_ids)
        word_embeddings = self.word_embeddings(input_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

#!SECTION Masked Language Model (MLM)
# 시퀀스의 일부 토큰을 마스킹하고, 해당 마스킹된 위치에서 원래 토큰을 맞추는 작업을 수행
# 대표 모델 : BERT --> 텍스트 분류에서 강점

#STUB MLM 훈련 과정
# 1) 훈련 텍스트 시퀀스에서 일부 토큰을 임의로 선택하여 마스킹 (보통 토큰의 15% 마스킹)
# 2) 마스킹 예정의 80% 토큰 - <mask>라는 특수 토큰이 들어감 --> 모델이 해당 위치에 있는 단어를 추론할 때 사용
# 10%는 랜덤 토큰, 10%는 토큰 없이 그대로
# 3) 마스킹된 문장을 모델에 입력한 뒤 마스킹된 위치에 있는 단어를 추론하도록 함
# 4) 추론한 단어와 정답 단어 간의 오차를 계산하여 모델을 학습
# MLM 학습을 위해 위의 인코더 모델에 학습용 헤드 부착 (= BERT 구현 참고) --> 인코더의 hidden states를 받아 vocab_size 길이를
# 가진 벡터를 결과 --> 해당 결과와 마스킹된 위치의 원래 단어와 오차를 계산해 학습
# 레이어의 가중치를 초기화하는 메소드를 구현해 모델 초기화 가능

#LINK https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L704
class MLMHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        
    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class MaskedLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder: Encoder = Encoder(config)
        self.mlm_head = MLMHead(config)
        self.apply(self.__init__weights)
    
    #LINK https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L748
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, padding_mask=None):
        x = self.encoder(input_ids, padding_mask=padding_mask)
        x = self.mlm_head(x)
        return x