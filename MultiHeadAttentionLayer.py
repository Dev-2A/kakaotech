#STUB Multi Head Attention Layer
# 여러 개의 어텐션 헤드가 존재하고, 각 헤드는 주어진 시퀀스에서 특정한 정보를 추출.
# h개의 어텐션 헤드가 있다면 토큰 임베딩의 각 토큰은 h개의 서로 다른 정보로 분석됨 = 이미지 처리에서 합성곱 신경망 필터와 유사
# 하나의 필터가 원을, 다른 필터가 사각형을 처리하는 경우와 비슷함
# 하나의 어텐션 헤드는 가중치 행렬을 계산 후 토큰 임베딩의 각 토큰마다 가중 평균을 구함 --> 가중치 행렬 = 어텐션 점수
# 토큰 임베딩의 각 토큰 사이에 유사도 점수가 계산됨 --> 연관성이 큰 토큰의 점수는 크고 / 연관성이 작다면 점수가 작음
# 강조할 부분의 점수를 크게 함으로써 어텐션 메커니즘 적용, 같은 토큰 임베딩끼리 계산해서 Self Attention이라고 이름

#STUB 어텐션 헤드의 구현
# 토큰 임베딩을 가지고 완전 연결 층(Fully connected layer)을 통해 쿼리 벡터(Q), 키 벡터(K), 밸류 벡터(V)를 만들고 계산
# 쿼리 벡터와 키 벡터의 점 곱(Dot product 또는 MatMul)로 가중치 행렬 만들기 -> 행렬 정규화 -> 소프트맥스 함수 적용 -> 열의 합이 1이 되게함
# 그 후 가중치 행렬과 밸류 벡터를 곱해 최종 결과 만듦

#STUB 다음 토큰을 예측하는 CLM 학습을 위한 추가작업
# 현재 토큰에서 다음 토큰들을 참고하지 못하도록 가중치 행렬의 소프트맥스 함수 전에 다음 토큰들의 위치에 음의 무한대의 값으로 처리
# = 마스킹 작업 (논문에서는 Masked Multi Head Attention)
# 이러한 어텐션 헤드를 여러 개 쌓으면 멀티 헤드 어텐션이 됨

import torch
from torch import nn

#!SECTION Multi Head Self Attention Layer Example
class AttentionHead(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.attention_head_size = hidden_size // num_attention_heads
        self.q = nn.Linear(hidden_size, self.attention_head_size)
        self.k = nn.Linear(hidden_size, self.attention_head_size)
        self.v = nn.Linear(hidden_size, self.attention_head_size)
        
    def forward(self, hidden_state, mask=None):
        attention_outputs = self.scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state), mask
        )
        return attention_outputs
    
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        dim_k = query.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights.bmm(value)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(hidden_size, num_attention_heads)
                for _ in range(num_attention_heads)
            ]
        )
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, mask=None):
        x = torch.cat([h(hidden_state, mask) for h in self.attention_heads], dim=-1)
        x = self.output_linear(x)
        return x