#STUB 완전 연결층 2개로 구성되어 있음
# 논문에서는 첫 번째 층의 크기를 임베딩 크기의 네 배로 설정하고, 활성화 함수로 GELU를 사용

#!SECTION Feed-Forward Layer Example
import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x