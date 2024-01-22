from tokenizers import CharBPETokenizer

tokenizer = CharBPETokenizer(suffix='', lowercase=True)

special_tokens = ['<pad>', '<unk>', '<s>', '</s>', '<mask>']
#NOTE <pad> : 서로 다른 길이의 문장을 처리하기 위해 짧은 문장을 긴 문장의 길이와 맞게 padding (token_id=0)
#NOTE <unk> : 토크나이저가 모르는 단어를 만나면 unknown으로 처리 (token_id=1)
#NOTE <s> : 문장의 시작. BOS(Begin of sentence), CLS(Classification)으로도 사용. (token_id=2)
#NOTE </s> : 문장의 끝. EOS(End of sentence), SEP(Separator)으로도 사용. (token_id=3)
#NOTE <mask> : MLM 학습 시 쓰이는 토큰. 토큰을 마스킹해 이 토큰을 맞추는 문제를 풀 때 사용

vocab_size = 36000
min_frequency = 2

data_path = r'C:\Users\ontheit\Desktop\2a\kakaotech\wikitext-2-raw'
tokenizer.train(files=data_path + '/wiki.train.raw',
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
                suffix='')

tokenizer.save('tokenizer.json')