from tokenizers import BertWordPieceTokenizer


# 데이터 경로
data_path = "./input.txt"

# 토크나이저 세팅
tokenizer = BertWordPieceTokenizer(
    # clean_text=True,
    # handle_chinese_chars=True,
    # strip_accents=False, # Must be False if cased model
    # lowercase=False,
    # wordpieces_prefix="##",
    )

tokenizer.train(
    files=data_path,
    vocab_size=3000,
    limit_alphabet=500,
    min_frequency=2,
    )

# encode / decode test
# vocab = tokenizer.get_vocab()
# print(sorted(vocab, key=lambda x: vocab[x]))


sample_text = "근처에 얼씬도 못하게해야지."

encoded_text = tokenizer.encode(sample_text)
decoded_text = tokenizer.decode(encoded_text.ids)

print(encoded_text.ids)     # list
print(encoded_text.tokens)  # list
print(decoded_text)         # string


# 저쪽에 tokenizer 종류 선택할 수 있게 하기. 

# https://wikidocs.net/99893