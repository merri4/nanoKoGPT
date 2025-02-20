from tqdm import tqdm
import torch

class KorCharTokenizer() :
    def __init__(self, path=""):
        self.path = path
        self.stoi = {}
        self.itos = {}
        self.all_kor_chars = "./all_kor_chars.txt"
        self.special_chars = " 0123456789~!@#$%^&*()[]{}:;,./<>?/*-+=_`"
        self._build_vocab()
    
    def _build_vocab(self) :
        
        # 텍스트 추출 (경로 주어지지 않은 경우 모든 한국어 음소)
        if self.path == "" :
            self.path = self.all_kor_chars

        with open(self.path, "r", encoding="utf-8") as f :
            text = f.read()

        # 특수문자 추가
        text += self.special_chars

        # 중복 제거, 정렬
        chars = sorted(list(set(text)))

        # character 단위 양방향 변환 딕셔너리
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

    def __len__(self) :
        return len(self.stoi)

    def get_vocab_size(self) -> int :
        return len(self.stoi)

    # encode 함수
    # TODO : encode 함수에 매칭되는 게 없을 경우 unk 토큰 처리
    def encode(self, s : str) -> list :
        return [self.stoi[c] for c in s]

    # decode 함수
    def decode(self, l : list) -> str :
        return ''.join([self.itos[i] for i in l])        


def train_val_split(data, ratio) :
    n = int(ratio*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(split, train_data, val_data, context_length, batch_size) :
    if split == "train" :
        data = train_data
    else :
        data = val_data

    # batch_size 개의 인덱스 위치를 랜덤하게 잡고,
    ix = torch.randint(len(data)-context_length, (batch_size, ))
    
    # 그 위치에서 context_length만큼 문장을 파싱해온다.
    x = torch.stack([data[i : i+context_length] for i in ix])
    y = torch.stack([data[i+1 : i+context_length+1] for i in ix]) # y는 같은 인덱스에서 한 칸 다음 토큰을 보여준다.
    return x,y
    


if __name__ == "__main__" :

    PATH = "./shakespeare.txt"
    sample_text = "hi there."

    ### Build Tokenizer ===========================================
    tokenizer = KorCharTokenizer(PATH)
    # print(tokenizer.get_vocab_size())
    # print(tokenizer.encode(sample_text))
    # print(tokenizer.decode(tokenizer.encode(sample_text)))

    # TODO : pickle로 토크나이저 저장하기





    ### All data into one array ==================================
    with open(PATH, "r") as f :
        full_text = f.read()
    data = torch.tensor(tokenizer.encode(full_text))
    
    ### split dataset ============================================
    train_data, val_data = train_val_split(data, 0.9) 

    ### get batch ================================================
    CONTEXT_LENGTH = 8          # 8 characters per one chunk
    BATCH_SIZE = 4              # 4 chunks trained in parallel
    x_batch, y_batch = get_batch("train", train_data, val_data, CONTEXT_LENGTH, BATCH_SIZE)
    # print(x_batch)
    # print(tokenizer.decode(x_batch[0].tolist()))