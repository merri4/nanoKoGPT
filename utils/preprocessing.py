## 파일 불러와서

## 띄어쓰기가 두 줄 이상일 경우 2줄로 줄인다.
## 엔터 전에 space가 있으면 없앤다.

import re  

def preprocess_text(raw_text) :
    # Replace more than two consecutive newlines (with or without spaces) with exactly two newlines  
    processed_text = re.sub(r'(\n\s*){3,}', '\n\n', raw_text)  
    # remove all spaces before new line character.
    processed_text = re.sub(r'[ \t]+\n', '\n', processed_text)
    return processed_text

if __name__ == "__main__" :

    data_path = "./input.txt"
    out_path = './input_processed.txt'

    raw_text = ""
    with open(data_path, "r", encoding="utf-8") as f :
        raw_text = f.read()

    processed_text = preprocess_text(raw_text)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)

    print("Successfully Saved!")