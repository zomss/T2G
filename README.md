## Text to Gloss

### 사용법

    python run.py

data_KO 폴더에 한국 수어 데이터셋을 삽입하시면 훈련 및 평가가 진행됩니다.

env/env.py 파일에 LANGUAGE=='ko' 항목에 있는 내용을 편집하시면 데이터셋 변경이 가능합니다.

### 환경


KLUE-BERT-base (https://github.com/KLUE-benchmark/KLUE)와 Transformer Decoder를 결합한 뒤, greedy decoding을 통해 output sequence를 생성하였음.