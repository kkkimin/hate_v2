import os
import pandas as pd
import torch
import wandb
import torchmetrics
import utils

class hate_dataset(torch.utils.data.Dataset):     # tokenized 입력을 받아와서 dataset class 로 반환하는 함수
    """dataframe을 torch dataset class로 변환"""
    def __init__(self, hate_dataset, labels):  # __inin__메서드 : 객체 초기화
        self.dataset = hate_dataset
        self.labels = labels

    def __getitem__(self, idx):  # dataset에 인덱싱 허용하는 함수 (밑에 설명 참고)
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
#----------------------------------------------------------------------------------------------------------------------------------------

def load_data(dataset_dir):                # csv file을 dataframe으로 반환하는 함수
    """csv file을 dataframe으로 load"""
    dataset = pd.read_csv(dataset_dir)
    print("dataframe의 형태")
    print("-"*100)
    print(dataset.head())
    return dataset

#----------------------------------------------------------------------------------------------------------------------------------------
def construct_tokenized_dataset(dataset, tokenizer, max_length):   # dataframe 을 입력으로 받아와서 토크나이징한 후에 반환하는 함수
    """입력값(input)에 대하여 토크나이징"""
    print("tokenizer에 들어가는 데이터 형태")
    print(dataset["input"][:3])                # dataset은 토크나이징할 데이터셋. input이라는 열에 텍스트 데이터가 포함되어 있어야함.
    tokenized_sentences = tokenizer(           # tokenizer : 텍스트를 토큰화하기 위한 모델의 토크나이저 객체
        dataset["input"].tolist(),             # tokenizer는 dataset["input"].tolist()의 각 문장을 토큰화 함, ※ tolist() : 값or컬럼을 리스트로 만들 수 있는 함수
        # 옵션 설명 #
        return_tensors="pt",                   # return_tensors : 토큰화된 결과를 PyTorch 텐서 형태로 반환
        padding=True,                          # padding=True: 입력 시퀀스의 길이가 다를 경우, 길이를 맞추기 위해 패딩을 추가
        max_length=max_length,                 # 토큰화된 시퀀스의 최대 길이(길면 잘리고, 짧으면 패딩됨)
        add_special_tokens=True,
        return_token_type_ids=False,           # 문장쌍을 처리할 때 사용하는 '토큰 유형 ID'를 반환하지 않겠다.
    )                                          # BERT 이후 모델(RoBERTa 등) 사용할때 False
    print("tokenizing 된 데이터 형태")
    print("-"*100)
    print(tokenized_sentences[:3])
    return tokenized_sentences        # (type : tensor)

#----------------------------------------------------------------------------------------------------------------------------------------
def prepare_dataset(dataset_dir, tokenizer, max_len, combined_data):   # csv file 로부터 읽어온 데이터를 tokenized dataset 으로 반환하는 함수
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    # load_data (type : df)
    valid_dataset = load_data(os.path.join(dataset_dir, "dev.csv"))
    test_dataset = load_data(os.path.join(dataset_dir, "test.csv"))
    print("--- data loading Done ---")

    # split label (type : narray)
    ## train_label = train_dataset["output"].values  # train_dataset의 'output' 열의 값을 'Series' 형태로 반환.
    valid_label = valid_dataset["output"].values  # .values: 해당 Series의 값을 'NumPy 배열' 형태로 변환
    test_label = test_dataset["output"].values    # 즉, output 열의 값들을 리스트가 아닌 "배열"로 변환하여 이후의 모델 학습 등에서 사용할 수 있도록 함.

    # combined_data에서 학습 데이터를 로드하고 라벨 분리
    train_label = combined_data["output"].values  # combined_data에서 output 열을 사용
    print("--- Combined Data 로드 완료 ---")

    # combined_data를 tokenizing
    tokenized_train = construct_tokenized_dataset(combined_data, tokenizer, max_len)
    print("--- Combined Data tokenizing 완료 ---")

    # tokenizing dataset (type : tensor)  
    tokenized_valid = construct_tokenized_dataset(valid_dataset, tokenizer, max_len)
    tokenized_test = construct_tokenized_dataset(test_dataset, tokenizer, max_len)
    print("--- data tokenizing Done ---")

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    hate_test_dataset = hate_dataset(tokenized_test, test_label)
    print("--- pytorch dataset class Done ---")

    return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset
      
