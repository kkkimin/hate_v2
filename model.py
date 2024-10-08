# 라이브러리 임포트
import pytorch_lightning as pl
import torch
import os
import datetime
import wandb
import pytz

from utils import compute_metrics
from data import prepare_dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_linear_schedule_with_warmup  
from tqdm import tqdm
import pandas as pd

def load_tokenizer_and_model_for_train(args):
    """학습(train)을 위해 '사전학습(protrainde)된 토크나이저와 모델을 huggingface에서 load"""
    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    print(model_config)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    print("--- Modeling Done ---")
    return tokenizer, model


def load_model_for_inference(model_name,model_dir):
    """추론(infer)에 필요한 모델과 토크나이저 load """
    # load tokenizer
    Tokenizer_NAME = model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    return tokenizer, model


def load_trainer_for_train(args, model, hate_train_dataset, hate_valid_dataset):
    """학습(train)을 위한 huggingface trainer 설정"""
    training_args = TrainingArguments(
        output_dir = args.save_path + "/results",  # 모델 훈련 후 결과 파일이 저장되는 디렉토리
        save_total_limit = args.save_limit,  # 저장할 모델의 최대 개수를 설정(for 공간절약)
        save_steps = args.save_step,  # 모델을 저장할 스텝 간격을 설정
        num_train_epochs = args.epochs,  
        learning_rate = args.lr,  
        lr_scheduler_type="linear",
        per_device_train_batch_size = args.batch_size,  # 각 장치에서 훈련할 배치 사이즈를 설정
        per_device_eval_batch_size = 8,  # 평가 시 사용할 배치 사이즈를 설정
        warmup_steps = args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay = args.weight_decay,  # 가중치 감소 값을 설정(과적합 방지를 위해)
        logging_dir = args.save_path + "logs",  # directory for storing logs
        logging_steps = args.logging_step,  # 로그 파일이 저장될 디렉토리 경로를 설정
        eval_strategy = "steps",  # 로그를 저장할 스텝 간격을 설정
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps = args.eval_step,  # 평가할 스텝 간격을 설정
        load_best_model_at_end = True,  # 훈련 종료 후 최상의 모델을 로드할지 여부를 설정
        report_to = "wandb",  # W&B 로깅 활성화(로깅 툴을 지정)
        run_name = args.run_name,  # W&B에 기록될 실행 이름을 설정(args.run_name: 실험을 식별할 수 있는 이름임)
    )

    ## Add callback & optimizer & scheduler
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=20,      # 몇 에폭 동안 검증 손실이 개선되지 않으면 훈련을 중단할지를 결정
        early_stopping_threshold=0.001  # 개선을 고려하는 손실 변화의 최소 크기
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    print("--- Set training arguments Done ---")

    trainer = Trainer(
        model=model,  # 기본 BERT 모델
        args=training_args,  # training arguments, defined above
        train_dataset=hate_train_dataset,  
        eval_dataset=hate_valid_dataset,  
        compute_metrics=compute_metrics,  
        callbacks=[MyCallback],
        
        # 옵티마이저와 스케쥴러 설정하는 부분
        optimizers=(
            optimizer, 
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=len(hate_train_dataset) * args.epochs // args.batch_size,
            )
        ),
    )
    print("--- Set Trainer Done ---")

    return trainer


def train(args, combined_data):   # combined_data 인자를 제거
    """모델을 학습(train)하고 best model을 저장"""
    
    # 원본 데이터 로드
    original_data_path = os.path.join(args.original_data_dir, args.original_data_file)
    original_data = pd.read_csv(original_data_path)
    
    # 증강 데이터 파일 확인 및 로드
    augmented_data_path = os.path.join(args.augmented_data_dir, args.augmented_data_file)
    
    if os.path.exists(augmented_data_path):
        augmented_data = pd.read_csv(augmented_data_path)
        print(f"AEDA로 증강된 데이터 개수: {len(augmented_data)}")
    else:
        augmented_data = pd.DataFrame()  # 증강 데이터가 없으면 빈 데이터프레임 할당
        print(f"증강된 데이터 파일이 {augmented_data_path}에 존재하지 않습니다. 빈 데이터프레임 사용.")
    
    # 원본 데이터와 증강 데이터 결합
    combined_data = pd.concat([original_data, augmented_data], ignore_index=True)
    print(f"결합된 데이터의 행의 수: {len(combined_data)}")

    wandb.init(project=args.wandb_project, name=args.run_name)
    wandb.config.update(args)

    # fix a seed
    pl.seed_everything(seed=42, workers=False)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # set model and tokenizer
    tokenizer, model = load_tokenizer_and_model_for_train(args)
    model.to(device)

    # set data (combined_data를 추가로 전달)
    hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset = (
        prepare_dataset(args.dataset_dir, tokenizer, args.max_len, combined_data)  
    )

    # set trainer
    trainer = load_trainer_for_train(
        args, model, hate_train_dataset, hate_valid_dataset
    )

    # train model
    print("--- Start train ---")
    trainer.train()
    print("--- Finish train ---")
    model.save_pretrained(args.model_dir)

