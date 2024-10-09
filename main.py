import os
import sys  
import argparse
import pandas as pd
import wandb
import datetime
import pytz
from model import train
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# parse_args 함수(인자) : model 학습 및 추론에 쓰일 config 를 관리
def parse_args():
    """학습(train)과 추론(infer)에 사용되는 arguments를 관리하는 함수"""
    parser = argparse.ArgumentParser(description="Training and Inference arguments")    

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/mean6021/hate_v2_git",
        help="원본 데이터셋 디렉토리 경로",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help='모델 타입 (예: "bert", "electra")',
    )

    parser.add_argument(
        "--model_name", type=str, default="klue/bert-base", help='모델 이름',
    )   
    parser.add_argument(
        "--save_path", type=str, default="./model", help="모델 저장 경로"
    )
    parser.add_argument(
        "--save_step", type=int, default=300, help="모델을 저장할 스텝 간격"  # 모델이 훈련 도중에 저장되는 간격을 설정
    )
    parser.add_argument(
        "--logging_step", type=int, default=300, help="로그를 출력할 스텝 간격"
    )
    parser.add_argument(
        "--eval_step", type=int, default=300, help="모델을 평가할 스텝 간격"  
    )
    parser.add_argument(
        "--save_limit", type=int, default=10, help="저장할 모델의 최대 개수"  # 저장할 체크포인트(모델)의 최대 개수
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 값")
    parser.add_argument("--epochs", type=int, default=10, help="에폭 수 (예: 10)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="배치 사이즈 (메모리에 맞게 조절, 예: 16 또는 32)",
    )
    parser.add_argument(
        "--max_len", type=int, default=256, help="입력 시퀀스의 최대 길이"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="학습률(learning rate)")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="가중치 감소(weight decay) 값"
    )
    parser.add_argument("--warmup_steps", type=int, default=500, help="워밍업 스텝 수")
    parser.add_argument(
        "--scheduler", type=str, default="linear", help="학습률 스케줄러 타입"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./best_model",
        help="추론 시 불러올 모델의 경로",
    )         
    parser.add_argument(    # wandb 수정 이름 변경!
        "--run_name",
        type=str,
        default="klue/bert-base_e1",
        help="wandb 에 기록되는 run name",
    )
    parser.add_argument(
    "--wandb_project",
    type=str,
    default="GCP_KIN",  # 기본 프로젝트 이름 설정
    help="WandB 프로젝트 이름"
    )

    parser.add_argument("--original_data_dir", type=str, default="./", help="원본 데이터의 경로")    # "./"는 현재 디렉토리 의미
    parser.add_argument("--original_data_file", type=str, default="train.csv", help="원본 데이터의 파일 이름")
    parser.add_argument("--augmented_data_dir", type=str, default="./", help="AEDA로 증강된 데이터의 경로")
    parser.add_argument("--augmented_data_file", type=str, default="aeda.csv", help="AEDA로 증강된 데이터 파일 이름")

    args = parser.parse_args()
    return args


from tqdm import tqdm

if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"    # tokenizer 사용 시 warning 방지
    args = parse_args()
    
    # 원본데이터(train.csv) 로딩
    original_data_path = os.path.join(args.original_data_dir, args.original_data_file)  # "./train.csv"
    original_data = pd.read_csv(original_data_path)  
    print(f"원본 데이터 개수: {len(original_data)}")

    # 증강된데이터(aeda.csv) 로딩
    augmented_data_path = os.path.join(args.augmented_data_dir, args.augmented_data_file)  # "./aeda.csv"
    
    if os.path.exists(augmented_data_path):
        augmented_data = pd.read_csv(augmented_data_path)  # augmented_data는 "aeda.csv"
        print(f"AEDA로 증강된 데이터 개수: {len(augmented_data)}")
    else:
        raise FileNotFoundError(f"증강된 데이터 파일이 {augmented_data_path}에 존재하지 않습니다.")  
    
    # combined_data = train데이터 + AEDA증강데이터
    combined_data = pd.concat([original_data, augmented_data], ignore_index=True)   # ignore_index=True 행방향 연결
    print(f"결합된 데이터의 행의 수: {len(combined_data)}")

    # 한국 표준시 (KST)로 현재 시간 가져오기
    kst = pytz.timezone('Asia/Seoul')
    current_time = datetime.datetime.now(kst)
    formatted_time = current_time.strftime("%m%d_%H%M")

    # 기존 run_name에 시간을 추가
    run_name = f"{args.run_name}_{formatted_time}"

    # 프로젝트 이름을 설정하여 실험 기록 시작
    wandb.init(project=args.wandb_project, name=run_name)    
    train(args, combined_data)                           # train 함수를 호출하여 실제로 모델 학습(train)을 진행함.

