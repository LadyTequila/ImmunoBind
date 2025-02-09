# main.py
import argparse
from config import Config
from train import train_model
from evaluate import evaluate_model
from models import MultiTaskModel
from utils import save_model  # 如需加载模型，可增加 load_model 函数

def main():
    parser = argparse.ArgumentParser(description="Multi-task TCR-Epitope Binding Prediction")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="运行模式：train 或 eval")
    args = parser.parse_args()

    # 加载配置
    config = Config()

    if args.mode == "train":
        train_model(config)
    elif args.mode == "eval":
        # 初始化模型并加载预训练权重
        model = MultiTaskModel(config)
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        evaluate_model(model, config)

if __name__ == "__main__":
    main()
