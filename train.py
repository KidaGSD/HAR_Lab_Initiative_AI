import argparse
from src.config import load_config
from src.training.loop import train_one_split, run_cv
from src.training.probe import train_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hierarchical.yaml")
    parser.add_argument("--processed-dir", type=str, default="data/processed_ego4d")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument('--run-name', type=str, default='hierarchical_har')
    parser.add_argument('--probe', action='store_true', help='Train action probe on frozen model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for probing')
    parser.add_argument('--cv', action='store_true', help='Use K-fold cross validation')
    parser.add_argument('--n-folds', type=int, default=4, help='Number of CV folds (default: 4)')
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.probe:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for probe mode")
        train_probe(args, config)
    elif args.cv:
        run_cv(args, config)
    else:
        train_one_split(args, config)


if __name__ == "__main__":
    main()
