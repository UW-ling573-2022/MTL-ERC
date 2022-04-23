import argparse
from os.path import abspath, dirname


def get_root_dir():
    result = dirname(abspath(__file__))
    src = "/src"
    if result.endswith(src):
        result = result[:-len(src) + 1]
    return result


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Specify the random seed."
    )
    parser.add_argument(
        "--num_past_utterances",
        type=int,
        default=1,
        help="Number of past utterances for generating dataset.",
    )
    parser.add_argument(
        "--num_future_utterances",
        type=int,
        default=1,
        help="Number of future utterances for generating dataset.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=get_root_dir() + "data/",
        help="Path to directory containing raw dataset",
    )
    parser.add_argument(
        "--do_train",
        action='store_true',
        help="Train the model and store to default directory",
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        default="emotion",
        help="Either \"emotion\" or \"speaker\" for evaluation",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=get_root_dir() + "outputs/multi_task_model",
        help="Path to store checkpoints",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=get_root_dir() + "outputs/D2",
        help="Path to stored model",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=5,
        help="Number of training epoch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="How many examples per batch"
    )
    args = parser.parse_args()
    return args

