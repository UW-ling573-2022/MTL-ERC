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
        default=10,
        help="Number of past utterances for generating dataset.",
    )
    parser.add_argument(
        "--num_future_utterances",
        type=int,
        default=0,
        help="Number of future utterances for generating dataset.",
    )
    parser.add_argument(
        "--speaker_in_context",
        action="store_true",
        help="Whether to include speaker in context.",
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
        "--train_task",
        type=str,
        nargs='+',
        default=["Emotion", "Speaker"],
        help="Task for training. (\"Emotion\", \"Speaker\",\"Sentiment\")",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        nargs='+',
        default=["MELD", "EmoryNLP", "MPDD"],
        help="Dataset for training. (\"MELD\", \"EmoryNLP\",\"MPDD\")",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="Emotion",
        help="Task for evaluation. (\"Emotion\", \"Speaker\",\"Sentiment\")",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="MPDD",
        help="Dataset for evaluation. (\"MELD\", \"EmoryNLP\",\"MPDD\")",
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
        default=get_root_dir() + "outputs/D3/pytorch_model.bin",
        help="Path to stored model",
    )
    parser.add_argument(
        "--output_file",
        type = str,
        default=get_root_dir() + "outputs/D3/predictions.out",
        help="Path to store predictions",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=6,
        help="Number of training epoch",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="roberta-base",
        help="Which checkpoint to use."
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
        default=16,
        help="How many examples per batch"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default=get_root_dir() + "results/D3_scores.out",
        help="Path to store results"
    )
    args = parser.parse_args()
    return args

