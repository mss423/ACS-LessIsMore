import json

import pandas as pd
# from fire import Fire
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from sklearn.metrics import classification_report
import os

LABELS = [
        "P1001",
        "P101",
        "P102",
        "P105",
        "P106",
        "P118",
        "P123",
        "P127",
        "P1303",
        "P131",
        "P1344",
        "P1346",
        "P135",
        "P136",
        "P137",
        "P140",
        "P1408",
        "P1411",
        "P1435",
        "P150",
        "P156",
        "P159",
        "P17",
        "P175",
        "P176",
        "P178",
        "P1877",
        "P1923",
        "P22",
        "P241",
        "P264",
        "P27",
        "P276",
        "P306",
        "P31",
        "P3373",
        "P3450",
        "P355",
        "P39",
        "P400",
        "P403",
        "P407",
        "P449",
        "P4552",
        "P460",
        "P466",
        "P495",
        "P527",
        "P551",
        "P57",
        "P58",
        "P6",
        "P674",
        "P706",
        "P710",
        "P740",
        "P750",
        "P800",
        "P84",
        "P86",
        "P931",
        "P937",
        "P974",
        "P991",
    ]

def linearize_input(text: str, head: str, tail: str) -> str:
    return f"Head Entity : {head} , Tail Entity : {tail} , Context : {text}"


def read_sample_dict(sample: dict):
    tokens = sample["tokens"]
    head = " ".join([tokens[i] for i in sample["h"][2][0]])
    tail = " ".join([tokens[i] for i in sample["t"][2][0]])
    return " ".join(tokens), head, tail


def load_data(path: str) -> pd.DataFrame:

    pairs = []
    with open(path) as f:
        raw = json.load(f)
        for label, lst in raw.items():
            y = LABELS.index(label)
            for sample in lst:
                text, head, tail = read_sample_dict(sample)
                x = linearize_input(text, head, tail)
                pairs.append((x, y))

    df = pd.DataFrame(pairs)
    df.columns = ["text", "labels"]
    df = df.sample(frac=1)  # Shuffle
    print(dict(path=path, data=df.shape, unique_labels=len(set(df["labels"].tolist()))))
    return df


def test_data(path: str = "data/new_split/new_train.json"):
    with open(path) as f:
        raw = json.load(f)
    breakpoint()


def run_train(data_train, path_test: str, epochs: int, target_dir: str):
    # data_train = load_data(path_train)
    data_test = load_data(path_test)

    args = ClassificationArgs(num_train_epochs=epochs, output_dir=target_dir,
                              cache_dir=os.path.join(target_dir,"cache_dir"),
                              tensorboard_dir=os.path.join(target_dir,"cache"),
                              use_multiprocessing=False, use_multiprocessing_for_evaluation=False,
                              overwrite_output_dir=True, save_steps=-1,
                              save_model_every_epoch = False)
    model = ClassificationModel(
        "bert", "bert-base-cased", num_labels=len(LABELS), args=args
    )
    model.train_model(data_train)
    result, model_outputs, wrong_predictions = model.eval_model(data_test)

    pred = model_outputs.argmax(-1).tolist()
    gold = data_test["labels"].tolist()
    report_dict = classification_report(gold, pred, output_dict=True)
    return report_dict
