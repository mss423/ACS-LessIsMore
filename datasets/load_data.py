import os
import pandas as pd
import json

FEWREL_LABELS = [
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
politics_labels = ['O', 'B-country', 'B-politician', 'I-politician', 'B-election', 'I-election', 'B-person', 'I-person', 'B-organisation', 'I-organisation', 'B-location', 'B-misc', 'I-location', 'I-country', 'I-misc', 'B-politicalparty', 'I-politicalparty', 'B-event', 'I-event']
science_labels = ['O', 'B-scientist', 'I-scientist', 'B-person', 'I-person', 'B-university', 'I-university', 'B-organisation', 'I-organisation', 'B-country', 'I-country', 'B-location', 'I-location', 'B-discipline', 'I-discipline', 'B-enzyme', 'I-enzyme', 'B-protein', 'I-protein', 'B-chemicalelement', 'I-chemicalelement', 'B-chemicalcompound', 'I-chemicalcompound', 'B-astronomicalobject', 'I-astronomicalobject', 'B-academicjournal', 'I-academicjournal', 'B-event', 'I-event', 'B-theory', 'I-theory', 'B-award', 'I-award', 'B-misc', 'I-misc']
music_labels = ['O', 'B-musicgenre', 'I-musicgenre', 'B-song', 'I-song', 'B-band', 'I-band', 'B-album', 'I-album', 'B-musicalartist', 'I-musicalartist', 'B-musicalinstrument', 'I-musicalinstrument', 'B-award', 'I-award', 'B-event', 'I-event', 'B-country', 'I-country', 'B-location', 'I-location', 'B-organisation', 'I-organisation', 'B-person', 'I-person', 'B-misc', 'I-misc']
literature_labels = ["O", "B-book", "I-book", "B-writer", "I-writer", "B-award", "I-award", "B-poem", "I-poem", "B-event", "I-event", "B-magazine", "I-magazine", "B-literarygenre", "I-literarygenre", 'B-country', 'I-country', "B-person", "I-person", "B-location", "I-location", 'B-organisation', 'I-organisation', 'B-misc', 'I-misc']
ai_labels = ["O", "B-field", "I-field", "B-task", "I-task", "B-product", "I-product", "B-algorithm", "I-algorithm", "B-researcher", "I-researcher", "B-metrics", "I-metrics", "B-programlang", "I-programlang", "B-conference", "I-conference", "B-university", "I-university", "B-country", "I-country", "B-person", "I-person", "B-organisation", "I-organisation", "B-location", "I-location", "B-misc", "I-misc"]

domain2labels = {"politics": politics_labels, "science": science_labels, "music": music_labels, "literature": literature_labels, "ai": ai_labels}


def load_train_data(datadir, dataset="sst2", synthetic=False):
    '''
    Returns the training dataframe train_df and number of labels as an int
    '''
    fname = "syn-train" if synthetic else "train"

    if dataset == "sst2":
        path = os.path.join(datadir,"SST2")
        train_df = pd.read_csv(os.path.join(path, fname+".tsv"),sep='\t',header=0)
        train_df.columns = ['sentence', 'label']
        return train_df, 2 
    elif dataset == "fewrel":
        path = os.path.join(datadir,"FewRel")
        train_df = load_fewrel_data(os.path.join(path, fname+".json"))
        return train_df, len(FEWREL_LABELS)
    elif dataset == "aste":
        path = os.path.join(datadir,"ASTE/" + fname + ".txt")
        train_df = load_aste_sentences(path)
        return train_df, 3
    elif dataset == "crossner":
        path = os.path.join(datadir,"CrossNER/" + fname + ".txt")
        train_df = load_crossner_sentences(path)

        if synthetic:
            dev_path = os.path.join(datadir,"CrossNER/dev.txt")
            dev_df = load_crossner_sentences(dev_path)
            train_df = pd.concat([train_df, dev_df], ignore_index=True)
            
        return train_df, 16
    else:
        raise ValueError('Invalid dataset name passed!')

def load_test_data(datadir, dataset="sst2"):
    if dataset == "sst2":
        path = os.path.join(datadir,"SST2")
        test_df = pd.read_csv(os.path.join(path,"test.tsv"),sep='\t',header=0)
        test_df.columns = ['label','sentence']
        test_df = test_df[['sentence', 'label']]
        return test_df
    elif dataset == "fewrel":
        path = os.path.join(datadir,"FewRel")
        test_df = load_fewrel_data(os.path.join(path,"test.json"))
        return test_df
    elif dataset == "aste":
        path = os.path.join(datadir,"ASTE/test.txt")
        test_df = load_aste_sentences(path)
        return test_df
    elif dataset == "crossner":
        path = os.path.join(datadir,"CrossNER/test.txt")
        test_df = load_crossner_sentences(path)
        return test_df
    else:
        raise ValueError('Invalid dataset name passed!')

# ------ CrossNER Data Functions ------ #
def load_crossner_sentences(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    sentences = []
    current_sentence = []

    for line in lines:
        if line.strip() == '':  # Empty line indicates end of sentence
            if current_sentence:
                sentences.append(' '.join(current_sentence))
                current_sentence = []
        else:
            word, _ = line.strip().split()  # Extract word, ignore label
            current_sentence.append(word)

    # Handle potential last sentence without an empty line
    if current_sentence:
        sentences.append(' '.join(current_sentence))

    return pd.DataFrame({"sentence": sentences})

def load_crossner_train(path, dev=False):
    with open(path, 'r') as f:
        lines = f.readlines()

    data = []
    sentence_id = 0
    if dev: 
        sentence_id += 2700

    for line in lines:
        if line.strip() == '':  # Empty line indicates end of sentence
            sentence_id += 1
        else:
            token, label = line.strip().split()  # Extract word, ignore label
            data.append([sentence_id, token, label])
    return data


# ------ ASTE Data Functions ------ #
def load_aste_sentences(path):
    sentences = []
    labels    = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("#### #### ####")
            sentences.append(parts[0])
            labels.append(parts[1])
    return pd.DataFrame({"sentence": sentences, "label": labels})


# ------ FewRel Data Functions ------- #
def linearize_input(text, head, tail):
    return f"Head Entity : {head} , Tail Entity : {tail} , Context : {text}"


def read_sample_dict(sample):
    tokens = sample["tokens"]
    head = " ".join([tokens[i] for i in sample["h"][2][0]])
    tail = " ".join([tokens[i] for i in sample["t"][2][0]])
    return " ".join(tokens), head, tail


def load_fewrel_data(path):
    pairs = []
    with open(path) as f:
        raw = json.load(f)
        for label, lst in raw.items():
            y = FEWREL_LABELS.index(label)
            for sample in lst:
                text, head, tail = read_sample_dict(sample)
                x = linearize_input(text, head, tail)
                pairs.append((x, y))

    df = pd.DataFrame(pairs)
    df.columns = ["sentence", "label"]
    df = df.sample(frac=1)  # Shuffle
    print(dict(path=path, data=df.shape, unique_labels=len(set(df["label"].tolist()))))
    return df