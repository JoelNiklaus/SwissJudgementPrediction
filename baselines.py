import json

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from root import ROOT_DIR

seed = 42
lang = 'de'
lang_folder = ROOT_DIR / 'data' / lang

base_folder = ROOT_DIR / 'sjp'

task = 'single_label_classification'


def run_k_neighbors():
    run_baseline(KNeighborsClassifier(random_state=seed))


def run_random_forest():
    run_baseline(RandomForestClassifier(random_state=seed))


def run_linear_svc():
    run_baseline(LinearSVC(random_state=seed))


def run_decision_tree():
    run_baseline(DecisionTreeClassifier(random_state=seed))


def run_dummy_stratified():
    run_baseline(DummyClassifier(strategy="stratified", random_state=seed))


def run_dummy_majority():
    run_baseline(DummyClassifier(strategy="most_frequent", random_state=seed))


def run_dummy_random():
    run_baseline(DummyClassifier(strategy="uniform", random_state=seed))


def run_baseline(model):
    model_name = model.__class__.__name__

    if isinstance(model, DummyClassifier):
        model_name += "-" + model.strategy

    label_dict = load_labels()

    X_test, X_train, y_test, y_train, mlb = prepare_data(label_dict, model)

    # fit classifier
    if task == 'multi_label_classification':
        clf = MultiOutputClassifier(model)
    else:
        clf = model
    clf.fit(X_train, y_train)

    # make predictions
    preds = clf.predict(X_test)

    make_reports(label_dict, mlb, model_name, preds, y_test)


def make_reports(label_dict, mlb, model_name, preds, y_test):
    label_list = get_label_list(label_dict)

    baselines_folder = base_folder / 'baselines'
    model_folder = baselines_folder / model_name
    model_folder.mkdir(parents=True, exist_ok=True)
    if task == 'multi_label_classification':
        preds, labels = preds_to_bools(preds), labels_to_bools(y_test)
    if task == 'single_label_classification':
        preds, labels = preds, [label_dict["label2id"][label] for label in y_test]
    # write predictions file
    with open(f'{model_folder}/predictons.txt', "w") as writer:
        writer.write("index\tprediction\n")
        for index, pred in enumerate(preds):
            if task == 'multi_label_classification':
                pred_strings = mlb.inverse_transform(np.array([pred]))[0]
            if task == 'single_label_classification':
                pred_strings = [label_dict["id2label"][pred]]
            writer.write(f"{index}\t{pred_strings}\n")
    # write report file
    with open(f'{model_folder}/prediction_report.txt', "w") as writer:
        writer.write("Multilabel Confusion Matrix\n")
        writer.write("=" * 75 + "\n\n")
        writer.write("reading help:\nTN FP\nFN TP\n\n")
        matrices = multilabel_confusion_matrix(labels, preds)
        for i in range(len(label_list)):
            writer.write(f"{label_list[i]}\n{str(matrices[i])}\n")
        writer.write("\n" * 3)

        writer.write("Classification Report\n")
        writer.write("=" * 75 + "\n\n")
        report = classification_report(labels, preds, target_names=label_list, digits=4)
        writer.write(str(report))


def prepare_data(label_dict, model):
    label_list = get_label_list(label_dict)
    mlb = MultiLabelBinarizer().fit([label_list])

    train = pd.read_csv(lang_folder / 'train.csv')
    test = pd.read_csv(lang_folder / 'test.csv')

    if task == 'multi_label_classification':
        train['label'] = [mlb.transform([eval(labels)])[0] for labels in train.label]
        test['label'] = [mlb.transform([eval(labels)])[0] for labels in test.label]
    if task == 'single_label_classification':
        train['label'] = [label_dict["label2id"][label] for label in train.label]
    train.dropna(subset=['text', 'label'])
    test.dropna(subset=['text', 'label'])

    if isinstance(model, DummyClassifier):  # here we don't need the input anyway
        X_train = np.zeros((len(train.index), 1))
        X_test = np.zeros((len(test.index), 1))
    else:
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train.text).toarray()
        X_test = vectorizer.transform(test.text)

    y_train = train.label.tolist()
    y_test = test.label
    return X_test, X_train, y_test, y_train, mlb


def get_label_list(label_dict):
    label_list = list(label_dict["label2id"].keys())
    return label_list


def preds_to_bools(predictions, threshold=0.5):
    return [pl > threshold for pl in predictions]


def labels_to_bools(labels):
    return [tl == 1 for tl in labels]


def load_labels():
    with open(lang_folder / 'labels.json', 'r') as f:
        label_dict = json.load(f)
        label_dict['id2label'] = {int(k): v for k, v in label_dict['id2label'].items()}
        label_dict['label2id'] = {k: int(v) for k, v in label_dict['label2id'].items()}
    return label_dict


if __name__ == '__main__':
    run_dummy_stratified()
    run_dummy_majority()
    run_dummy_random()
