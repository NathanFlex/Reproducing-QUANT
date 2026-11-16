import os
import time
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from aeon.datasets import load_from_ts_file
from quant import Quant
from quant_scale_aware import ScaleAwareQuant
# from quant_scale_6 import ScaleAwareQuant ----- Swap with the one above to use ScaleAwareQuant with 6 Representation Features
import random
import pandas as pd

RANDOM_SEED = 34

random.seed(RANDOM_SEED)

data_root_path = "../datasets/"
resample_directory = "../PythonResampleIndices"

dataset_names = open("ucr142.txt", "r").read().splitlines()

# Trackers for final macro averages across everything
metrics_without_quant = {"acc": [], "bal_acc": [], "log_loss": [], "auroc": []}
metrics_with_quant = {"acc": [], "bal_acc": [], "log_loss": [], "auroc": []}
dataset_metrics = {}


def combine_and_get_resample(data_train, data_test, resample_path):
    X_train, y_train = load_from_ts_file(data_train)
    X_test, y_test = load_from_ts_file(data_test)

    combined_data = np.concatenate([X_train, X_test], axis=0)
    combined_labels = np.concatenate([y_train, y_test], axis=0)

    train_index_list = []
    test_index_list = []

    for index_file in os.listdir(resample_path):
        f = open(os.path.join(resample_path, index_file), "r")
        indices = f.read().splitlines()
        if "TRAIN" in index_file:
            train_index_list.append(indices)
        if "TEST" in index_file:
            test_index_list.append(indices)

    return combined_data, combined_labels, train_index_list, test_index_list

def evaluate_and_store(X_train, y_train, X_test, y_test, name, resample_idx, m_without, m_with):
    if name not in dataset_metrics:
        dataset_metrics[name] = {"acc": [], "bal_acc": [], "log_loss": [], "auroc": [], "time_transform_training": [],"time_transform_test": [],"time_fit_classifier": [],"time_predict": []}

    # Convert arrays to torch tensors for QUANT
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    n_classes = len(np.unique(y_train_arr))

    transform = ScaleAwareQuant(depth = 6, div_schedule=[4, 8, 8, 8, 16, 16]) # To implement baseline quant, replace the ScaleAwareQuant Class with Quant and set div to 4

    t1 = time.time()
    X_train_q = transform.fit_transform(X_train_t, y_train_arr)
    dataset_metrics[name]["time_transform_training"].append(time.time() - t1)

    t2 = time.time()
    X_test_q = transform.transform(X_test_t)
    dataset_metrics[name]["time_transform_test"].append(time.time() - t2)

    clf_q = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy", n_jobs=-1, random_state=RANDOM_SEED)
    t3 = time.time()
    clf_q.fit(X_train_q, y_train_arr)
    dataset_metrics[name]["time_fit_classifier"].append(time.time() - t3)

    t4 = time.time()
    preds_q = clf_q.predict(X_test_q)
    dataset_metrics[name]["time_predict"].append(time.time() - t4)
    probs_q = clf_q.predict_proba(X_test_q)

    m_with["acc"].append(np.mean(preds_q == y_test_arr))
    m_with["bal_acc"].append(balanced_accuracy_score(y_test_arr, preds_q))
    m_with["log_loss"].append(log_loss(y_test_arr, probs_q, labels=clf_q.classes_))

    dataset_metrics[name]["acc"].append(np.mean(preds_q == y_test_arr))
    dataset_metrics[name]["bal_acc"].append(balanced_accuracy_score(y_test_arr, preds_q))
    dataset_metrics[name]["log_loss"].append(log_loss(y_test_arr, probs_q, labels=clf_q.classes_))

    if n_classes == 2:
        roc_score = roc_auc_score(y_test_arr, probs_q[:, 1])
    else:
        roc_score = roc_auc_score(y_test_arr, probs_q, multi_class='ovr', average='macro', labels=clf_q.classes_)

    m_with["auroc"].append(roc_score)
    dataset_metrics[name]["auroc"].append(roc_score)

    print(f"[{name} - R{resample_idx}] QUANT Acc: {m_with['acc'][-1]:.2f}%, Bal_Acc {m_with['bal_acc'][-1]:.2f}%, Log Loss {m_with['log_loss'][-1]:.4f}, ROC AUC {m_with['auroc'][-1]:.4f}")


print(f"Starting evaluation across {len(dataset_names)} datasets...\n")

time_start = time.time()
for dataset_name in [dataset_names[i] for i in np.arange(len(dataset_names), step=1)]:
    dataset_dir = os.path.join(data_root_path, dataset_name)

    if not os.path.exists(dataset_dir):
        print(f"Directory not found for {dataset_name}, skipping.")
        continue

    # Check if multiple resamples (0 to 29) exist for this dataset
    resamples_found = False
    for r in range(30):
        train_path = os.path.join(dataset_dir, f"{dataset_name}{r}_TRAIN.ts")
        test_path = os.path.join(dataset_dir, f"{dataset_name}{r}_TEST.ts")

        if os.path.exists(train_path) and os.path.exists(test_path):
            resamples_found = True
            X_train, y_train = load_from_ts_file(train_path)
            X_test, y_test = load_from_ts_file(test_path)

            evaluate_and_store(X_train, y_train, X_test, y_test, dataset_name, r,
                               metrics_without_quant, metrics_with_quant)
        else:
            break

    # If no multi-resample files exist, create the splits
    if not resamples_found:
        train_path = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
        test_path = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")

        if os.path.exists(train_path) and os.path.exists(test_path):
            combined_data, combined_labels, train_indices, test_indices = combine_and_get_resample(train_path, test_path, os.path.join(resample_directory, dataset_name))
            for r in range(30):
                X_train = combined_data[[int(x) for x in train_indices[r]]]
                y_train = combined_labels[[int(x) for x in train_indices[r]]]
                X_test = combined_data[[int(x) for x in test_indices[r]]]
                y_test = combined_labels[[int(x) for x in test_indices[r]]]

                evaluate_and_store(X_train, y_train, X_test, y_test, dataset_name, r, metrics_without_quant, metrics_with_quant)

        else:
            print(f"  [Warning] Could not find default TRAIN/TEST .ts files for {dataset_name}")

time_end = time.time()
overall_df = pd.DataFrame.from_dict(dataset_metrics, orient='index')
overall_df.to_csv("results.csv")

print("FINAL MACRO AVERAGE ACROSS ALL PROCESSED DATASETS")

print(f"Mean Accuracy:     {np.mean(metrics_with_quant['acc']):.3f}")
print(f"Mean Balanced Acc: {np.mean(metrics_with_quant['bal_acc']):.3f}")
print(f"Mean Log Loss:     {np.mean(metrics_with_quant['log_loss']):.4f}")
print(f"Mean AUROC (OVR):  {np.mean(metrics_with_quant['auroc']):.4f}")
print(f"Total Time: {time_end - time_start:.4f}s")