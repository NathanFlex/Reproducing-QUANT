import pandas as pd
import numpy as np

df = pd.read_csv("./results_scale_6.csv")
new_df = df.copy()

for i in range(df.shape[0]):
    values = df.iloc[i]
    acc = list(eval(values["acc"]))
    bal_acc = list(eval(values["bal_acc"]))
    logloss = list(eval(values["log_loss"]))
    auc = list(eval(values["auroc"]))
    t1 = list(eval(values["time_transform_training"]))
    t2 = list(eval(values["time_transform_test"]))
    t3 = list(eval(values["time_fit_classifier"]))
    t4 = list(eval(values["time_predict"]))

    dict_mapping = {"acc":acc, "bal_acc":bal_acc, "log_loss":logloss,"auroc":auc,"time_transform_training":t1,"time_transform_test":t2,"time_fit_classifier":t3,"time_predict":t4}

    for name, attribute in dict_mapping.items():
        if len(attribute) > 1:
            mean_att = np.mean(np.array(attribute))
        elif len(attribute) == 1:
            mean_att = attribute[0]

        new_df.at[i, name] = mean_att

new_df.to_csv("processed.csv")



