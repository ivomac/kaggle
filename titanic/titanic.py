###############
### TITANIC ###
###############

import graphviz
from sklearn.tree import export_graphviz
from subprocess import run
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from itertools import product
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from collections import Counter


def bar(*args, **kwargs):
    return tqdm(
        *args,
        colour="blue",
        dynamic_ncols=True,
        **kwargs,
    )


@dataclass
class Set:
    ID: pd.Series
    X: pd.DataFrame
    Y: pd.Series | None = None


##################
### DATA SETUP ###
##################

OUT_DIR = Path("./out")
DATA_DIR = Path("./data")

train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")

all = pd.concat([train_data, test_data])
Id = all.loc[:, "PassengerId"]
Y = all.loc[:, "Survived"]

################
### FILL NAN ###
################

groups = all.groupby(["Pclass", "Sex"])

all.fillna(
    {
        "Cabin": "U",
        "Embarked": groups["Embarked"].transform(lambda x: x.mode().iat[0]),
        "Fare": groups["Fare"].transform(lambda x: x.mean()),
        "Age": groups["Age"].transform(lambda x: x.median()),
    },
    inplace=True,
)

################
### FEATURES ###
################


def title(name: str):
    parts = name.split()
    for part in parts:
        if "." in part:
            if part in ["Mrs.", "Mme.", "Miss.", "Mlle.", "Ms."]:
                return 0
            if part == "Master.":
                return 1
            if part == "Mr.":
                return 4
    return 3


def cabin_letters(cabin: str):
    counts = Counter([c for c in cabin if c.isalpha()])
    most_common = counts.most_common(1)[0][0]
    if most_common in ["B", "C", "D", "T"]:
        return "U"
    return most_common


def multi_cabin(cabin: str):
    return len(cabin.split()) > 1


age_bins = [0, 12, 24, 36, 48, 60, all.loc[:, "Age"].max() + 1]
all.loc[:, "Age"] = pd.cut(
    all.loc[:, "Age"],
    bins=age_bins,
    labels=range(len(age_bins) - 1),
)

all.loc[:, "Fare"] = pd.qcut(all.loc[:, "Fare"], q=5, labels=range(5))

all.loc[:, "Family"] = (all.loc[:, "SibSp"] + all.loc[:, "Parch"] + 1).map(
    lambda x: 0 if x == 1 else 1 if x < 5 else 2
)

all.loc[:, "Title"] = all.loc[:, "Name"].map(title)
all.loc[:, "CabinLetters"] = all.loc[:, "Cabin"].map(cabin_letters)
all.loc[:, "MultiCabin"] = all.loc[:, "Cabin"].map(multi_cabin)
all.loc[:, "NumTicket"] = all.loc[:, "Ticket"].map(lambda x: x.isnumeric())
all.loc[:, "Sex"] = all.pop("Sex") == "male"

Cat_cols = [
    "Pclass",
    "Age",
    "Fare",
    "Title",
    "CabinLetters",
    "FamilySize",
    "Embarked",
]

X_cols = [
    "Sex",
    *Cat_cols,
]

X = pd.get_dummies(all.loc[:, X_cols], columns=Cat_cols)

assert X.isna().sum().sum() == 0

#############
### SPLIT ###
#############

is_test = Y.isna()

train = Set(
    ID=Id.loc[~is_test],
    X=X.loc[~is_test],
    Y=Y.loc[~is_test],
)
test = Set(
    ID=Id.loc[is_test],
    X=X.loc[is_test],
)

###########
### RFC ###
###########

RFC_PARAMS = {
    "n_estimators": [500, 1000],
    "max_depth": [20],
    "min_samples_split": [12, 16],
    "min_samples_leaf": [4, 6, 8],
    "criterion": ["entropy"],
    "oob_score": [True],
    "random_state": [42],
}

KEYS, VALS = zip(*RFC_PARAMS.items())

prod_size = np.prod([len(val) for val in VALS], dtype=int)

cross_list = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for var_vals in bar(product(*VALS), total=prod_size, desc="Cross Validation"):
    loc_params = dict(zip(KEYS, var_vals))
    RFC = RandomForestClassifier(n_jobs=-1, **loc_params)
    cross_vals = cross_val_score(
        RFC,
        train.X,
        train.Y,
        cv=kf,
        n_jobs=-1,
    )
    mean = cross_vals.mean()
    std = cross_vals.std()
    cross_list.append(
        {
            **loc_params,
            "mean": mean,
            "std": std,
            "min": mean - std,
        }
    )

cross_df = pd.DataFrame(cross_list).sort_values("mean", ascending=False)

time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
cross_df.to_csv(OUT_DIR / f"cross_{time}.csv", index=False)

##########################
### FEATURE IMPORTANCE ###
##########################

last_file = sorted(OUT_DIR.glob("cross_*.csv"))[-1]
cross_df = pd.read_csv(last_file)

PERM_PARAMS = {
    "n_repeats": 12,
    "random_state": 42,
    "n_jobs": -1,
}

X_train, X_val, y_train, y_val = train_test_split(
    train.X,
    train.Y,
    stratify=train.Y,
    shuffle=True,
    random_state=42,
    test_size=0.2,
)

importance_list = []

for i in bar(range(min(6, len(cross_df))), desc="Importance"):
    params = cross_df.iloc[i].to_dict()
    for key in ["mean", "std", "min"]:
        params.pop(key, None)
    RFC = RandomForestClassifier(**params)
    RFC.fit(X_train, y_train)
    result = permutation_importance(
        RFC,
        X_val,
        y_val,
        **PERM_PARAMS,
    )
    importance_list.append(
        dict(
            zip(
                train.X.columns,
                result.importances_mean,
            )
        )
    )

importance_df = pd.DataFrame(importance_list).mean().sort_values(ascending=False)

##################
### PREDICTION ###
##################

params = cross_df.iloc[0].to_dict()
for key in ["mean", "std", "min"]:
    params.pop(key, None)

RFC = RandomForestClassifier(**params)
RFC.fit(train.X, train.Y)

prediction = RFC.predict(test.X)

submission = pd.DataFrame(
    {
        "PassengerId": test.ID,
        "Survived": prediction,
    },
    dtype=int,
)

message = f"RFC - {last_file.name}\nParameters:\n{cross_df.iloc[0]}\nImportance:\n{importance_df}\n"
if input(f"{message}\n\nSubmit [y/N]:") == "y":
    sub_file = OUT_DIR / f"submission_{time}.csv"
    submission.to_csv(sub_file, index=False)
    run(["kaggle", "competitions", "submit", "-c", "titanic", "-f", sub_file, "-m", message])
    sub_file.unlink()

###################
### GRAPH PRINT ###
###################

if False and input("Print trees [y/N]:") == "y":
    for i, estimator in enumerate(RFC.estimators_):
        file = OUT_DIR / "trees" / f"tree_{i:04d}.png"
        dot_data = export_graphviz(
            estimator,
            out_file=None,
            feature_names=X.columns,
            class_names=["0", "1"],
            filled=True,
            rounded=True,
            special_characters=True,
            impurity=True,
        )
        graphviz.Source(
            dot_data,
            filename=None,
        ).render(outfile=file, cleanup=True)

###########
### END ###
###########
