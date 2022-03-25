from typing import Any, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(
    G: Any,
    base_mean: float = 0,
    base_var: float = 0.3,
    mean: float = 0,
    var: float = 1,
    SIZE: int = 10000,
    err_type: str = "normal",
    perturb: list = [],
    sigmoid: bool = True,
    expon: float = 1.1,
) -> pd.DataFrame:
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean, var, SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == "gumbel":
                g.append(np.random.gumbel(base_mean, base_var, SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var, SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]:  # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1 / 1 + np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
    g = np.swapaxes(g, 0, 1)

    return pd.DataFrame(g, columns=list(map(str, list_vertex)))


def load_adult():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df = pd.read_csv(path, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    for row in replace:
        df = df.replace(row, range(len(row)))

    continuous = ["education-num", "hours-per-week", "age"]
    for col in continuous:
        df[col] = df[col] / df[col].max()

    df = df[["race", "age", "sex", "native-country", "marital-status", "education-num", "occupation", "hours-per-week", "workclass", "relationship", "label"]]
    categorical = ["race", "native-country", "marital-status", "occupation", "workclass", "relationship"]
    df = pd.concat([pd.get_dummies(df[col]) if col in categorical else df[col] for col in df.columns], axis=1)
    df = df.values

    feature_num = {0: 5, 1: 1, 2:1, 3: 41, 4: 7, 5: 1, 6:14, 7: 1, 8:7, 9:6, 10:1}
    
    # X = df[:, :-1].astype(np.uint32)
    # y = df[:, -1].astype(np.uint8)

    dag_seed = [[0, 4], [1, 4], [2, 4], [3, 4], [0, 5], [4, 5], [1, 5], [2, 5], [3, 5], [0, 6], [1, 6], [2, 6], [4, 6], [5, 6], [0, 7], [1, 7], [4, 7], [3, 7], [2, 7], [5, 7], [1, 8], [4, 8], [2, 8], [5, 8], [3, 8], [4, 9], [5, 9], [1, 9], [2, 9], [3, 9], [0, 10], [1, 10], [2, 10], [3, 10], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]]
    return df, dag_seed, feature_num

def load_adult_binary():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df = pd.read_csv(path, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]
    
    df = df.dropna()
    
    continuous = ["education-num", "hours-per-week", "age"]
    for col in continuous:
        df[col] = df[col] / df[col].max()

    df = df[["race", "age", "sex", "native-country", "marital-status", "education-num", "occupation", "hours-per-week", "workclass", "relationship", "label"]]

    df['native-country'] = df['native-country'].map(lambda x: int(x == "United-States"))

    df['label'] = df['label'].map({'<=50K': 1., '>50K': 0.})

    df['marital-status'] = df['marital-status'].replace(['Divorced', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'Single')
    df['marital-status'] = df['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse'], 'Couple')
    df['marital-status'] = df['marital-status'].map({'Couple': 0., 'Single': 1.})
    
    df["race"] = df["race"].map({"White": 1, "Asian-Pac-Islander": 0, "Amer-Indian-Eskimo": 0, "Other":0 , "Black":0})

    df["sex"] = df["sex"].map({"Female": 1, "Male": 0})


    df['occupation'] = df['occupation'].replace(["Exec-managerial", "Prof-specialty", "Sales", "Tech-support", "Protective-serv"], 'High Wage')
    df['occupation'] = df['occupation'].replace([
            "Other-service",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Armed-Forces",
        ], 'Low Wage')
    df['occupation'] = df["occupation"].map({'High Wage': 1., "Low Wage": 0.})

    df['relationship'] = df['relationship'].replace(["Husband", "Wife"], 'High Wage')
    df['relationship'] = df['relationship'].replace(['Not-in-family', 'Own-child', 'Unmarried',
       'Other-relative'], 'Low Wage')
    df['relationship'] = df["relationship"].map({'High Wage': 1., "Low Wage": 0.})

    df['workclass'] = df['workclass'].replace(['Self-emp-not-inc', "Without-pay", "Never-worked"], 'No Org')
    df['workclass'] = df['workclass'].replace(['State-gov', 'Private', 'Federal-gov',
       'Local-gov', 'Self-emp-inc'], 'Org')
    df['workclass'] = df["workclass"].map({'Org': 1., "No Org": 0.})
    
    df = df.values.astype(np.uint32)
    
    dag_seed = [[0, 4], [1, 4], [2, 4], [3, 4], [0, 5], [4, 5], [1, 5], [2, 5], [3, 5], [0, 6], [1, 6], [2, 6], [4, 6], [5, 6], [0, 7], [1, 7], [4, 7], [3, 7], [2, 7], [5, 7], [1, 8], [4, 8], [2, 8], [5, 8], [3, 8], [4, 9], [5, 9], [1, 9], [2, 9], [3, 9], [0, 10], [1, 10], [2, 10], [3, 10], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]]

    feature_num = {0: 1, 1: 1, 2:1, 3: 1, 4: 1, 5: 1, 6:1, 7: 1, 8:1, 9:1, 10:1}

    return df, dag_seed, feature_num
