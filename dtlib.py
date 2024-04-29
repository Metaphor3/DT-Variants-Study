import time
import pickle
import os
import json
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd

import functions
import evaluate as dt_eval
import training

# ------------------------


def fit(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    target_label: str = "Decision",
    validation_df: Optional[pd.DataFrame] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Build (a) decision tree model(s)

    Args:
            df (pandas data frame): Training data frame.

            config (dictionary): training configuration. e.g.

                    config = {
                            'algorithm' (string): ID3, 'C4.5, CART, CHAID or Regression
                    }

            target_label (str): target label for supervised learning.
                Default is Decision at the end of dataframe.

            validation_df (pandas data frame): validation data frame
                if nothing is passed to validation data frame, then the function validates
                built trees for training data frame

            silent (bool): set this to True if you do not want to see
                any informative logs

    Returns:
            chefboost model
    """

    # ------------------------

    process_id = os.getpid()

    # ------------------------
    # rename target column name
    if target_label != "Decision":
        # TODO: what if another column name is Decision?
        df = df.rename(columns={target_label: "Decision"})

    # if target is not the last column
    if df.columns[-1] != "Decision":
        if "Decision" in df.columns:
            new_column_order = df.columns.drop(
                "Decision").tolist() + ["Decision"]
            df = df[new_column_order]
        else:
            raise ValueError("Please set the target_label")

    # ------------------------

    base_df = df.copy()

    # ------------------------

    target_label = df.columns[len(df.columns) - 1]

    # ------------------------
    # handle NaN values

    nan_values = []

    for column in df.columns:
        if df[column].dtypes != "object":
            min_value = df[column].min()
            idx = df[df[column].isna()].index

            nan_value = []
            nan_value.append(column)

            if idx.shape[0] > 0:
                df.loc[idx, column] = min_value - 1
                nan_value.append(min_value - 1)
            else:
                nan_value.append(None)

            nan_values.append(nan_value)

    # ------------------------

    # initialize params and folders
    config = functions.initializeParams(config)
    functions.initializeFolders()

    # ------------------------

    algorithm = config["algorithm"]

    valid_algorithms = ["ID3", "C4.5", "CART", "CHAID", "Regression"]

    if algorithm not in valid_algorithms:
        raise ValueError(
            "Invalid algorithm passed. You passed ",
            algorithm,
            " but valid algorithms are ",
            valid_algorithms,
        )

    # ------------------------

    num_of_columns = df.shape[1]

    if algorithm == "Regression":
        if df["Decision"].dtypes == "object":
            raise ValueError(
                "Regression trees cannot be applied for nominal target values!"
                "You can either change the algorithm or data set."
            )

    if (
        df["Decision"].dtypes != "object"
    ):  # this must be regression tree even if it is not mentioned in algorithm
        algorithm = "Regression"
        config["algorithm"] = "Regression"

    # -------------------------

    # initialize a dictionary. this is going to be used to check features numeric or nominal.
    # numeric features should be transformed to nominal values based on scales.
    dataset_features = {}

    header = "def findDecision(obj): #"

    num_of_columns = df.shape[1] - 1
    for i in range(0, num_of_columns):
        column_name = df.columns[i]
        dataset_features[column_name] = df[column_name].dtypes
        header += f"obj[{str(i)}]: {column_name}"

        if i != num_of_columns - 1:
            header = header + ", "

    header = header + "\n"

    # ------------------------

    begin = time.time()

    trees = []
    alphas = []

    # regular decision tree building
    root = 1
    file = "outputs/rules/rules.py"
    functions.createFile(file, header)

    trees = training.buildDecisionTree(
        df,
        root=root,
        file=file,
        config=config,
        dataset_features=dataset_features,
        parent_level=0,
        leaf_id=0,
        parents="root",
        validation_df=validation_df,
        main_process_id=process_id,
    )

    obj = {"trees": trees, "alphas": alphas,
           "config": config, "nan_values": nan_values}

    # -----------------------------------------

    # train set accuracy
    df = base_df.copy()
    trainset_evaluation = evaluate(obj, df, task="train", silent=silent)
    obj["evaluation"] = {"train": trainset_evaluation}

    # validation set accuracy
    if isinstance(validation_df, pd.DataFrame):
        validationset_evaluation = evaluate(
            obj, validation_df, task="validation", silent=silent)
        obj["evaluation"]["validation"] = validationset_evaluation

    return obj

    # -----------------------------------------


def predict(model: dict, param: list) -> Union[str, int, float]:
    """
    Predict the target label of given features from a pre-trained model
    Args:
        model (built chefboost model): pre-trained model which is the output
            of fit function
        param (list): pass input features as python list
            e.g. chef.predict(model, param = ['Sunny', 'Hot', 'High', 'Weak'])
    Returns:
            prediction
    """

    trees = model["trees"]
    config = model["config"]

    alphas = []
    if "alphas" in model:
        alphas = model["alphas"]

    nan_values = []
    if "nan_values" in model:
        nan_values = model["nan_values"]

    # -----------------------
    # handle missing values

    column_index = 0
    for column in nan_values:
        column_name = column[0]
        missing_value = column[1]

        if pd.isna(missing_value) != True:
            if pd.isna(param[column_index]):
                param[column_index] = missing_value

        column_index = column_index + 1

    # -----------------------

    classification = False
    prediction = 0
    prediction_classes = []

    # -----------------------

    if len(trees) > 1:  # bagging or boosting
        index = 0
        for tree in trees:
            custom_prediction = tree.findDecision(param)

            if custom_prediction != None:
                if not isinstance(custom_prediction, str):  # regression
                    prediction += custom_prediction
                else:
                    classification = True
                    prediction_classes.append(custom_prediction)
            index = index + 1

    else:  # regular decision tree
        tree = trees[0]
        prediction = tree.findDecision(param)

    if classification == False:
        return prediction
    else:
        # classification
        # e.g. random forest
        # get predictions made by different trees
        predictions = np.array(prediction_classes)

        # find the most frequent prediction
        (values, counts) = np.unique(predictions, return_counts=True)
        idx = np.argmax(counts)
        prediction = values[idx]

        return prediction


def save_model(base_model: dict, file_name: str = "model.pkl") -> None:
    """
    Save pre-trained model on file system
    Args:
            base_model (dict): pre-trained model which is the output
                of the fit function
            file_name (string): target file name as exact path.
    """

    model = base_model.copy()

    # modules cannot be saved. Save its reference instead.
    module_names = []
    for tree in model["trees"]:
        module_names.append(tree.__name__)

    model["trees"] = module_names

    with open(f"outputs/rules/{file_name}", "wb") as f:
        pickle.dump(model, f)


def load_model(file_name: str = "model.pkl") -> dict:
    """
    Load the save pre-trained model from file system
    Args:
            file_name (str): exact path of the target saved model
    Returns:
            built model (dict)
    """

    with open("outputs/rules/" + file_name, "rb") as f:
        model = pickle.load(f)

    # restore modules from its references
    modules = []
    for model_name in model["trees"]:
        module = functions.restoreTree(model_name)
        modules.append(module)

    model["trees"] = modules

    return model


def restoreTree(module_name) -> Any:
    """
    Load built model from set of decision rules
    Args:
        module_name (str): e.g. outputs/rules/rules to restore outputs/rules/rules.py
    Returns:
            built model (dict)
    """

    return functions.restoreTree(module_name)


def feature_importance(rules: Union[str, list], silent: bool = False) -> pd.DataFrame:
    """
    Show the feature importance values of a built model
    Args:
        rules (str or list): e.g. decision_rules = "outputs/rules/rules.py"
            or this could be retrieved from built model as shown below.

            ```python
            decision_rules = []
            for tree in model["trees"]:
               rule = .__dict__["__spec__"].origin
               decision_rules.append(rule)
            ```
        silent (bool): set this to True if you do want to see
            any informative logs.
    Returns:
            feature importance (pd.DataFrame)
    """

    if not isinstance(rules, list):
        rules = [rules]

    # -----------------------------

    dfs = []

    for rule in rules:
        with open(rule, "r", encoding="UTF-8") as file:
            lines = file.readlines()

        pivot = {}
        rules = []

        # initialize feature importances
        line_idx = 0
        for line in lines:
            if line_idx == 0:
                feature_explainer_list = line.split("#")[1].split(", ")
                for feature_explainer in feature_explainer_list:
                    feature = feature_explainer.split(
                        ": ")[1].replace("\n", "")
                    pivot[feature] = 0
            else:
                if "# " in line:
                    rule = line.strip().split("# ")[1]
                    rules.append(json.loads(rule))

            line_idx = line_idx + 1

        feature_names = list(pivot.keys())

        for feature in feature_names:
            for rule in rules:
                if rule["feature"] == feature:
                    score = rule["metric_value"] * rule["instances"]
                    current_depth = rule["depth"]

                    child_scores = 0
                    # find child node importances
                    for child_rule in rules:
                        if child_rule["depth"] == current_depth + 1:
                            child_score = child_rule["metric_value"] * \
                                child_rule["instances"]

                            child_scores = child_scores + child_score

                    score = score - child_scores

                    pivot[feature] = pivot[feature] + score

        # normalize feature importance

        total_score = 0
        for feature, score in pivot.items():
            total_score = total_score + score

        for feature, score in pivot.items():
            pivot[feature] = round(pivot[feature] / total_score, 4)

        instances = []
        for feature, score in pivot.items():
            instance = []
            instance.append(feature)
            instance.append(score)
            instances.append(instance)

        df = pd.DataFrame(instances, columns=["feature", "final_importance"])
        df = df.sort_values(by=["final_importance"], ascending=False)

        if len(rules) == 1:
            return df
        else:
            dfs.append(df)

    if len(rules) > 1:
        hf = pd.DataFrame(feature_names, columns=["feature"])
        hf["importance"] = 0

        for df in dfs:
            hf = hf.merge(df, on=["feature"], how="left")
            hf["importance"] = hf["importance"] + hf["final_importance"]
            hf = hf.drop(columns=["final_importance"])

        # ------------------------
        # normalize
        hf["importance"] = hf["importance"] / hf["importance"].sum()
        hf = hf.sort_values(by=["importance"], ascending=False)

        return hf


def evaluate(
    model: dict,
    df: pd.DataFrame,
    target_label: str = "Decision",
    task: str = "test",
    silent: bool = False,
) -> dict:
    """
    Evaluate the performance of a built model on a data set
    Args:
        model (dict): built model which is the output of fit function
        df (pandas data frame): data frame you would like to evaluate
        target_label (str): target label
        task (string): set this to train, validation or test
        silent (bool): set this to True if you do not want to see
            any informative logs
    Returns:
        evaluation results (dict)
    """

    # --------------------------

    if target_label != "Decision":
        df = df.rename(columns={target_label: "Decision"})

    # if target is not the last column
    if df.columns[-1] != "Decision":
        new_column_order = df.columns.drop("Decision").tolist() + ["Decision"]
        df = df[new_column_order]

    # --------------------------

    functions.bulk_prediction(df, model)

    return dt_eval.evaluate(df, task=task, silent=silent)
