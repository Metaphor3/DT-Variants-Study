import pathlib
import os
import sys
from os import path
from types import ModuleType
from typing import Optional, Union
import numpy as np
import pandas as pd
import dtlib as dt
from module import load_module


def bulk_prediction(df: pd.DataFrame, model: dict) -> None:
    """
    Perform a bulk prediction on given dataframe
    Args:
        df (pd.DataFrame): input data frame
        model (dict): built model
    Returns:
        None
    """
    predictions = []
    for _, instance in df.iterrows():
        features = instance.values[0:-1]
        prediction = dt.predict(model, features)
        predictions.append(prediction)

    df["Prediction"] = predictions


def restoreTree(module_name: str) -> ModuleType:
    """
    Restores a built tree
    """
    return load_module(module_name)


def softmax(w: list) -> np.ndarray:
    """
    Softmax function
    Args:
        w (list): probabilities
    Returns:
        result (numpy.ndarray): softmax of inputs
    """
    e = np.exp(np.array(w, dtype=np.float32))
    dist = e / np.sum(e)
    return dist


def sign(x: Union[int, float]) -> int:
    """
    Sign function
    Args:
        x (int or float): input
    Returns
        result (int) 1 for positive inputs, -1 for negative
            inputs, 0 for neutral input
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def formatRule(root: int) -> str:
    """
    Format a rule in the output file (tree)
    Args:
        root (int): degree of current rule
    Returns:
        formatted rule (str)
    """
    resp = ""

    for _ in range(0, root):
        resp = resp + "   "

    return resp


def storeRule(file: str, content: str) -> None:
    """
    Store a custom rule
    Args:
        file (str): target file
        content (str): content to store
    Returns:
        None
    """
    with open(file, "a+", encoding="UTF-8") as f:
        f.writelines(content)
        f.writelines("\n")


def createFile(file: str, content: str) -> None:
    """
    Create a file with given content
    Args:
        file (str): target file
        content (str): content to store
    Returns
        None
    """
    with open(file, "w", encoding="UTF-8") as f:
        f.write(content)


def initializeFolders() -> None:
    """
    Initialize required folders
    """
    sys.path.append("..")
    pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs/data").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs/rules").mkdir(parents=True, exist_ok=True)

    # -----------------------------------

    # clear existing rules in outputs/

    outputs_path = os.getcwd() + os.path.sep + "outputs" + os.path.sep

    try:
        if path.exists(outputs_path + "data"):
            for file in os.listdir(outputs_path + "data"):
                os.remove(outputs_path + "data" + os.path.sep + file)

        if path.exists(outputs_path + "rules"):
            for file in os.listdir(outputs_path + "rules"):
                if (
                    ".py" in file
                    or ".json" in file
                    or ".txt" in file
                    or ".pkl" in file
                    or ".csv" in file
                ):
                    os.remove(outputs_path + "rules" + os.path.sep + file)
    except Exception as err:
        pass

    # ------------------------------------


def initializeParams(config: Optional[dict] = None) -> dict:
    """
    Arrange a chefboost configuration
    Args:
        config (dict): initial configuration
    Returns:
        config (dict): final configuration
    """
    if config == None:
        config = {}

    algorithm = "ID3"
    max_depth = 5

    for key, value in config.items():
        if key == "algorithm":
            algorithm = value
        # ---------------------------------
        elif key == "max_depth":
            max_depth = value
        # ---------------------------------

    config["algorithm"] = algorithm
    config["max_depth"] = max_depth

    return config
