from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def decile_table(y_true: np.array, y_prob: np.array, change_deciles: int = 10, round_decimal: int = 3) -> pd.DataFrame:
    """Generates the Decile Table from labels and probabilities

    The Decile Table is creared by first sorting the customers by their predicted
    probabilities, in decreasing order from highest (closest to one) to
    lowest (closest to zero). Splitting the customers into equally sized segments,
    we create groups containing the same numbers of customers, for example, 10 decile
    groups each containing 10% of the customer base.

    Args:
        y_true (np.array, shape (n_samples)):
            Ground truth (correct/actual) target values.

        y_prob (np.array, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier/algorithm.

        change_deciles (int, optional): The number of partitions for creating the table
            can be changed. Defaults to '10' for deciles.

        round_decimal (int, optional): The decimal precision till which the result is
            needed. Defaults to '3'.

    Returns:
        dt: The dataframe dt (decile-table) with the deciles and related information.
    """
    df = pd.DataFrame()
    df["y_true"] = y_true
    df["y_prob"] = y_prob
    # df['decile']=pd.qcut(df['y_prob'], 10, labels=list(np.arange(10,0,-1)))
    # ValueError: Bin edges must be unique

    df.sort_values("y_prob", ascending=False, inplace=True)
    df["decile"] = np.linspace(1, change_deciles + 1, len(df), False, dtype=int)

    # dt abbreviation for decile_table
    dt = (
        df.groupby("decile")
        .apply(
            lambda x: pd.Series(
                [
                    np.min(x["y_prob"]),
                    np.max(x["y_prob"]),
                    np.mean(x["y_prob"]),
                    np.size(x["y_prob"]),
                    np.sum(x["y_true"]),
                    np.size(x["y_true"][x["y_true"] == 0]),
                ],
                index=(
                    [
                        "prob_min",
                        "prob_max",
                        "prob_avg",
                        "cnt_cust",
                        "cnt_resp",
                        "cnt_non_resp",
                    ]
                ),
            )
        )
        .reset_index()
    )

    dt["prob_min"] = dt["prob_min"].round(round_decimal)
    dt["prob_max"] = dt["prob_max"].round(round_decimal)
    dt["prob_avg"] = round(dt["prob_avg"], round_decimal)
    # dt=dt.sort_values(by='decile',ascending=False).reset_index(drop=True)

    tmp = df[["y_true"]].sort_values("y_true", ascending=False)
    tmp["decile"] = np.linspace(1, change_deciles + 1, len(tmp), False, dtype=int)

    dt["cnt_resp_rndm"] = np.sum(df["y_true"]) / change_deciles
    dt["cnt_resp_wiz"] = tmp.groupby("decile", as_index=False)["y_true"].sum()["y_true"]

    dt["resp_rate"] = round(dt["cnt_resp"] * 100 / dt["cnt_cust"], round_decimal)
    dt["cum_cust"] = np.cumsum(dt["cnt_cust"])
    dt["cum_resp"] = np.cumsum(dt["cnt_resp"])
    dt["cum_resp_wiz"] = np.cumsum(dt["cnt_resp_wiz"])
    dt["cum_non_resp"] = np.cumsum(dt["cnt_non_resp"])
    dt["cum_cust_pct"] = round(dt["cum_cust"] * 100 / np.sum(dt["cnt_cust"]), round_decimal)
    dt["cum_resp_pct"] = round(dt["cum_resp"] * 100 / np.sum(dt["cnt_resp"]), round_decimal)
    dt["cum_resp_pct_wiz"] = round(dt["cum_resp_wiz"] * 100 / np.sum(dt["cnt_resp_wiz"]), round_decimal)
    dt["cum_non_resp_pct"] = round(dt["cum_non_resp"] * 100 / np.sum(dt["cnt_non_resp"]), round_decimal)
    dt["KS"] = round(dt["cum_resp_pct"] - dt["cum_non_resp_pct"], round_decimal)
    dt["lift"] = round(dt["cum_resp_pct"] / dt["cum_cust_pct"], round_decimal)

    return dt


def get_pr(preds: np.array, ytrue: np.array, label: str) -> Tuple[pd.DataFrame, float]:
    """
    Calculate PR curve and AUC
    """
    precision, recall, thresholds = precision_recall_curve(ytrue, preds)
    auc_ = auc(recall, precision)

    ret_df = pd.DataFrame(
        {
            "PRECISION": precision,
            "RECALL": recall,
            "THRESHOLDS": np.append(thresholds, 1),
        }
    )
    ret_df["LABEL"] = label + "; AUC = " + str(round(auc_, 3))
    return ret_df, auc_


def get_roc(preds: np.array, ytrue: np.array, label: str) -> Tuple[pd.DataFrame, float]:
    """
    Calculate ROC curve and AUC
    """
    fpr, tpr, thresholds = roc_curve(ytrue, preds)
    auc_ = auc(fpr, tpr)

    df = pd.DataFrame(
        {
            "TPR": tpr,
            "FPR": fpr,
            "THRESHOLDS": thresholds,
            "LABEL": label + "; AUC = " + str(round(auc_, 3)),
        }
    )

    return df, auc_
