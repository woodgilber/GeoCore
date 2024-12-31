from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

from modeling.utils.dekart_map_config import application_set_config


def calculate_bootstrap_bounds(
    true_labels: list, probabilities: list, n_bins=10, n_bootstrap=1000, lower_bound=2.5, upper_bound=97.5
):
    """Calculate bootstrap confidence bounds for calibration curves.

    Args:
    true_labels (list or np.ndarray): True binary labels.
    probabilities (list or np.ndarray): Predicted probabilities.
    n_bins (int, optional): Number of bins to use for calibration curve. Defaults to 10.
    n_bootstrap (int, optional): Number of bootstrap samples to draw. Defaults to 1000.
    lower_bound(int,optional): Lower limit of bootstrap curve. Defaults to 2.5.
    upper_bound(int,optional): Upper limit of bootstrap curve. Defaults to 97.5.

    Returns:
    mean_predicted_value (np.ndarray): Mean predicted probabilities for each bin.
    fraction_of_positives (np.ndarray): Fraction of positives for each bin.
    lower_bounds (list): Lower confidence bounds for each bin.
    upper_bounds (list): Upper confidence bounds for each bin.
    """
    mean_predicted_value, fraction_of_positives = calibration_curve(true_labels, probabilities, n_bins=n_bins)

    lower_bounds = []
    upper_bounds = []
    for i in range(len(mean_predicted_value)):
        bin_fractions = []
        for _ in range(n_bootstrap):
            resampled_indices = np.array(resample(range(len(probabilities))))
            resampled_true = np.array(true_labels)[resampled_indices]
            resampled_pred = np.array(probabilities)[resampled_indices]
            fraction_of_positives_resampled, _ = calibration_curve(resampled_true, resampled_pred, n_bins=n_bins)
            if i < len(fraction_of_positives_resampled):
                bin_fractions.append(fraction_of_positives_resampled[i])
        lower_bounds.append(np.percentile(bin_fractions, lower_bound))
        upper_bounds.append(np.percentile(bin_fractions, upper_bound))

    return mean_predicted_value, fraction_of_positives, lower_bounds, upper_bounds


def plot_fold_metrics(
    folds_df: Dict[str, pd.DataFrame], folder: Optional[str], legend: Union[str, bool] = "auto"
) -> None:
    """Plot metrics for each fold

    Args:
        folds_df (Dict[str, pd.DataFrame]): dict index by the type of metric (val_roc, train_lift etc) and with values
            containing a dataframe
        folder (Optional[str]): destination folder to save the plots
    """
    fig, axs = plt.subplots(ncols=1, nrows=10, figsize=(7, 55))
    plt.style.use("ggplot")

    # plot ROC
    sb.lineplot(
        data=folds_df["train_roc_dfs"],
        x="FPR",
        y="TPR",
        hue="LABEL",
        estimator=None,
        ax=axs[0],
        legend=legend,
    ).set_title("TRAIN ROC")
    if legend:
        sb.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))

    sb.lineplot(
        data=folds_df["validation_roc_dfs"],
        x="FPR",
        y="TPR",
        hue="LABEL",
        estimator=None,
        ax=axs[1],
        legend=legend,
    ).set_title("VALID. ROC")
    if legend:
        sb.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))

    # plot PR
    sb.lineplot(
        data=folds_df["train_pr_dfs"],
        x="RECALL",
        y="PRECISION",
        hue="LABEL",
        estimator=None,
        ax=axs[2],
        legend=legend,
    ).set_title("TRAIN PR")
    if legend:
        sb.move_legend(axs[2], "upper left", bbox_to_anchor=(1, 1))

    sb.lineplot(
        data=folds_df["validation_pr_dfs"],
        x="RECALL",
        y="PRECISION",
        hue="LABEL",
        estimator=None,
        ax=axs[3],
        legend=legend,
    ).set_title("VALID. PR")
    if legend:
        sb.move_legend(axs[3], "upper left", bbox_to_anchor=(1, 1))

    # plot cumulative lift
    ax = sb.lineplot(
        data=folds_df["train_lift"],
        x="DECILE",
        y="CUM_LIFT",
        hue="LABEL",
        estimator=None,
        ax=axs[4],
        legend=legend,
    )
    ax.set_title("CUMULATIVE TRAIN LIFT")
    ax.axhline(y=1, xmin=0, xmax=1, color="k", linestyle="--")
    if legend:
        sb.move_legend(axs[4], "upper left", bbox_to_anchor=(1, 1))

    ax = sb.lineplot(
        data=folds_df["val_lift"],
        x="DECILE",
        y="CUM_LIFT",
        hue="LABEL",
        estimator=None,
        ax=axs[5],
        legend=legend,
    )
    ax.set_title("CUMULATIVE VALID LIFT")
    ax.axhline(y=1, xmin=0, xmax=1, color="k", linestyle="--")
    if legend:
        sb.move_legend(axs[5], "upper left", bbox_to_anchor=(1, 1))

    # plot lift
    ax = sb.lineplot(
        data=folds_df["train_lift"],
        x="DECILE",
        y="LIFT",
        hue="LABEL",
        estimator=None,
        ax=axs[6],
        legend=legend,
    )
    ax.set_title("DECILE-WISE TRAIN LIFT")
    ax.axhline(y=1, xmin=0, xmax=1, color="k", linestyle="--")
    if legend:
        sb.move_legend(axs[6], "upper left", bbox_to_anchor=(1, 1))

    ax = sb.lineplot(
        data=folds_df["val_lift"],
        x="DECILE",
        y="LIFT",
        hue="LABEL",
        estimator=None,
        ax=axs[7],
        legend=legend,
    )
    ax.set_title("DECILE-WISE VALID LIFT")
    ax.axhline(y=1, xmin=0, xmax=1, color="k", linestyle="--")
    if legend:
        sb.move_legend(axs[7], "upper left", bbox_to_anchor=(1, 1))

    # plot gain
    ax = sb.lineplot(
        data=folds_df["train_gain"],
        x="DECILE",
        y="RESP_PCT_MOD",
        hue="LABEL",
        estimator=None,
        ax=axs[8],
        legend=legend,
    )
    ax.set_title("TRAIN GAIN")
    axs[8].plot([0, 10], [0, 100], "k--")
    if legend:
        sb.move_legend(axs[8], "upper left", bbox_to_anchor=(1, 1))

    ax = sb.lineplot(
        data=folds_df["val_gain"],
        x="DECILE",
        y="RESP_PCT_MOD",
        hue="LABEL",
        estimator=None,
        ax=axs[9],
        legend=legend,
    )
    ax.set_title("VAL GAIN")
    axs[9].plot([0, 10], [0, 100], "k--")
    if legend:
        sb.move_legend(axs[9], "upper left", bbox_to_anchor=(1, 1))

    if folder:
        # save dataframe output
        folds_df["validation_roc_dfs"].to_parquet("./runs/validation_predictions.pq")
        # save figures
        fig.savefig(f"{folder}/train_roc_plots.png", bbox_inches="tight")
    else:
        plt.show()


def plot_calibration_metrics(fold_evaluation: list, folder: Optional[str]) -> None:
    """Plot calibration metrics for each fold, histograms, and overall performance with validation data

    Args:
        fold_evaluation (list): list containing evaluation metrics for each fold
        folder (Optional[str]): destination folder to save the plots
    """
    # Plots for calibration
    fig_calibration, ax_calibration = plt.subplots(figsize=(10, 10))
    fig_histogram, ax_histogram = plt.subplots(figsize=(10, 10))
    fig_single_calibration, ax_single_calibration = plt.subplots(figsize=(10, 10))
    num_folds = len(fold_evaluation)
    num_cols = 2
    num_rows = (num_folds + 1) // num_cols  # Ceiling division to get number of rows needed
    fig_reliability, axs_reliability = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 10 * num_rows))
    axs_reliability = axs_reliability.flatten()  # Flatten in case of single row/column subplots
    all_validation_probs = []
    all_validation_true = []
    for idx, fold_stat in enumerate(fold_evaluation):
        validation_probs = fold_stat["validation_probabilities"]
        validation_true = fold_stat["validation_true"]
        label = fold_stat["label"]
        # Collect all probabilities and true labels for the new single calibration curve
        all_validation_probs.extend(validation_probs)
        all_validation_true.extend(validation_true)
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(validation_true, validation_probs, n_bins=10)
        ax_calibration.plot(mean_predicted_value, fraction_of_positives, "s-", label=label)
        # Histogram
        ax_histogram.hist(validation_probs, bins=50, alpha=0.5, label=label, edgecolor="black")
        # Reliability diagram with confidence bounds
        axs_reliability[idx].plot(mean_predicted_value, fraction_of_positives, "s-", label=label)
        axs_reliability[idx].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        # Bootstrapping to calculate confidence intervals
        frac_pos, mean_predicted_value, lower_bounds, upper_bounds = calculate_bootstrap_bounds(
            validation_true, validation_probs, n_bins=10, n_bootstrap=1000
        )
        axs_reliability[idx].fill_between(mean_predicted_value, lower_bounds, upper_bounds, color="gray", alpha=0.2)
        axs_reliability[idx].set_xlabel("Mean predicted value")
        axs_reliability[idx].set_ylabel("Fraction of positives")
        axs_reliability[idx].legend()
        axs_reliability[idx].set_title(f"Reliability curve for {label}")

    # Hide any unused subplots
    for j in range(idx + 1, len(axs_reliability)):
        fig_reliability.delaxes(axs_reliability[j])

    # Finalize calibration plot
    ax_calibration.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax_calibration.set_xlabel("Mean predicted value")
    ax_calibration.set_ylabel("Fraction of positives")
    ax_calibration.legend()
    ax_calibration.set_title("Reliability Curves")
    if folder:
        fig_calibration.savefig(f"{folder}/calibration_plot.png")
    plt.show()

    # Finalize histogram plot
    ax_histogram.set_xlabel("Predicted probability")
    ax_histogram.set_ylabel("Count")
    ax_histogram.legend()
    ax_histogram.set_title("Histogram of Predicted Probabilities by Fold")
    if folder:
        fig_histogram.savefig(f"{folder}/prediction_histogram.png")
    plt.show()

    # Finalize reliability plots
    fig_reliability.tight_layout()
    if folder:
        fig_reliability.savefig(f"{folder}/reliability_plots.png")
    plt.show()

    # Plot single aggregated calibration curve
    all_fraction_of_positives_val, all_mean_predicted_value_val, lower_bounds_val, upper_bounds_val = (
        calculate_bootstrap_bounds(all_validation_true, all_validation_probs, n_bins=10, n_bootstrap=1000)
    )
    ax_single_calibration.plot(
        all_mean_predicted_value_val, all_fraction_of_positives_val, "s-", label="Validation (All Folds)"
    )

    # Plot the validation fill first
    ax_single_calibration.fill_between(
        all_mean_predicted_value_val, lower_bounds_val, upper_bounds_val, color="blue", alpha=0.2
    )
    ax_single_calibration.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax_single_calibration.set_xlabel("Mean predicted value")
    ax_single_calibration.set_ylabel("Fraction of positives")
    ax_single_calibration.legend()
    ax_single_calibration.set_title("Aggregated Calibration Plot (All Folds)")
    if folder:
        fig_single_calibration.savefig(f"{folder}/single_aggregated_calibration_plot.png")
    plt.show()


def plot_test_data(
    gain: pd.DataFrame,
    lift: pd.DataFrame,
    pr: pd.DataFrame,
    roc: pd.DataFrame,
    prefix: str,
    folder: Optional[str],
) -> None:
    fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(7, 35))
    plt.style.use("ggplot")

    # plot cumulative lift
    ax = sb.lineplot(
        data=lift,
        x="DECILE",
        y="CUM_LIFT",
        estimator=None,
        ax=axs[0],
    )
    ax.set_title("CUMULATIVE TEST LIFT")
    ax.axhline(y=1, xmin=0, xmax=1, color="k", linestyle="--")
    # sb.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))

    # plot lift
    ax = sb.lineplot(
        data=lift,
        x="DECILE",
        y="LIFT",
        estimator=None,
        ax=axs[1],
    )
    ax.set_title("DECILE-WISE TEST LIFT")
    ax.axhline(y=1, xmin=0, xmax=1, color="k", linestyle="--")
    # sb.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))

    # plot gain
    ax = sb.lineplot(
        data=gain,
        x="DECILE",
        y="RESP_PCT_MOD",
        estimator=None,
        ax=axs[2],
    )
    ax.set_title("TEST GAIN")
    axs[2].plot([0, 10], [0, 100], "k--")
    # sb.move_legend(axs[2], "upper left", bbox_to_anchor=(1, 1))

    # plot PR
    sb.lineplot(
        data=pr,
        x="RECALL",
        y="PRECISION",
        estimator=None,
        ax=axs[3],
    ).set_title("TEST PR")
    # sb.move_legend(axs[3], "upper left", bbox_to_anchor=(1, 1))

    # plot ROC
    sb.lineplot(
        data=roc,
        x="FPR",
        y="TPR",
        hue="LABEL",
        estimator=None,
        ax=axs[4],
    ).set_title("TEST ROC")

    if folder:
        fig.savefig(f"{folder}/{prefix}_diagnostic.png", bbox_inches="tight")
        fig.clear()
    else:
        plt.show()


def train_test_map(
    train: pd.DataFrame,
    ytrain: pd.Series,
    test: pd.DataFrame,
    destination_folder: str,
) -> None:
    """
    Saves H3 Indices as datasets for the train/test/application splits.
    Good for visual inspection of what is going on early in the run process.
    """
    from keplergl import KeplerGl

    plot_data = {}

    if train is not None:
        plot_data["train"] = pd.DataFrame({"train": train.index, "label": ytrain})

    if test is not None:
        plot_data["test"] = pd.DataFrame({"test": test.index})

    if len(plot_data.keys()) < 1:
        raise ValueError("Some data has to be provided for the plot! ")

    KeplerGl(height=400, data=plot_data).save_to_html(
        file_name=f"{destination_folder}/data_split_map.html", center_map=True
    )


def application_map(
    applied_outputs: pd.DataFrame,
    h3_col: str = "H3_BLOCKS",
    score_col: str = "SCORE",
    percentile_col: str = "PERCENTILE",
    destination_folder=str,
) -> None:
    """
    Saves H3 indices as datasets for the application outputs.
    Useful for visual inspection of predictions on a map.

    Args:
        applied_outputs (pd.DataFrame): DataFrame containing application outputs with H3 indices and scores.
        destination_folder (str): Folder to save the output map.
        h3_col (str, optional): Column name for H3 indices. Defaults to "H3_BLOCKS".
        score_col (str, optional): Column name for scores. Defaults to "SCORE".
        percentile_col (str, optional): Column name for percentiles. Defaults to "PERCENTILE".
    """
    from keplergl import KeplerGl

    # Ensure the required columns exist
    for col in [h3_col, score_col, percentile_col]:
        if col not in applied_outputs.columns:
            raise ValueError(f"Column '{col}' not found in the applied_outputs DataFrame.")

    # Prepare data for KeplerGl
    plot_data = {"application": applied_outputs[[h3_col, score_col, percentile_col]]}

    # Create the map and save it
    application_map = KeplerGl(height=400, data=plot_data, config=application_set_config())
    output_file = f"{destination_folder}/application_map.html"
    application_map.save_to_html(file_name=output_file, center_map=True)
