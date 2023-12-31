"""
relAI - Python library for reliability assessment of ML predictions.
"""

__version__ = "0.1.0"


import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from ReliabilityPackage.ReliabilityClasses import AE, ReliabilityDetector, DensityPrincipleDetector
from ReliabilityPackage.ReliabilityPrivateFunctions import _train_one_epoch, _compute_synpts_accuracy, _val_scores_diff_mse, \
    _contains_only_integers, _extract_values_proportionally
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# Functions


def create_autoencoder(layer_sizes):
    """
    Gets an autoencoder model with the specified sizes of the layers.

    This function gets an autoencoder model using the `AE` class, implemented as a PyTorch module, with the specified
    layers' sizes.
    The autoencoder is used for the implementation of the Density Principle.

    :param list layer_sizes: A list containing the number of nodes of each layer of the encoder (decoder built with
     symmetry).

    :return: An instance of the autoencoder model.
    :rtype: ReliabilityClasses.AE
    """
    ae = AE(layer_sizes)
    return ae


def train_autoencoder(ae, training_set, validation_set, batchsize, epochs=1000, optimizer=None,
                      loss_function=torch.nn.MSELoss(),
                      ):
    """
    Trains the autoencoder model using the provided training and validation sets.

    This function trains the autoencoder model using the provided training and validation sets.
    It performs multiple epochs of training, updating the model parameters based on the specified optimizer
    and loss function. The training progress is evaluated on the validation set after each epoch, and the resulting
    validation loss is shown in the image.

    :param torch.nn.Module ae: The autoencoder model to be trained.
    :param numpy.ndarray training_set: The training set.
    :param numpy.ndarray validation_set: The validation set.
    :param int batchsize: The batch size used for training.
    :param int epochs: The number of training epochs (default: 1000).
    :param torch.optim.Optimizer optimizer: The optimizer used for parameter updates.
        If None, an Adam optimizer with default parameters will be used (default: None).
    :param torch.nn.Module loss_function: The loss function used for training.
        If None, the mean squared error (MSE) loss function will be used (default: torch.nn.MSELoss()).

    :return: The trained autoencoder model.
    :rtype: torch.nn.Module
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4, weight_decay=1e-8)

    training_loader = DataLoader(dataset=training_set, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batchsize, shuffle=True)

    validation_loss = []
    epoch_number = 0

    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch_number + 1))
        ae.train(True)
        avg_loss = _train_one_epoch(epoch_number, training_set, training_loader, optimizer, loss_function, ae)
        ae.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs = vdata
            voutputs = ae(vinputs.float())
            vloss = loss_function(voutputs, vinputs.float())
            running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        validation_loss.append(avg_vloss.tolist())
        epoch_number += 1

    fig, ax = plt.subplots()
    plt.plot(validation_loss)
    plt.xlabel('epochs')
    plt.ylabel('Validation Loss')
    plt.title('Loss')
    plt.show()
    return ae


def get_and_train_autoencoder(training_set, validation_set, batchsize, layer_sizes=None, epochs=1000,
                                 optimizer=None, loss_function=torch.nn.MSELoss(),
                                 ):
    """
    Gets and trains an autoencoder model using the provided training and validation sets.

    This function gets an autoencoder model based on the specified layers' sizes and trains it using
    the provided training and validation sets. It performs multiple epochs of training, updating the model
    parameters based on the specified optimizer and loss function. The training progress is evaluated on
    the validation set after each epoch, and the resulting validation loss is shown in the image.

    :param numpy.ndarray training_set: The training set.
    :param numpy.ndarray validation_set: The validation set.
    :param int batchsize: The batch size used for training.
    :param list layer_sizes: A list containing the number of nodes of each layer of the encoder (decoder built with
     symmetry).
        If None, the default dimension of the encoder's layers is [dim_input, dim_input + 4, dim_input + 8,
        dim_input + 16, dim_input + 32]
    :param int epochs: The number of training epochs (default: 1000).
    :param torch.optim.Optimizer optimizer: The optimizer used for parameter updates.
        If None, an Adam optimizer with default parameters will be used (default: None).
    :param torch.nn.Module loss_function: The loss function used for training.
        If None, the mean squared error (MSE) loss function will be used (default: torch.nn.MSELoss()).

    :return: The trained autoencoder model.
    :rtype: torch.nn.Module
    """
    if layer_sizes is None:
        dim_input = training_set.shape[1]
        layer_sizes = [dim_input, dim_input + 4, dim_input + 8, dim_input + 16, dim_input + 32]
    ae = AE(layer_sizes)

    if optimizer is None:
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4, weight_decay=1e-8)

    training_loader = DataLoader(dataset=training_set, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batchsize, shuffle=True)

    validation_loss = []
    epoch_number = 0

    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch_number + 1))
        ae.train(True)
        avg_loss = _train_one_epoch(epoch_number, training_set, training_loader, optimizer, loss_function, ae)
        ae.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs = vdata
            voutputs = ae(vinputs.float())
            vloss = loss_function(voutputs, vinputs.float())
            running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        validation_loss.append(avg_vloss.tolist())
        epoch_number += 1

    fig, ax = plt.subplots()
    plt.plot(validation_loss)
    plt.xlabel('epochs')
    plt.ylabel('Validation Loss')
    plt.title('Loss')
    plt.show()
    return ae


def compute_dataset_avg_mse(ae, X):
    """
    Compute the average mean squared error (MSE) for a given autoencoder model and dataset.

    :param torch.nn.Module ae: The autoencoder model.
    :param numpy.ndarray X: The dataset of interest

    :return: The average MSE value for the reconstructed samples.
    :rtype: float
    """
    mse = []
    for i in range(len(X)):
        mse.append(mean_squared_error(X[i],  ae((torch.tensor(X[i, :])).float()).detach().numpy()))
    return np.mean(mse)


def generate_synthetic_points(predict_func, X_train, y_train, method='GN', k=5):
    """
    Generates synthetic points based on the specified method.

    This function generates synthetic points based on the method specified in "method".
    'GN': the synthetic points are generated from the training set by adding gaussian random noise, with different
    values of variance, to the continous variables,
          and by randomly extracting, proportionally to their frequencies, the values of binary and integer variables.

    :param numpy.ndarray X_train: The training set with shape (n_samples, n_features).
    :param str method: The method used to generate synthetic points (default: 'GN').
        Currently, only the 'GN' (Gaussian Noise) method is supported.

    :return: The synthetic points generated with the specified method.
    :rtype: numpy.ndarray
    """
    allowed_methods = ['GN']
    if method not in allowed_methods:
        raise ValueError(f"Invalid value for method. Allowed values are {allowed_methods}.")

    if method == 'GN':
        noisy_data = X_train.copy()
        for j in range(2, 7):
            noisy_data_temp = X_train.copy()
            for i in range(X_train.shape[1]):
                if _contains_only_integers(X_train[:, i]):
                    noisy_data_temp[:, i] = _extract_values_proportionally(X_train[:, i])
                else:
                    noise = np.random.normal(0, j * 0.1, size=X_train.shape[0])
                    noisy_data_temp[:, i] += noise

            noisy_data = np.concatenate((noisy_data, noisy_data_temp))

    acc_syn = _compute_synpts_accuracy(predict_func, noisy_data, X_train, y_train, k)

    return noisy_data, acc_syn


def perc_mse_threshold(ae, validation_set, perc=95):
    """
    Computes the MSE threshold as a percentile of the MSE of the validation set.

    This function computes the MSE threshold as a percentile of the MSE of the validation set
    using an autoencoder model. It calculates the MSE for each sample in the validation set
    and returns the specified percentile threshold.

    :param torch.nn.Module ae: The autoencoder model.
    :param numpy.ndarray validation_set: The validation set with shape (n_samples, n_features).
    :param int perc: The percentile threshold to compute (default: 95).

    :return: The MSE threshold as the specified percentile of the MSE of the validation set.
    :rtype: float
    """
    val_projections = []
    mse_val = []
    for i in range(len(validation_set)):
        val_projections.append(ae((torch.tensor(validation_set[i, :])).float()))
        mse_val.append(mean_squared_error(validation_set[i], val_projections[i].detach().numpy()))

    return np.percentile(mse_val, perc)


def mse_threshold_plot(ae, X_val, y_val, predict_func, metric='f1_score'):
    """
    Generates a plot of performance metrics based on different MSE thresholds (selected as percentiles of the MSE of
    the validation set).

    This function generates a plot of performance metrics based on different Mean Squared Error (MSE) thresholds.
    It computes the number (and percentage) of the reliable and unreliable samples obtained with each threshold, and
    different performance metrics using the `val_scores_diff_mse` function.
    The plot shows the performance metric selected ('metric') (e.g., balanced_accuracy, precision, recall, F1-score,
    MCC, or Brier score) for reliable and unreliable samples at different MSE thresholds, and their number and
    percentage. A slider allows to move the x-axis.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like X_val: The validation dataset.
    :param array-like y_val: The validation labels.
    :param callable predict_func: The predict function of the classifier.
    :param str metric: The performance metric to display on the plot. Available options: 'balanced_accuracy',
    'precision', 'recall', 'f1_score', 'mcc', 'brier_score'. Default is 'f1_score'.

    :return: A Plotly Figure object representing the MSE threshold plot.
    :rtype: go.Figure
    """
    allowed_metrics = ['balanced_accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'brier_score']
    if metric not in allowed_metrics:
        raise ValueError(f"Invalid value for metric. Allowed values are {allowed_metrics}.")

    if metric == 'balanced_accuracy':
        metrx = 0
    elif metric == 'precision':
        metrx = 1
    elif metric == 'recall':
        metrx = 2
    elif metric == 'f1_score':
        metrx = 3
    elif metric == 'mcc':
        metrx = 4
    elif metric == 'brier_score':
        metrx = 5

    mse_threshold_list, rel_scores, unrel_scores, num_unrel, perc_unrel = _val_scores_diff_mse(ae, X_val, y_val,
                                                                                              predict_func)
    perc_rel = ['{:.2f}'.format((1 - perc_unrel[i]) * 100) for i in range(len(perc_unrel))]
    perc_unrel = ['{:.2f}'.format(perc_unrel[i] * 100) for i in range(len(perc_unrel))]
    num_rel = [X_val.shape[0] - i for i in num_unrel]
    percentiles = [i for i in range(2, 100)]
    y_rel = [lst[metrx] for lst in rel_scores]
    y_unrel = [lst[metrx] for lst in unrel_scores]

    htxt_rel = [str('{:.2f}'.format(perf)) for perf in y_rel]
    htxt_unrel = [str('{:.2f}'.format(perf)) for perf in y_unrel]

    # Create figure
    fig = go.Figure()
    fig.update_yaxes(range=[min(y_unrel + y_rel), max(y_unrel + y_rel)])
    # fig.update_xaxes(tickformat=".2e")

    for step in range(len(percentiles)):
        fig.add_trace(
            go.Scatter(
                x=percentiles[:step + 2],
                y=y_rel[:step + 2],
                visible=False,
                name='Reliable ' + metric,
                mode='lines',
                line=dict(color='lightgreen'),
                customdata=[[perc, num] for perc, num in zip(perc_rel[:step + 2], num_rel[:step + 2])],
                hovertemplate='%{y:.3f}<br>Reliable samples: %{customdata[1]} (%{customdata[0]}%)',
            )
        )
    for step in range(len(percentiles)):
        fig.add_trace(
            go.Scatter(
                x=percentiles[:step + 2],
                y=y_unrel[:step + 2],
                visible=False,
                name='Unreliable ' + metric,
                mode='lines',
                line=dict(color='salmon'),
                customdata=[[perc, num] for perc, num in zip(perc_unrel[:step + 2], num_unrel[:step + 2])],
                hovertemplate='%{y:.3f}<br>Unreliable samples: %{customdata[1]} (%{customdata[0]}%)',
            )
        )
    # Create and add slider
    steps = []
    for i in range(int(len(fig.data) / 2)):
        step = dict(
            method="update",
            label=str(percentiles[i]) + "°-P",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": str(metric) + " variation on the validation set at different values of the MSE threshold"},
                  {"x": [percentiles[:i + 2]]},  # Update x-axis data
                  ],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][int(len(fig.data) / 2) + i] = True
        steps.append(step)

    # Make last traces visible
    fig.data[len(percentiles) - 1].visible = True
    fig.data[2 * len(percentiles) - 1].visible = True
    sliders = [dict(
        active=len(percentiles) - 1,
        # currentvalue={"prefix": "MSE x-limit: "},
        pad={"t": 20},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        hovermode="x unified",
        xaxis=dict(
            # tickformat='.2e',
            title='MSE threshold'
        ),
        title=str(metric) + " variation on the validation set at different values of the MSE threshold"
    )

    return fig


def mse_threshold_barplot(ae, X_val, y_val, predict_func):
    """
    Generates a bar plot of performance metrics based on different MSE thresholds.

    This function generates a bar plot of performance metrics based on different Mean Squared Error (MSE) thresholds
    (selected as percentiles of the MSE of the validation set).
    It computes different scores for the reliable and unreliable samples obtained, and the number and percentage of
    unreliable samples, using the `val_scores_diff_mse` function.
    The bar plot shows the percentage of unreliable samples, as well as various performance metrics (e.g.,
    balanced_accuracy, precision, recall, F1-score,  MCC, or Brier score) for reliable and unreliable samples at each
    MSE threshold.
    A slider allows selecting the MSE threshold and updating the plot accordingly.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like X_val: The validation dataset.
    :param array-like y_val: The validation labels.
    :param callable predict_func: The predict function of the classifier.

    :return: A Plotly Figure object representing the MSE threshold bar plot.
    :rtype: go.Figure
    """
    mse_threshold_list, rel_scores, unrel_scores, num_unrel, perc_unrel = _val_scores_diff_mse(ae, X_val, y_val,
                                                                                              predict_func)
    percentiles = [i for i in range(2, 100)]

    # Creazione del layout con legenda
    hovertext = ["% Unreliable Validation Set",
                 "Balanced Accuracy Reliable set", "Balanced Accuracy Unreliable set",
                 "Precision Reliable set", "Precision Unreliable set",
                 "Recall Reliable set", "Recall Unreliable set",
                 "F1 Score Reliable set", "F1 Score Unreliable set",
                 "MCC Reliable set", "MCC Unreliable set",
                 "Brier Score Reliable set", "Brier Score Unreliable set", ]

    # Create figure
    fig = go.Figure()
    fig.update_yaxes(range=[0, 1])

    colors = ['black',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon',
              'lightgreen', 'salmon']

    # Add traces, one for each slider step
    for step in range(len(mse_threshold_list)):
        ybar = [perc_unrel[step],
                rel_scores[step][0], unrel_scores[step][0],
                rel_scores[step][1], unrel_scores[step][1],
                rel_scores[step][2], unrel_scores[step][2],
                rel_scores[step][3], unrel_scores[step][3],
                rel_scores[step][4], unrel_scores[step][4],
                rel_scores[step][5], unrel_scores[step][5],
                ]
        format_ybar = ["{:.3f}".format(val) for val in ybar]
        fig.add_trace(
            go.Bar(
                x=['% UR',
                   'R-Bal Accuracy', 'UR-Bal Accuracy',
                   'R-Precision', 'UR-Precision',
                   'R-Recall', 'UR-Recall',
                   'R-f1', 'UR-f1',
                   'R-MCC', 'UR-MCC',
                   'R-brier score', 'UR-brier score'
                   ],
                y=ybar,
                visible=False,
                marker=dict(color=colors),
                name='',
                width=0.8,
                text=format_ybar,
                showlegend=False,
                hovertext=hovertext,
                hoverinfo='text'
            )
        )

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            label=str(i + 2) + "°-P",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "MSE threshold: " + str('{:.4e}'.format(mse_threshold_list[i])) + ": " + str(
                      i + 1) + "°-percentile" +
                            " --- # Unreliable: " + str(num_unrel[i]) +
                            " (" + str('{:.2f}'.format(perc_unrel[i] * 100)) + "%)"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Make 10th trace visible
    fig.data[49].visible = True

    sliders = [dict(
        active=49,
        currentvalue={"prefix": "MSE threshold: "},
        pad={"t": 20},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="MSE threshold: " + str('{:.4e}'.format(mse_threshold_list[49])) + ": " + str(50) + "°-percentile" +
              " --- # Unreliable: " + str(num_unrel[49]) +
              " (" + str('{:.2f}'.format(perc_unrel[49] * 100)) + "%)"
    )

    return fig


def density_predictor(ae, mse_thresh):
    """
    Creates a DensityPrinciplePredictor object for a given autoencoder and MSE threshold.

    This function creates a DensityPrinciplePredictor object using the specified autoencoder and MSE threshold.
    The DensityPrinciplePredictor is a density-based predictor that assigns reliability scores to samples based on their
    reconstruction error (MSE) compared to the MSE threshold.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param float mse_thresh: The MSE threshold used for assigning reliability scores.

    :return: A DensityPrinciplePredictor object.
    :rtype: DensityPrincipleDetector
    """
    DP = DensityPrincipleDetector(ae, mse_thresh)
    return DP


def create_reliability_detector(ae, syn_pts, acc_syn, mse_thresh, acc_thresh, proxy_model='MLP'):
    """
    Gets a ReliabilityPredictor object for a given autoencoder, synthetic points, accuracy of the synthetic points,
    MSE threshold, and accuracy threshold.

    This function gets a ReliabilityPredictor object using the specified autoencoder, synthetic points, accuracy of
    the synthetic points, MSE threshold, and accuracy threshold. The ReliabilityPredictor assigns the density
    reliability of samples based on their reconstruction error (MSE), with respect to the MSE threshold, while assigns
    the local fit reliability based on the prediction of a model ('proxy_model'), trained on the synthetic points
    labelled as "local-fit" reliable/unreliable according to their associated accuracy with respect to the accuracy
    threshold.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like syn_pts: The synthetic points used for training the "local-fit" reliability predictor.
    :param array-like acc_syn: The accuracy scores corresponding to the synthetic points.
    :param float mse_thresh: The MSE threshold used for assigning the density reliability scores.
    :param float acc_thresh: The accuracy threshold used for assigning the "local-fit" reliability scores.
    :param str proxy_model: The type of proxy model used for training the "local-fit"reliability predictor.
        Available options: 'MLP', 'tree'. Default is 'MLP' (Multi-Layer Perceptron).

    :return: A ReliabilityPackage object.
    :rtype: ReliabilityDetector
    """
    allowed_proxy_model = ['MLP', 'tree']
    if proxy_model not in allowed_proxy_model:
        raise ValueError(f"Invalid value for proxy_model. Allowed values are {allowed_proxy_model}.")
    y_syn_pts = []
    for i in range(len(acc_syn)):
        y_syn_pts.append(1) if acc_syn[i] >= acc_thresh else y_syn_pts.append(0)

    if proxy_model == 'MLP':
        clf = MLPClassifier(activation="tanh", random_state=42, max_iter=1000).fit(syn_pts, y_syn_pts)
    elif proxy_model == 'tree':
        clf = tree.DecisionTreeClassifier(random_state=42).fit(syn_pts, y_syn_pts)

    RP = ReliabilityDetector(ae, clf, mse_thresh)

    return RP


def compute_dataset_reliability(RD, X, mode='total'):
    """
    Computes the reliability of the samples in a dataset

    This function computes the density/local-fit/total reliability of the samples in the X dataset, based on the mode
    specified, with the ReliabilityPackage RD
    :param ReliabilityDetector RD: A ReliabilityPackage object.
    :param array-like X: the specified dataset
    :param str mode: the type of reliability to compute; Available options: 'density', 'local-fit', 'total'. Default is
    'total'
    :return: a numpy 1-D array containing the reliability of each sample (1 for reliable, 0 for unreliable)
    :rtype: numpy.ndarray
    """
    if mode == 'total':
        return np.asarray([RD.compute_total_reliability(x) for x in X])
    elif mode == 'density':
        return np.asarray([RD.compute_density_reliability(x) for x in X])
    elif mode == 'local-fit':
        return np.asarray([RD.compute_localfit_reliability(x) for x in X])

