from scoring.wodeutil.nlp.metrics.eval_constants import *
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor, \
    StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from tensorflow import keras

regr_dict = {
    regr_forest: RandomForestRegressor(n_jobs=-1, random_state=random_state),
    regr_dtree: DecisionTreeRegressor(random_state=random_state),
    regr_lin: LinearRegression(n_jobs=-1),
    regr_mlp: MLPRegressor(random_state=random_state),
    regr_lin_svr: LinearSVR(epsilon=1.5, random_state=random_state),
    regr_ridge: Ridge(alpha=1, solver='cholesky'),
    # regr_lasso: Lasso(alpha=0.1),
    # regr_elastic_net: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
    regr_adaboost: AdaBoostRegressor(DecisionTreeRegressor(random_state=random_state, max_depth=10), n_estimators=200,
                                     learning_rate=0.5, random_state=random_state),
    regr_bagging: BaggingRegressor(DecisionTreeRegressor(random_state=random_state, max_depth=10), n_estimators=100,
                                   max_samples=1.0, bootstrap=True,
                                   n_jobs=-1),
    regr_voting: VotingRegressor(
        estimators=[(regr_ridge, Ridge(alpha=1, solver='cholesky')), (regr_forest, RandomForestRegressor(n_jobs=-1)),
                    (regr_mlp, MLPRegressor())], n_jobs=-1),
    regr_grad_boost: GradientBoostingRegressor(n_estimators=150, random_state=random_state),
    regr_stacking: StackingRegressor(
        estimators=[(regr_ridge, Ridge(alpha=1, solver='cholesky')), (regr_lin_svr, LinearSVR(epsilon=1.5)),
                    (regr_mlp, MLPRegressor())],
        final_estimator=RandomForestRegressor(n_jobs=-1, n_estimators=10, random_state=11), n_jobs=-1)
}
regr_dict_regularized = {
    regr_forest: RandomForestRegressor(n_jobs=-1, random_state=random_state),
    regr_dtree: DecisionTreeRegressor(random_state=random_state, max_depth=3),
    regr_lin: LinearRegression(n_jobs=-1),
    regr_mlp: MLPRegressor(random_state=random_state),
    regr_lin_svr: LinearSVR(epsilon=1.5, random_state=random_state),
    regr_ridge: Ridge(alpha=1, solver='cholesky'),
    # regr_lasso: Lasso(alpha=0.1),
    # regr_elastic_net: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
    regr_adaboost: AdaBoostRegressor(DecisionTreeRegressor(random_state=random_state, max_depth=3), n_estimators=200,
                                     learning_rate=0.5, random_state=random_state),
    regr_bagging: BaggingRegressor(DecisionTreeRegressor(random_state=random_state, max_depth=3), n_estimators=100,
                                   max_samples=1.0, bootstrap=True,
                                   n_jobs=-1),
    regr_voting: VotingRegressor(
        estimators=[(regr_ridge, Ridge(alpha=1, solver='cholesky')), (regr_forest, RandomForestRegressor(n_jobs=-1)),
                    (regr_mlp, MLPRegressor())], n_jobs=-1),
    regr_grad_boost: GradientBoostingRegressor(n_estimators=150, random_state=random_state),
    regr_stacking: StackingRegressor(
        estimators=[(regr_ridge, Ridge(alpha=1, solver='cholesky')), (regr_lin_svr, LinearSVR(epsilon=1.5)),
                    (regr_mlp, MLPRegressor())],
        final_estimator=RandomForestRegressor(n_jobs=-1, n_estimators=10, random_state=11), n_jobs=-1)
}


def get_model_name(regr_type, prefix=None, suffix=None, is_keras_nn=False):
    if regr_type:
        model_name = regr_type
        if prefix:
            model_name = f"{prefix}_{model_name}"
        if suffix:
            model_name = f"{model_name}_{suffix}"
        model_name = model_name + ".h5" if is_keras_nn else model_name + ".joblib"
        return model_name


def get_model_path(regr_type, prefix=None, suffix=None, is_keras_nn=False, model_dir=None):
    if model_dir:
        chkpt_dir = model_dir
    else:
        chkpt_dir = "coherence"
    model_path = os.path.join(chkpt_dir, get_model_name(regr_type=regr_type, prefix=prefix, suffix=suffix,
                                                        is_keras_nn=is_keras_nn))
    return model_path


def save_regress_model(regressor, regr_type, prefix=None, suffix=None, is_keras_nn=False, model_dir=None):
    model_path = get_model_path(regr_type, prefix=prefix, suffix=suffix, is_keras_nn=is_keras_nn, model_dir=model_dir)
    if is_keras_nn:
        regressor.save(model_path)
    else:
        dump(regressor, model_path)
    print(f".............. saved model to {model_path} ...................")


def load_regress_model(regr_type, prefix=None, suffix=None, is_keras_nn=False, model_dir=None):
    model_path = get_model_path(regr_type, prefix=prefix, suffix=suffix, is_keras_nn=is_keras_nn, model_dir=model_dir)
    if is_keras_nn:
        model = keras.models.load_model(model_path)
    else:
        model = load(model_path)
    return model


def build_keras_regressor(n_hidden=2, n_neurons=32, lr=0.001, input_shape=[20], activation="tanh", out_dim=1,
                          kernel_initializer=None, kernel_regularizer=None, metrics='accuracy'):
    """
    regression model for predicting a quality score
    """
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        if kernel_initializer and kernel_regularizer:
            model.add(keras.layers.Dense(n_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer))
        else:
            model.add(keras.layers.Dense(n_neurons, activation=activation))
    model.add(keras.layers.Dense(out_dim))
    optimizer = keras.optimizers.RMSprop(lr=lr)
    model.compile(loss="mse", optimizer=optimizer, metrics=metrics)
    return model
