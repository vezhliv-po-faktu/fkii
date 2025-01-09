import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
)
from sklearn.datasets import load_breast_cancer, load_diabetes # датасеты
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor # классы алгоритмов классификации и регрессии
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


storage = "sqlite:///lroptuna.db"


# Классификация 
def objective_classification(trial):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    clf = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


# Регрессия 
def objective_regression(trial):
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    reg = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    return mean_squared_error(y_test, y_pred) #, squared=False)


# Оптимизация гиперпараметров классификации 
for sampler in [TPESampler(), RandomSampler()]:
    if isinstance(sampler, RandomSampler):
        pruner = MedianPruner()
    elif isinstance(sampler, TPESampler):
        pruner = HyperbandPruner()
    study_name = f"classification_{sampler.__class__.__name__}_{pruner.__class__.__name__}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective_classification, n_trials=10)

    print(f"Study {study_name} best params: {study.best_params}")
    print(f"Study {study_name} best value: {study.best_value}")

    plot_optimization_history(study).show()
    plot_param_importances(study).show()


# Оптимизация гиперпараметров регрессии 
for sampler in [TPESampler(), RandomSampler()]:
    if isinstance(sampler, RandomSampler):
        pruner = MedianPruner()
    elif isinstance(sampler, TPESampler):
        pruner = HyperbandPruner()
    study_name = f"regression_{sampler.__class__.__name__}_{pruner.__class__.__name__}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(objective_regression, n_trials=10)

    print(f"Study {study_name} best params: {study.best_params}")
    print(f"Study {study_name} best value: {study.best_value}")

    plot_optimization_history(study).show()
    plot_param_importances(study).show()
