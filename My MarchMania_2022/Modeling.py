import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import warnings
import logging


warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("requests").setLevel(logging.CRITICAL)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


##########
def train_model(X, X_test, y, params, folds, eval_type='regression'):
    trained = np.zeros(X.shape[0])
    predictions = np.zeros(X_test.shape[0])
    scores_at_folds = []
    for fold_n, (train_idx, test_idx) in enumerate(folds.split(X, y)):
        trn_data = lgb.Dataset(X[train_idx], y[train_idx])
        val_data = lgb.Dataset(X[test_idx], y[test_idx])
        model = lgb.train(params, trn_data, num_boost_round=20000, valid_sets=[trn_data, val_data],
                               verbose_eval=-1)

        # clf = LGBMClassifier()
        # clf.fit(X_train, y_train)
        trained[test_idx] = model.predict(X[test_idx], num_iteration=model.best_iteration)
        predictions += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

        # print(predictions)
        scores_at_folds.append(mean_squared_error(trained[test_idx], y[test_idx]) ** 0.5)

    # Visualize
    plt.rcParams["figure.figsize"] = (12, 22)

    lgb.plot_importance(model).set_yticklabels(['Season','Team1ID','Team2ID', 'T1PPG','T1FG%',
                                                'T1FT%','T1Ast','T1Def','T1ORating','T1DRating','T2PPG',
                                                'T2FG%','T2FT%','T2Ast','T2Def','T2ORating','T2DRating',
                                                'T1_Seed','T2_Seed','SeedDiff','T1_Rank','T2_Rank','RankDiff'])

    plt.show()

    return predictions, scores_at_folds
##########

stat_cols = [col for col in train.columns if col not in ['T1Win']]

lgb_params = {'min_data_in_leaf': 30,  # minimum data in a single leaf (columns)
              'objective': 'binary',  # Saw this on others kaggle and it improves accuracy
              'learning_rate': 0.01,  # improvised to return the most accuracy
              "boosting": "gbdt",  # Gradient Boosted Decision Tree provided most accuracy
              "feature_fraction": 0.9,  # adds a little randomness in results
              "bagging_freq": 5,
              "bagging_fraction": 0.9,  # adds a little randomness in results
              "metric": 'binary_logloss',  # to calculate my own log loss
              'early_stopping_round': 300}  # to make the program run faster

folds = KFold(n_splits=5, shuffle=True, random_state=1940)
X_train = train[stat_cols].values
X_test = test[stat_cols].values
y_train = train['T1Win'].values
predictions, scores = train_model(X_train, X_test, y_train,
                                  params=lgb_params, folds=folds,
                                  eval_type='regression')

# predictions
sub_df = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage2/MSampleSubmissionStage2.csv')
sub_df['Pred'] = predictions
sub_df.to_csv('submission.csv', index=False)
# print(sub_df.to_string())
print('Log Loss:', round(np.mean(scores), 6))
