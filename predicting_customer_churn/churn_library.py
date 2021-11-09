"""
Author: Bruno
Date Created: 2021-11-08

This is the churn_library.py module that encapsulates churn model eda, training and predictions.
Artifact produced will be in images, logs and models folders.
"""

import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from config import Config


logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    """
    logging.info('Reading data from path %s.', pth)
    data = pd.read_csv(pth)

    return data


def encode_target(data):
    """
    Function that encoder Attrition_Flag column into Churn column

    input:
            data: pandas dataframe

    output:
            data: pandas dataframe
    """
    logging.info('Encoding target column into 0 or 1.')
    data['Churn'] = np.where(
        data['Attrition_Flag'] == "Existing Customer", 0, 1
    )

    return data


def perform_eda(data):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    logging.info('Making plots for EDA.')

    for column_name in Config.COLS_TO_PLOT:
        plt.figure(figsize=(20, 10))
        if column_name == "Churn":
            data['Churn'].hist()
        elif column_name == "Customer_Age":
            data['Customer_Age'].hist()
        elif column_name == "Marital_Status":
            data['Marital_Status'].value_counts("normalize").plot(kind="bar")
        elif column_name == "Total_Trans":
            sns.displot(data['Total_Trans_Ct'])
        elif column_name == "Heatmap":
            sns.heatmap(data.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig(f"images/eda/{column_name}.jpg")
        plt.close()


def encoder_helper(data, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            data: pandas dataframe with new columns for
    """
    for cat_cols in category_lst:
        data[cat_cols + '_Churn'] = data[cat_cols].map(
            data.groupby(cat_cols)[response].mean()
        )

    return data


def perform_feature_engineering(data, response):
    """
    input:
              data: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    logging.info('Peforming feature engineering.')
    data = encoder_helper(data, Config.CATEGORICAL_COLS, response)

    logging.info('Splitting data into train and tests sets.')
    features = data[Config.COLS_TO_KEEP]
    target = data[response]

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=42
    )

    return train_features, test_features, train_target, test_target


def classification_report_image(
        train_target,
        test_target,
        train_preds_lr,
        train_preds_rf,
        test_preds_lr,
        test_preds_rf
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            train_target: training response values
            test_target:  test response values
            train_preds_lr: training predictions from logistic regression
            train_preds_rf: training predictions from random forest
            test_preds_lr: test predictions from logistic regression
            test_preds_rf: test predictions from random forest

    output:
             None
    """

    model_results = {
        'Logistic Regression': {
            'train': classification_report(train_target, train_preds_lr),
            'test': classification_report(test_target, test_preds_lr)
        },
        'Random Forest': {
            'train': classification_report(train_target, train_preds_rf),
            'test': classification_report(test_target, test_preds_rf)
        }
    }

    for model_name, results in model_results.items():

        logging.info('Saving {} results on image folder.'.format(model_name))
        plt.rc('figure', figsize=(5, 5))
        plt.text(
            x=0.01,
            y=1.25,
            s='{} Train'.format(model_name),
            fontdict={'fontsize': 10},
            fontproperties='monospace'
        )
        plt.text(
            x=0.01,
            y=0.05,
            s='{}'.format(results['train']),
            fontdict={'fontsize': 10},
            fontproperties='monospace'
        )
        plt.text(
            x=0.01,
            y=0.6,
            s='{} Test'.format(model_name),
            fontdict={'fontsize': 10},
            fontproperties='monospace'
        )
        plt.text(
            x=0.01,
            y=0.7,
            s='{}'.format(results['test']),
            fontdict={'fontsize': 10},
            fontproperties='monospace'
        )
        plt.axis('off')
        if model_name == 'Logistic Regression':
            plt.savefig('./images/results/logistic_results.jpg')
        elif model_name == 'Random Forest':
            plt.savefig('./images/results/rf_results.jpg')
        plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importance in pth
    input:
            model: model object containing feature_importance_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.bar(range(x_data.shape[1]), feature_importance[indices])
    plt.ylabel("Importance")
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(f"images/{output_pth}/feature_importance.jpg")
    plt.close()


def roc_curve_plot(models_to_compare, test_features, test_target, output_pth):

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    for model in models_to_compare:
        plot_roc_curve(model, test_features, test_target, ax=ax, alpha=0.8)
    plt.savefig(f"images/{output_pth}/roc_curve_results.jpg")
    plt.show()


def grid_search(
        estimator,
        train_features,
        train_target,
        param_grid=Config.PARAM_GRID
):
    """
    Function that executes grid search of a model on param grid
    :param estimator: model to use
    :param train_features: train features
    :param train_target: train target
    :param param_grid: grid of paramaters to use on grid search
    :return: Best estimator
    """
    model = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
    model.fit(train_features, train_target)

    return model.best_estimator_


def saving_model(model, save_pth):
    """
    Saves model into path

    input:
            model: Sklearn model
            save_pth: Path to save the model
    output:
            None
    """
    joblib.dump(model, save_pth)


def train_models(train_features, test_features, train_target, test_target):
    """
    train, store model results: images + scores, and store models
    input:
              train_features: train features
              test_features: test_features
              train_target: train target
              test_target: test target
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    logging.info('Training random forest.')
    cv_rfc = grid_search(
        estimator=rfc,
        train_features=train_features,
        train_target=train_target
    )

    logging.info('Training logistic regression.')
    lrc.fit(train_features, train_target)

    logging.info(
        'Making predictions with random forest on training and testing data.'
    )
    train_preds_rf = cv_rfc.predict(train_features)
    test_preds_rf = cv_rfc.predict(test_features)

    logging.info(
        'Making predictions with logistic regression on training and testing data.'
    )
    train_preds_lr = lrc.predict(train_features)
    test_preds_lr = lrc.predict(test_features)

    classification_report_image(
        train_target,
        test_target,
        train_preds_lr,
        train_preds_rf,
        test_preds_lr,
        test_preds_rf
    )

    feature_importance_plot(
        model=cv_rfc,
        x_data=test_features,
        output_pth="results"
    )
    roc_curve_plot(
        models_to_compare=[lrc, cv_rfc],
        test_features=test_features,
        test_target=test_target,
        output_pth='results'
    )

    logging.info('Saving Random Forest model into models folder.')
    saving_model(cv_rfc, './models/rfc_model.pkl')

    logging.info('Saving Logistic Regression model into models folder.')
    saving_model(lrc, './models/logistic_model.pkl')


def main():
    """
    Main function of the module
    :return: None
    """
    data = import_data(Config.DATA_PATH)
    data = encode_target(data)
    perform_eda(data)
    train_features, test_features, train_target, test_target = perform_feature_engineering(
        data,
        response='Churn'
    )
    train_models(
        train_features,
        test_features,
        train_target,
        test_target
    )


if __name__ == '__main__':
    main()

