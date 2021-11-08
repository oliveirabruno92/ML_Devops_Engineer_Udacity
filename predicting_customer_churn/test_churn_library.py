import os
import sys
import glob
import logging
import pytest
import joblib
import churn_library
from config import Config

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name='dataset_raw')
def dataset_raw_():
    try:
        dataset_raw = churn_library.import_data(Config.DATA_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    return dataset_raw


# @pytest.fixture(name='dataset_sample')
# def dataset_sample_(dataset_raw):
#     try:
#         dataset_sample = dataset_raw.sample(frac=0.1, random_state=42)
#         logging.info("Testing dataset sample: SUCCESS")
#     except BaseException as err:
#         logging.error("Dataset sample: Failed to sample from dataset raw")
#         raise err
#
#     return dataset_sample


@pytest.fixture(name='dataset_with_target')
def dataset_with_target_(dataset_raw):

    try:
        dataset_with_target = churn_library.encode_target(dataset_raw)
        logging.info("Testing encode target: SUCCESS")
    except KeyError as err:
        logging.error("Encode target creation: Attrition Flag not in dataset")
        raise err

    return dataset_with_target


@pytest.fixture(name='dataset_encoded')
def dataset_encoded_(dataset_with_target):
    """
    encoded dataframe fixture - returns the encoded dataframe on some specific column
    """
    try:
        dataset_encoded = churn_library.encoder_helper(
            data=dataset_with_target,
            category_lst=Config.CATEGORICAL_COLS,
            response=Config.RESPONSE
        )
        logging.info("Encoded dataset fixture creation: SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoded dataset fixture creation: Not existent column to encode"
        )
        raise err

    return dataset_encoded


@pytest.fixture(name='split_data')
def split_data_(dataset_encoded):
    """
    feature sequences fixtures - returns 4 series containing features sequences
    """
    try:
        train_features, test_features, train_target, test_target = churn_library.perform_feature_engineering(
            data=dataset_encoded,
            response=Config.RESPONSE
        )
        logging.info("Split data fixture creation: SUCCESS")
    except BaseException:
        logging.error(
            "Split data fixture creation: Sequences length mismatch"
        )
        raise
    return train_features, test_features, train_target, test_target


def test_import(dataset_raw):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        assert dataset_raw.shape[0] > 0
        assert dataset_raw.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(dataset_encoded):
    """
    test perform eda function
    """
    churn_library.perform_eda(dataset_encoded)
    for column in Config.COLS_TO_PLOT:
        try:
            with open(f"images/eda/{column}.jpg", 'r'):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing perform_eda: generated images missing")
            raise err


def test_encode_target(dataset_with_target):

    try:
        assert dataset_with_target.shape[0] > 0
        assert dataset_with_target.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have rows and columns")
        raise err

    try:
        assert Config.RESPONSE in dataset_with_target
    except AssertionError as err:
        logging.error(
            "Testing encoder_target: The dataframe doesn't have the target column")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")


def test_encoder_helper(dataset_encoded):
    """
    test encoder helper
    """
    try:
        assert dataset_encoded.shape[0] > 0
        assert dataset_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have rows and columns")
        raise err

    try:
        for column in Config.CATEGORICAL_COLS:
            assert column in dataset_encoded
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")

    return dataset_encoded


def test_perform_feature_engineering(split_data):
    """
    test perform_feature_engineering
    """
    try:
        train_features = split_data[0]
        test_features = split_data[1]
        train_target = split_data[2]
        test_target = split_data[3]
        assert len(train_features) == len(train_target)
        assert len(test_features) == len(test_target)
        logging.info("Testing feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing feature_engineering: Sequences length mismatch")
        raise err

    return split_data


def test_train_models(split_data):
    """
    test train_models
    """
    train_features = split_data[0]
    test_features = split_data[1]
    train_target = split_data[2]
    test_target = split_data[3]
    churn_library.train_models(
        train_features,
        test_features,
        train_target,
        test_target
    )
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The files weren't found")
        raise err

    for image_name in Config.RESULTS_IMG:
        try:
            with open(f"images/results/{image_name}.jpg", 'r'):
                logging.info("Testing testing_models (report generation): SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing testing_models (report generation): generated images missing")
            raise err


if __name__ == "__main__":
    for directory in ["logs", "images/eda", "images/results", "./models"]:
        for file in glob.glob(f"{directory}/*"):
            os.remove(file)
    sys.exit(pytest.main(["-s"]))
