import pytest
import subprocess
import sys
import pandas as pd

import auto_causality.data_utils
import auto_causality.utils
from auto_causality import datasets


def check_header(df: pd.DataFrame, n_covariates: int):
    """checks if header of dataset is in right format

    Args:
        df (pd.DataFrame): dataset for causal inference, with columns for treatment, outcome and covariates
        n_covariates (int): number of covariates in dataset
    """

    cols = list(df.columns)
    assert "treatment" in cols
    assert "y_factual" in cols
    for i in range(1, n_covariates + 1):
        assert "x" + str(i) in cols
    xcols = [c for c in cols if "x" in c]
    assert len(list(df[xcols].columns)) == n_covariates


def check_preprocessor(df: pd.DataFrame, treatment="treatment", targets=["y_factual"]):
    """checks if dataset can be preprocessed (dummy encoding of vars etc...)

    Args:
        df (pd.DataFrame): dataset for causal inference, with cols for treatment, outcome and covariates
    """
    x = auto_causality.data_utils.preprocess_dataset(df, treatment, targets)
    assert x is not None


class TestDatasets:
    def test_nhefs(self):
        # check if dataset can be imported:
        data = datasets.nhefs()
        # check if header variables follow naming convention
        check_header(data, n_covariates=9)
        # verify that preprocessing works
        check_preprocessor(data)

    def test_lalonde_nsw(self):
        # check if dataset can be imported:
        data = datasets.lalonde_nsw()
        # check if header variables follow naming convention
        check_header(data, n_covariates=8)
        # verify that preprocessing works
        check_preprocessor(data)

    def test_ihdp(self):
        # check if dataset can be imported:
        data = datasets.synth_ihdp()
        # check if header variables follow naming convention
        check_header(data, n_covariates=25)
        # verify that preprocessing works
        check_preprocessor(data)

    def test_acic(self):
        # test defaults:
        # check if dataset can be imported:
        data = datasets.synth_acic()
        # check if header variables follow naming convention
        check_header(data, n_covariates=58)
        # verify that preprocessing works
        check_preprocessor(data)
        # test all conditions:
        for cond in range(1, 11):
            # check if dataset can be imported:
            data = datasets.synth_acic(condition=cond)
            # check if header variables follow naming convention
            check_header(data, n_covariates=58)
            # verify that preprocessing works
            check_preprocessor(data)
        # sanity check
        # check if dataset can be imported:
        data = datasets.synth_acic(condition=12)
        assert data is None

    @pytest.mark.skip(
        reason="""this test occassionally fails to download the dataset from google drive,
        due to gdrive's virus check for big files.
        This however doesn't indicate problems with the code on our end"""
    )
    def test_amazon(self):
        # test error handling:
        try:
            import gdown  # noqa F401
        except ImportError:
            # check if dataset can be imported:
            data = datasets.amazon_reviews()
            assert data is None
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        finally:
            # test if data can be imported and is in right format:
            for rating in ["pos", "neg"]:
                data = datasets.amazon_reviews(rating=rating)
                # check if header variables follow naming convention
                check_header(data, n_covariates=300)
                # verify that preprocessing works
                check_preprocessor(data)


if __name__ == "__main__":
    pytest.main([__file__])
