import os, sys
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import auto_causality
import dowhy

from causaldata import thornton_hiv
from auto_causality import AutoCausality
from auto_causality.datasets import synth_ihdp
from auto_causality.data_utils import preprocess_dataset
from auto_causality.scoring import ate, group_ate

if __name__ == "__main__":
    

    thl = thornton_hiv.load_pandas()
    df_thl = thl.data
    df_thl = df_thl.dropna()
    # categ = lambda y: 0 if y == 0 else round(y) + 1
    # df_thl["treatment"] = df_thl.apply(lambda row: categ(row.tinc), axis=1)
    # df_thl["treatment"] = df_thl.apply(lambda row: row.treatment - 1 if row.treatment == 4 else row.treatment, axis=1)
    
    treatment = 'tinc'
    targets='got'
    data_df, features_X, features_W = preprocess_dataset(df_thl, treatment, targets)
    outcome = targets #[0]
    train_df, test_df = train_test_split(data_df, test_size=0.2)
    
    ac = AutoCausality(
        time_budget=100,
        estimator_list=["CausalForestDML"], 
        # estimator_list=[
        #         # "Dummy",
        #         "SparseLinearDML",
        #         "ForestDRLearner",
        #         "TransformedOutcome",
        #         "CausalForestDML",
        #         ".LinearDML",
        #         "DomainAdaptationLearner",
        #         "SLearner",
        #         # "XLearner",
        #         # "TLearner",
        #         # "Ortho",
        #     ],
        metric="norm_erupt", 
        verbose=3, # 3
        components_verbose=2,# 2
        components_time_budget=10,
    )


    # run autocausality
    ac.fit(train_df, treatment, outcome, features_W, features_X)

    # return best estimator
    print(f"Best estimator: {ac.best_estimator}")
    # config of best estimator:
    print(f"best config: {ac.best_config}")
    # best score:
    print(f"best score: {ac.best_score}")