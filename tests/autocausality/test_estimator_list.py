import pytest
import pandas as pd

from auto_causality import AutoCausality
from auto_causality.params import SimpleParamService


class TestEstimatorListGenerator:
    """tests if estimator list is correctly generated"""

    def test_auto_list(self):
        """tests if "auto" setting yields all available estimators"""
        cfg = SimpleParamService(propensity_model=None, outcome_model=None)
        auto_estimators = cfg.estimator_names_from_patterns("auto")
        # verify that returned estimator list includes all available estimators
        assert len(auto_estimators) == 6

    def test_all_list(self):
        """tests if "auto" setting yields all available estimators"""
        cfg = SimpleParamService(
            propensity_model=None, outcome_model=None, include_experimental=True
        )
        all_estimators = cfg.estimator_names_from_patterns("all", data_rows=1)
        # verify that returned estimator list includes all available estimators
        assert len(all_estimators) == len(cfg._configs())

        cfg = SimpleParamService(
            propensity_model=None, outcome_model=None, include_experimental=True
        )
        all_estimators = cfg.estimator_names_from_patterns("all", data_rows=10000)
        # verify that returned estimator list includes all available estimators
        assert len(all_estimators) == len(cfg._configs()) - 2

    def test_substring_group(self):
        """tests if substring match to group of estimators works"""
        cfg = SimpleParamService(propensity_model=None, outcome_model=None)

        estimator_list = cfg.estimator_names_from_patterns(["dml"])
        available_estimators = [e for e in cfg._configs().keys() if "dml" in e]
        # verify that returned estimator list includes all available estimators
        assert all(e in available_estimators for e in estimator_list)

        # or all econml models:
        estimator_list = cfg.estimator_names_from_patterns(["econml"])
        available_estimators = [e for e in cfg._configs().keys() if "econml" in e]
        # verify that returned estimator list includes all available estimators
        assert all(e in available_estimators for e in estimator_list)

    def test_substring_single(self):
        """tests if substring match to single estimators works"""
        cfg = SimpleParamService(propensity_model=None, outcome_model=None)
        estimator_list = cfg.estimator_names_from_patterns(["DomainAdaptationLearner"])
        assert estimator_list == [
            "backdoor.econml.metalearners.DomainAdaptationLearner"
        ]

    def test_checkduplicates(self):
        """tests if duplicates are removed"""
        cfg = SimpleParamService(propensity_model=None, outcome_model=None)
        estimator_list = cfg.estimator_names_from_patterns(
            [
                "DomainAdaptationLearner",
                "DomainAdaptationLearner",
                "DomainAdaptationLearner",
            ]
        )
        assert len(estimator_list) == 1

    def test_invalid_choice(self):
        """tests if invalid choices are handled correctly"""
        # this should raise a ValueError
        # unavailable estimators:

        cfg = SimpleParamService(propensity_model=None, outcome_model=None)

        with pytest.raises(ValueError):
            cfg.estimator_names_from_patterns(["linear_regression", "pasta", 12])

        with pytest.raises(ValueError):
            cfg.estimator_names_from_patterns(5)

    def test_invalid_choice_fitter(self):
        with pytest.raises(ValueError):
            """tests if empty list is correctly handled"""
            ac = AutoCausality()
            ac.fit(
                pd.DataFrame({"treatment": [0, 1]}),
                "treatment",
                "outcome",
                [],
                [],
                estimator_list=[],
            )


if __name__ == "__main__":
    pytest.main([__file__])
