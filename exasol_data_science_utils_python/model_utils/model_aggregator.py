from typing import List, Any

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder


def combine_random_forrest_classifier(rf_a, rf_b):
    import copy
    rf = copy.deepcopy(rf_a)
    rf.estimators_ += rf_b.estimators_
    rf.n_estimators = len(rf.estimators_)
    return rf


def combine_to_voting_classifier(estimators: List[Any], classes: int):
    estimator_tuples = [(str(i), estimator) for i, estimator in enumerate(estimators)]
    classifier = VotingClassifier(estimator_tuples,voting='soft')
    classifier.le_ = LabelEncoder().fit(np.arange(classes))
    classifier.classes_ = classes
    classifier.estimators_ = estimators
    return classifier
