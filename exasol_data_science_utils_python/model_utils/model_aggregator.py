from typing import List, Any

import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def combine_random_forrest_classifier(estimators: List[RandomForestClassifier]):
    import copy
    rf = copy.deepcopy(estimators[0])
    for i in range(1, len(estimators)):
        rf.estimators_ += estimators[i].estimators_
        rf.n_estimators = len(estimators[i].estimators_)
    return rf


def combine_to_voting_classifier(estimators: List[Any], classes: int, **kwargs) -> VotingClassifier:
    estimator_tuples = [(str(i), estimator) for i, estimator in enumerate(estimators)]
    classifier = VotingClassifier(estimator_tuples, **kwargs)
    classifier.le_ = LabelEncoder().fit(np.arange(classes))
    classifier.classes_ = classes
    classifier.estimators_ = estimators
    return classifier
