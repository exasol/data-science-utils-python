from typing import List, Any

from sklearn.ensemble import RandomForestClassifier, VotingRegressor


def combine_random_forrest_classifier(estimators: List[RandomForestClassifier]):
    import copy
    rf = copy.deepcopy(estimators[0])
    for i in range(1, len(estimators)):
        rf.estimators_ += estimators[i].estimators_
        rf.n_estimators = len(estimators[i].estimators_)
    return rf

# TODO test combine_to_voting_classifier with different class lists
# def combine_to_voting_classifier(estimators: List[Any], classes: Union[List[int], List[str]],
#                                  **kwargs) -> VotingClassifier:
#     estimator_tuples = [(str(i), estimator) for i, estimator in enumerate(estimators)]
#     classifier = VotingClassifier(estimator_tuples, **kwargs)
#     classifier.le_ = LabelEncoder().fit(classes)
#     classifier.classes_ = classes
#     classifier.estimators_ = estimators
#     return classifier


def combine_to_voting_regressor(estimators: List[Any], **kwargs) -> VotingRegressor:
    estimator_tuples = [(str(i), estimator) for i, estimator in enumerate(estimators)]
    regressor = VotingRegressor(estimator_tuples, **kwargs)
    regressor.estimators_ = estimators
    return regressor
