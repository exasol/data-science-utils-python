from sklearn.datasets import make_blobs, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor

from exasol_data_science_utils_python.model_utils.model_aggregator import combine_random_forrest_classifier, \
    combine_to_voting_regressor


def test_combine_random_forrest_classifier():
    factor = 100

    X, y = make_blobs(n_samples=10 * factor, n_features=500, centers=100,
                      random_state=0)

    X1, y1 = X[:5 * factor], y[:5 * factor]
    X2, y2 = X[5 * factor:], y[5 * factor:]

    clf1 = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=5,
                                  random_state=0)
    clf2 = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=5,
                                  random_state=0)
    clf2.fit(X2, y2)
    clf1.fit(X1, y1)

    clf_combine = combine_random_forrest_classifier([clf1, clf2])

    clf1_x1_y1 = clf1.score(X1, y1)
    print("clf1.score(X1, y1)", clf1_x1_y1)
    clf1_x2_y2 = clf1.score(X2, y2)
    print("clf1.score(X2, y2)", clf1_x2_y2)
    clf1_x_y = clf1.score(X, y)
    print("clf1.score(X, y)", clf1_x_y)

    clf2_x1_y1 = clf2.score(X1, y1)
    print("clf2.score(X1, y1)", clf2_x1_y1)
    clf2_x2_y2 = clf2.score(X2, y2)
    print("clf2.score(X2, y2)", clf2_x2_y2)
    clf2_x_y = clf2.score(X, y)
    print("clf2.score(X, y)", clf2_x_y)

    clf_combine_x1_y1 = clf_combine.score(X1, y1)
    print("clf_combine.score(X1, y1)", clf_combine_x1_y1)
    clf_combine_x2_y2 = clf_combine.score(X2, y2)
    print("clf_combine.score(X2, y2)", clf_combine_x2_y2)
    clf_combine_x_y = clf_combine.score(X, y)
    print("clf_combine.score(X, y)", clf_combine_x_y)

    assert clf1_x_y <= clf_combine_x2_y2 and \
           clf2_x_y <= clf_combine_x1_y1 and \
           clf1_x_y <= clf_combine_x_y and \
           clf2_x_y <= clf_combine_x_y


def test_combine_to_voting_classifier():
    factor = 100

    X, y = make_regression(n_samples=10 * factor, n_features=1000, random_state=0)

    X1, y1 = X[:5 * factor], y[:5 * factor]
    X2, y2 = X[5 * factor:], y[5 * factor:]

    clf1 = SGDRegressor(random_state=0)
    clf2 = SGDRegressor(random_state=0)
    clf2.fit(X2, y2)
    clf1.fit(X1, y1)

    clf_combine = combine_to_voting_regressor([clf1, clf2])

    clf1_x1_y1 = clf1.score(X1, y1)
    print("clf1.score(X1, y1)", clf1_x1_y1)
    clf1_x2_y2 = clf1.score(X2, y2)
    print("clf1.score(X2, y2)", clf1_x2_y2)
    clf1_x_y = clf1.score(X, y)
    print("clf1.score(X, y)", clf1_x_y)

    clf2_x1_y1 = clf2.score(X1, y1)
    print("clf2.score(X1, y1)", clf2_x1_y1)
    clf2_x2_y2 = clf2.score(X2, y2)
    print("clf2.score(X2, y2)", clf2_x2_y2)
    clf2_x_y = clf2.score(X, y)
    print("clf2.score(X, y)", clf2_x_y)

    clf_combine_x1_y1 = clf_combine.score(X1, y1)
    print("clf_combine.score(X1, y1)", clf_combine_x1_y1)
    clf_combine_x2_y2 = clf_combine.score(X2, y2)
    print("clf_combine.score(X2, y2)", clf_combine_x2_y2)
    clf_combine_x_y = clf_combine.score(X, y)
    print("clf_combine.score(X, y)", clf_combine_x_y)

    assert clf1_x_y <= clf_combine_x2_y2 and \
           clf2_x_y <= clf_combine_x1_y1 and \
           clf1_x_y <= clf_combine_x_y and \
           clf2_x_y <= clf_combine_x_y


# TODO test combine_to_voting_classifier with different class lists
# def test_combine_to_voting_classifier():
#     factor = 100
#
#     X, y = make_blobs(n_samples=10 * factor, n_features=10, centers=100,
#                       random_state=0)
#
#     X1, y1 = X[:5 * factor], y[:5 * factor]
#     X2, y2 = X[5 * factor:], y[5 * factor:]
#
#     clf1 = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=10,
#                                   random_state=0)
#     clf2 = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=10,
#                                   random_state=0)
#     clf2.fit(X2, y2)
#     clf1.fit(X1, y1)
#
#     clf_combine = combine_to_voting_classifier([clf1, clf2], 100, voting="soft")
#
#     clf1_x1_y1 = clf1.score(X1, y1)
#     print("clf1.score(X1, y1)", clf1_x1_y1)
#     clf1_x2_y2 = clf1.score(X2, y2)
#     print("clf1.score(X2, y2)", clf1_x2_y2)
#     clf2_x1_y1 = clf2.score(X1, y1)
#     print("clf2.score(X1, y1)", clf2_x1_y1)
#     clf2_x2_y2 = clf2.score(X2, y2)
#     print("clf2.score(X2, y2)", clf2_x2_y2)
#     clf_combine_x1_y1 = clf_combine.score(X1, y1)
#     print("clf_combine.score(X1, y1)", clf_combine_x1_y1)
#     clf_combine_x2_y2 = clf_combine.score(X2, y2)
#     print("clf_combine.score(X2, y2)", clf_combine_x2_y2)
#
#     assert clf1_x1_y1 <= clf_combine_x1_y1 and \
#            clf1_x2_y2 <= clf_combine_x2_y2 and \
#            clf2_x1_y1 <= clf_combine_x1_y1 and \
#            clf2_x2_y2 <= clf_combine_x2_y2
