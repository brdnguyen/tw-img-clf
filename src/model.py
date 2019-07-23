from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
SEED = 15


def fit_predict(model, X, y, X_cv):
    model.fit(X, y)
    preds = model.predict(X_cv)
    return preds


def cal_score(y_preds, y):
    # ("y_preds", y_preds)
    # print("y", y)
    l = sum([1 if y_preds[idx] == y[idx] else 0 for idx in range(len(y_preds))])
    return l / len(y_preds) * 1.0


def scoring(estimator, X, y):
    y_preds = estimator.predict(X)
    return cal_score(y_preds, y)



def dummy_model(images, NCLASS):
    ''' Dummy model just to say hello world'''
    import random
    preds = []
    for idx, img in enumerate(images):
        preds.append(random.randint(0, NCLASS - 1))
    return preds


def cross_validate(model, X_train, y_train, print_folds=False):
    ''' Useful to see how different folds converge '''
    KFOLD = 3 #3 is enough to tell the range of scoring
    mean_auc = 0.
    for i in range(KFOLD):
        if print_folds: print("calculating fold ", i + 1)
        X, X_cv, y, y_cv = train_test_split(
            X_train, y_train,
            test_size = 0.2,
            random_state = i*SEED
        )

        y_preds = fit_predict(model, X, y, X_cv)
        scoring = cal_score(y_preds, y_cv)

        if print_folds: print("Score (fold %d/%d): %f" % (i + 1, KFOLD, scoring))
        mean_auc += scoring
    return mean_auc/KFOLD

def model_selection(X_train, y_train):
    models = [
        LogisticRegression(solver='lbfgs', multi_class="auto", max_iter=4000),  # compensate for redundancy - few non-colinear data
        RandomForestClassifier(n_estimators = 200, max_features = 'sqrt', max_depth = 7, n_jobs = 4),
        GradientBoostingClassifier(n_estimators = 200, max_depth = 7, subsample = 0.8)
    ]

    KFOLD = KFold(n_splits=10, random_state=SEED)
    best_score = -1

    print("X_train, y_train: ", len(X_train), len(y_train))
    for model in models:
        # cv_result = cross_val_score(model, X_train, y_train, cv=KFOLD, scoring='accuracy').mean()
        cv_result = cross_validate(model, X_train, y_train)
        print(str(model)[:3].upper(), ": ", cv_result)
        if cv_result > best_score:
            best_score = cv_result
            best_model = model
    best_model.fit(X_train, y_train)
    return best_model
