from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaboostEnsembles:
    # creates n_classes number of ensembles
    # X : numpy array of size n_samples, n_features
    # Y: numpy array of size n_samples,
    def fit(self,X, y,n_estimators = 50):
        # create n_classes number of ensembles
        #@TODO experiment with the number of estimators
        # print(X)
        # print(y)

        le = LabelEncoder()
        y_new = le.fit_transform(y)
        # print(y_new)
        self.ensemble_list = []
        for class_label in le.classes_:
            y_class_label = y_new == le.transform([class_label])
            y_class_label = y_class_label.astype(int) # hack to convert boolean array into numeric array
            # train ensemble for this
            ensemble = AdaBoostClassifier(n_estimators=n_estimators, base_estimator=DecisionTreeClassifier(max_depth=1))
            ensemble.fit(X,y_class_label)
            self.ensemble_list.append(ensemble)

    """
    X : numpy array corresponding to a single sequence  n_samples , n_features
    """
    def ensemble_scores(self,X):
        scores = np.zeros((X.shape[0],len(self.ensemble_list)))
        for i,ensemble in enumerate(self.ensemble_list):
            scores[:,i] = ensemble.decision_function(X)
        return scores
