from sklearn.ensemble import RandomForestClassifier as rfc
import numpy as np

class RFC:

    def __init__(self, trees, depth, class_to_int, int_to_class, feat, classes):
        self.class_to_int = class_to_int
        self.int_to_class = int_to_class
        self.model = rfc(n_estimators=trees, max_depth=depth, n_jobs=-1, verbose=1)
        self.model.fit(feat, classes)

    def classify(self, testData):
        return self.model.predict(testData)

    def classify_conf(self, testData):
        proba = self.model.predict_proba(testData)
        return np.max(proba), np.argmax(proba)