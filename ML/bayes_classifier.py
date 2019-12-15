from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB


class Classifier:
    def __init__(self, features, labels):
        self.X = features
        self.y_hat = labels

    def preprocess(self):  # expects X to be a Pandas df from main.py
        lb = LabelEncoder()
        for x in self.X:
            self.X[x] = lb.fit_transform(list(self.X[x]))

    def train(self):
        model = CategoricalNB()
        self.preprocess()
        model.fit(self.X, self.y_hat)
        train_validation_split = 0.95
        print("Model Accuracy: ")
        print(100 * model.score(self.X[int(len(self.X) * train_validation_split):],
                          self.y_hat[int(len(self.y_hat) * train_validation_split):]))

    #
    # def eval(self, categories):
    #     if len(categories) > 9:
    #         print("Too many items!")

        # voted_for_in_2012_election,political_spectrum,social_class,gender,citizenship,living_location,sexuality,income,education,reltrad

