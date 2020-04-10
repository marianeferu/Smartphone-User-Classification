import glob
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.svm import SVC


def grid_search(X, y):
    C = [1, 0.1]
    degree = [3, 4]
    kernel = ['poly']

    param = {'C': C, 'kernel': kernel, 'degree': degree}
    algorithm = SVC(gamma=1)
    folds = KFold(n_splits=3)  # impartirea in 3 folduri
    clf = GridSearchCV(algorithm, param, cv=folds, scoring="accuracy")  # metoda de gridsearch pentru cautarea param
    cross_score = cross_val_score(estimator=algorithm, X=X, y=y.values.ravel(),
                                  cv=folds)  # realizarea scorului de antrenare cu cross validare
    clf.fit(X, y.values.ravel())  # antrenarea

    print("Training Average Score: ", np.average(cross_score))  # media scorurilor obtinute
    print('Best score: {}'.format(clf.best_score_))
    print('Best parameters: {}'.format(clf.best_params_))
    return clf.best_params_


# citirea si incarcarea datelor
def load_data(data_type):
    all_files = glob.glob(PATH + data_type + "/*.csv")
    names = []
    data = []
    for filename in all_files:
        sample = pd.read_csv(filename, header=None, names=["Xs", "Ys", "Zs"])
        data.append(sample)
        names.append(filename[-9: -4])

    return names, data

PATH = "C:\\Users\\Maria\\Desktop\\ProiectML\\Dataset\\"
path_train_data = "Train Set\\"
path_test_data = "Test Set\\"

names_train, train_data = load_data(path_train_data)
names_test, test_data = load_data(path_test_data)

labels = pd.read_csv(PATH + "Labels.csv")
labels.drop('id', axis = 1, inplace = True)

display(train_data[0].head(n = 10)) #primele 10 din datele de antrenare


# extragere feature-uri

def get_statistics(raw_data):
    column_names = ["mean_x", "std_x", "median_x", "min_x", "max_x", "mean_y", "std_y", "median_y", "min_y", "max_y",
                    "mean_z", "std_z", "median_z", "min_z", "max_z", ]
    data = pd.DataFrame(columns=column_names)
    for index in range(len(raw_data)):
        xs = raw_data[index]["Xs"].values
        ys = raw_data[index]["Ys"].values
        zs = raw_data[index]["Zs"].values

        # de-a lungul axei Xs:
        x_median = np.median(xs)  # calcularea medianei de-a lungul axei Xs
        x_mean = xs.mean()  # calcularea mediei
        x_std = xs.std()  # calcularea deviatiei standard
        x_min = xs.min()  # calcularea minimului
        x_max = xs.max()  # calcularea maximului

        # de-a lungul axei Ys:
        y_median = np.median(ys)  # calcularea medianei
        y_mean = ys.mean()  # calcularea mediei
        y_std = ys.std()  # calcularea deviatiei standard
        y_min = ys.min()  # calcularea minimului
        y_max = ys.max()  # calcularea maximului

        # de-a lungul axei Zs :
        z_median = np.median(zs)  ##calcularea medianei
        z_mean = zs.mean()  # calcularea mediei
        z_std = zs.std()  # calcularea deviatiei standard
        z_min = zs.min()  # calcularea minimului
        z_max = zs.max()  # calcularea maximului

        data.loc[len(data)] = [x_mean, x_std, x_median, x_min, x_max,
                               y_mean, y_std, y_median, y_min, y_max,
                               z_mean, z_std, z_median, z_min, z_max]

    return data

statistics = get_statistics(train_data)

X_train, X_test, y_train, y_test = train_test_split(statistics, labels, train_size = 0.8, test_size = 0.2)

dictionary = grid_search(X_train, y_train)
C = dictionary['C']
kernel = dictionary['kernel']
degree = dictionary['degree']
# C = C, kernel = kernel, degree = degree

classifier = SVC(C = C, kernel = kernel, degree = degree, gamma = 1)
classifier.fit(X_train, y_train.values.ravel()) # clasificare pe datele de antrenare
y_predict = classifier.predict(X_test) # prezicere pe datele de test
accuracy = accuracy_score(y_test, y_predict) #acuratetea
print("SVM Accuracy: ", accuracy)

test = get_statisitics(test_data)
predictions = classifier.predict(test)


#submisia
column_names = ["id", "class"]
submission = pd.DataFrame(columns = column_names)

submission["id"] = names_test
submission["class"] = predictions

submission.to_csv("submission_svm_4.csv", sep = ",", columns = ["id","class"], index=False)