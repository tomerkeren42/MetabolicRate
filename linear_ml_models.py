# from sklearn import preprocessing
# import time
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# import numpy as np
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class DataMLPredictor:
    def __init__(self, df):
        self.df = df
        self.target_column = 'RMR'
        self.columns = list(df.columns.values)
        self.columns.remove(self.target_column)
        self.model = None
        self.accuracies = {}
        print("\n")
        print("*" * 125)
        print("Starting perform ML algorithms for predicting the 'RMR' featuer")
        print("*" * 125)

    @staticmethod
    def standard_scale_data(train, test):
        sc = StandardScaler()
        X_train = sc.fit_transform(train)
        X_test = sc.transform(test)
        return X_train, X_test

    def ComputeViaLinearRegressionModel(self, features, target, test_size):
        print("Computing Linear Regression Model\n")

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=4)
        X_train, X_test = self.standard_scale_data(X_train, X_test)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        y_test_pred = self.model.predict(X_test)
        self.accuracies["acc_linreg"] = metrics.r2_score(y_test, y_test_pred)
        # y_pred = self.model.predict(X_train)
        # coeffcients = pd.DataFrame([X_train.columns, self.model.coef_]).T
        # coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
        # print('R^2:', metrics.r2_score(y_train, y_pred))
        # print('Adjusted R^2:', 1 - (1 - metrics.r2_score(y_train, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
        # print('MSE:', metrics.mean_squared_error(y_train, y_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        # plt.scatter(y_train, y_pred)
        # plt.xlabel("RMR")
        # plt.ylabel("Predicted RMR")
        # plt.title("RMR vs Predicted RMR")
        # plt.scatter(y_pred, y_train - y_pred)
        # plt.title("Predicted vs residuals")
        # plt.xlabel("Predicted")
        # plt.ylabel("Residuals")
        # sns.histplot(y_train - y_pred)
        # plt.title("Histogram of Residuals")
        # plt.xlabel("Residuals")
        # plt.ylabel("Frequency")
        # Predicting Test data with the model
        # print('R^2:', acc_linreg)
        # print('Adjusted R^2:', 1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
        # print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

    def ComputeViaRandomForestRegressor(self, features, target, test_size):
        print("Computing Random Forest Model\n")

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=4)
        X_train, X_test = self.standard_scale_data(X_train, X_test)

        self.model = RandomForestRegressor()
        self.model.fit(X_train, y_train)
        y_test_pred = self.model.predict(X_test)
        self.accuracies["acc_rf"] = metrics.r2_score(y_test, y_test_pred)
        # y_pred = self.model.predict(X_train)
        # print('R^2:', metrics.r2_score(y_train, y_pred))
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_train, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
        # print('MSE:', metrics.mean_squared_error(y_train, y_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        # plt.scatter(y_train, y_pred)
        # plt.xlabel("RMR")
        # plt.ylabel("Predicted RMR")
        # plt.title("RMR vs Predicted RMR")
        # plt.scatter(y_pred, y_train - y_pred)
        # plt.title("Predicted vs residuals")
        # plt.xlabel("Predicted")
        # plt.ylabel("Residuals")

        # print('R^2:', acc_rf)
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
        # print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
        # plt.show()

    def ComputeViaXGBoostRegressor(self, features, target, test_size):
        print("Computing XG Boost Model\n")

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=4)
        X_train, X_test = self.standard_scale_data(X_train, X_test)

        self.model = XGBRegressor()
        self.model.fit(X_train, y_train)
        y_test_pred = self.model.predict(X_test)
        self.accuracies["acc_xgb"] = metrics.r2_score(y_test, y_test_pred)
        # y_pred = reg.predict(X_train)
        # print('R^2:', metrics.r2_score(y_train, y_pred))
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_train, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
        # print('MSE:', metrics.mean_squared_error(y_train, y_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        # plt.scatter(y_train, y_pred)
        # plt.xlabel("RMR")
        # plt.ylabel("Predicted RMR")
        # plt.title("RMR vs Predicted RMR")
        # plt.scatter(y_pred, y_train - y_pred)
        # plt.title("Predicted vs residuals")
        # plt.xlabel("Predicted")
        # plt.ylabel("Residuals")
        # y_test_pred = reg.predict(X_test)

        # print('R^2:', acc_xgb)
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
        # print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
        # plt.show()

    def ComputeViaSVMRegressor(self, features, target, test_size):
        print("Computing SVM Model\n")

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=4)
        X_train, X_test = self.standard_scale_data(X_train, X_test)

        self.model = svm.SVR()
        self.model.fit(X_train, y_train)
        y_test_pred = self.model.predict(X_test)
        self.accuracies["acc_svm"] = metrics.r2_score(y_test, y_test_pred)
        # y_pred = self.model.predict(X_train)
        # print('R^2:', metrics.r2_score(y_train, y_pred))
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_train, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
        # print('MSE:', metrics.mean_squared_error(y_train, y_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        # plt.scatter(y_train, y_pred)
        # plt.xlabel("RMR")
        # plt.ylabel("Predicted RMR")
        # plt.title("RMR vs Predicted RMR")
        # plt.scatter(y_pred, y_train - y_pred)
        # plt.title("Predicted vs residuals")
        # plt.xlabel("Predicted")
        # plt.ylabel("Residuals")

        # print('R^2:', acc_xgb)
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
        # print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

    def CompareModels(self):
        features = self.df.drop([self.target_column], axis=1)
        target = self.df[self.target_column]
        test_set_size = 0.2

        self.ComputeViaLinearRegressionModel(features.copy(), target.copy(), test_set_size)
        self.ComputeViaRandomForestRegressor(features.copy(), target.copy(), test_set_size)
        self.ComputeViaXGBoostRegressor(features.copy(), target.copy(), test_set_size)
        self.ComputeViaSVMRegressor(features.copy(), target.copy(), test_set_size)

        return [self.accuracies["acc_linreg"] * 100, self.accuracies["acc_rf"] * 100, self.accuracies["acc_xgb"] * 100, self.accuracies["acc_svm"] * 100]
