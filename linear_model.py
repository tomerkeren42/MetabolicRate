from sklearn import preprocessing
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from scipy import stats
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm


class DataPredictor:
    def __init__(self, df):
        self.df = df
        self.target_column = 'RMR'
        self.columns = list(df.columns.values)
        self.columns.remove(self.target_column)
        self.model = None
        self.accuracies = {}

    def preprocess(self):
        pass

    def ComputeMSELinearApproaches(self):
        scores_map = {}
        x = self.df.loc[:, self.columns]
        y = self.df['RMR']
        min_max_scaler = preprocessing.MinMaxScaler()
        x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=self.columns)
        y = np.log1p(y)
        for col in x.columns:
            if np.abs(x[col].skew()) > 0.3:
                x[col] = np.log1p(x[col])
        start_time = time.time()
        l_model = linear_model.LinearRegression()
        kf = KFold(n_splits=10)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        scores = cross_val_score(l_model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        scores_map['LinearRegression'] = scores

        print("MSE Linear: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
        print(f"Time passed: {time.time() - start_time}")

        l_ridge = linear_model.Ridge()
        scores = cross_val_score(l_ridge, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        scores_map['Ridge'] = scores
        print("MSE Ridge: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
        print(f"Time passed: {time.time() - start_time}")

        for degree in range(2, 6):
           model = make_pipeline(PolynomialFeatures(degree=degree), linear_model.Ridge())
           scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
           print(f"MSE pipeline degree {degree}: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
        model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
        scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        scores_map['PolyRidge'] = scores
        print("MSE PolyRidge: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
        print(f"Time passed: {time.time() - start_time}")

        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        grid_sv = GridSearchCV(svr_rbf, cv=kf, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='neg_mean_squared_error')
        grid_sv.fit(x_scaled, y)
        print("Best SVR classifier :", grid_sv.best_estimator_)
        scores = cross_val_score(svr_rbf, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        scores_map['SVR'] = scores
        print("MSE SVR: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
        print(f"Time passed: {time.time() - start_time}")

        desc_tr = DecisionTreeRegressor(max_depth=5)
        grid_sv = GridSearchCV(desc_tr, cv=kf, param_grid={"max_depth" : [1, 2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
        grid_sv.fit(x_scaled, y)
        scores = cross_val_score(desc_tr, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        scores_map['DecisionTreeRegressor'] = scores
        print("MSE DecisionTree: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
        print(f"Time passed: {time.time() - start_time}")

        knn = KNeighborsRegressor(n_neighbors=7)
        scores = cross_val_score(knn, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        scores_map['KNeighborsRegressor'] = scores
        grid_sv = GridSearchCV(knn, cv=kf, param_grid={"n_neighbors" : [2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
        grid_sv.fit(x_scaled, y)
        print("Best classifier :", grid_sv.best_estimator_)
        print("KNN Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
        print(f"Time passed: {time.time() - start_time}")

        gbr = GradientBoostingRegressor(alpha=0.9, learning_rate=0.05, max_depth=2, min_samples_leaf=5,
                                        min_samples_split=2, n_estimators=100, random_state=30)
        param_grid={'n_estimators':[100, 200], 'learning_rate': [0.1, 0.05, 0.02], 'max_depth':[2, 4, 6], 'min_samples_leaf':[3, 5, 9]}
        grid_sv = GridSearchCV(gbr, cv=kf, param_grid=param_grid, scoring='neg_mean_squared_error')
        grid_sv.fit(x_scaled, y)
        print("Best GradientBoosting classifier :", grid_sv.best_estimator_)
        scores = cross_val_score(gbr, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        scores_map['GradientBoostingRegressor'] = scores
        print("MSE GradientBoosting: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
        print(f"Time passed: {time.time() - start_time}")

        plt.figure(figsize=(20, 10))
        scores_map = pd.DataFrame(scores_map)
        sns.boxplot(data=scores_map)
        plt.show()

    def ComputeViaLinearRegressionModel(self):
        X = self.df.drop([self.target_column], axis=1)
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        coeffcients = pd.DataFrame([X_train.columns, self.model.coef_]).T
        coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
        y_pred = self.model.predict(X_train)
        # print('R^2:', metrics.r2_score(y_train, y_pred))
        # print('Adjusted R^2:', 1 - (1 - metrics.r2_score(y_train, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
        # print('MSE:', metrics.mean_squared_error(y_train, y_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        plt.scatter(y_train, y_pred)
        plt.xlabel("RMR")
        plt.ylabel("Predicted RMR")
        plt.title("RMR vs Predicted RMR")
        plt.scatter(y_pred, y_train - y_pred)
        plt.title("Predicted vs residuals")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        sns.histplot(y_train - y_pred)
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        # Predicting Test data with the model
        y_test_pred = self.model.predict(X_test)
        self.accuracies["acc_linreg"] = metrics.r2_score(y_test, y_test_pred)
        # print('R^2:', acc_linreg)
        # print('Adjusted R^2:', 1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
        # print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

    def ComputeViaRandomForestRegressor(self):
        X = self.df.drop([self.target_column], axis=1)
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

        # Create a Random Forest Regressor
        reg = RandomForestRegressor()

        # Train the model using the training sets
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_train)
        # print('R^2:', metrics.r2_score(y_train, y_pred))
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_train, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
        # print('MSE:', metrics.mean_squared_error(y_train, y_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        plt.scatter(y_train, y_pred)
        plt.xlabel("RMR")
        plt.ylabel("Predicted RMR")
        plt.title("RMR vs Predicted RMR")
        plt.scatter(y_pred, y_train - y_pred)
        plt.title("Predicted vs residuals")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        y_test_pred = reg.predict(X_test)
        self.accuracies["acc_rf"] = metrics.r2_score(y_test, y_test_pred)

        # print('R^2:', acc_rf)
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
        # print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
        # plt.show()

    def ComputeViaXGBoostRegressor(self):
        X = self.df.drop([self.target_column], axis=1)
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
        reg = XGBRegressor()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_train)
        # print('R^2:', metrics.r2_score(y_train, y_pred))
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_train, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
        # print('MSE:', metrics.mean_squared_error(y_train, y_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        plt.scatter(y_train, y_pred)
        plt.xlabel("RMR")
        plt.ylabel("Predicted RMR")
        plt.title("RMR vs Predicted RMR")
        plt.scatter(y_pred, y_train - y_pred)
        plt.title("Predicted vs residuals")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        y_test_pred = reg.predict(X_test)

        self.accuracies["acc_xgb"] = metrics.r2_score(y_test, y_test_pred)
        # print('R^2:', acc_xgb)
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
        # print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
        # plt.show()

    def ComputeViaSVMRegressor(self):
        X = self.df.drop([self.target_column], axis=1)
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        reg = svm.SVR()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_train)
        # print('R^2:', metrics.r2_score(y_train, y_pred))
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_train, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
        # print('MSE:', metrics.mean_squared_error(y_train, y_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
        plt.scatter(y_train, y_pred)
        plt.xlabel("RMR")
        plt.ylabel("Predicted RMR")
        plt.title("RMR vs Predicted RMR")
        plt.scatter(y_pred, y_train - y_pred)
        plt.title("Predicted vs residuals")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        y_test_pred = reg.predict(X_test)
        self.accuracies["acc_svm"] = metrics.r2_score(y_test, y_test_pred)
        # print('R^2:', acc_xgb)
        # print('Adjusted R^2:',
        #       1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
        # print('MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
        # print('MSE:', metrics.mean_squared_error(y_test, y_test_pred))
        # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

    def CompareModels(self):
        print("Computing Linear Regression Model")
        self.ComputeViaLinearRegressionModel()
        print("Computing Random Forest Model")
        self.ComputeViaRandomForestRegressor()
        print("Computing XG Boost Model")
        self.ComputeViaXGBoostRegressor()
        print("Computing SVM Model")
        self.ComputeViaSVMRegressor()
        models = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Support Vector Machines'],
            'R-squared Score': [self.accuracies["acc_linreg"] * 100, self.accuracies["acc_rf"] * 100, self.accuracies["acc_xgb"] * 100, self.accuracies["acc_svm"] * 100]})
        models = models.sort_values(by='R-squared Score', ascending=False)
        print(models)

        print(f"We've found out that the best model to predict the RMR is: {models.iloc[0][0]} with R-2 Accuracy of {np.round(models.iloc[0][1], 3)}%")
