import pandas as pd
import plotly.express as px
# ______Import streamlit library for building App______#
import streamlit as st
import streamlit_ext as ste
# ______end______#

import pickle
import io

import numpy as np
from sklearn.metrics import r2_score, accuracy_score, precision_score
# ______Import libraries for regulariztion and scaling______#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PolynomialFeatures
# ______end______#

# ______Model Import______#
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
# ______end of Importing models______#


class WebApp:
    def __init__(self) -> None:
        # init the shape of the web page
        self.headers = []
        self.train_split = 0.0
        self.test_split = 0.0
        self.X_whole = pd.DataFrame()
        self.Y_whole = pd.DataFrame()
        self.data = pd.DataFrame()
        self.UniqueDic = {}
        self.headers = []
        self.startEDA = False
        st.title("Models Trainer")
        st.header("please upload your dataset")
        self.tab1, self.tab2 = st.tabs(
            ["Data Preprocessing", "select the model"])
        self.MainApp()
        pass

    @st.cache_data
    def LoadData(_self, file):
        '''
        input: 
            file->csv file 
        functionality:
            transform the csv file into a pandas dataframe
        return:
            pd.DataFrame
        '''
        with st.spinner('Wait for it...'):
            while (file is None):
                st.stop()  # wait until loading the data
        st.success('Done✅')
        return pd.read_csv(file)  # return the loaded data

    @st.cache_resource
    def pickle_model(_self, _model):
        """
        input: 
            sciket-learn Model
        functionality:
            Pickle the model inside bytes.
        return:
            the encoded model
        """
        f = io.BytesIO()
        pickle.dump(_model, f)
        return f

    @st.cache_resource(experimental_allow_widgets=True)
    def dowloadTheModel(_self, _data, _modelName):
        return ste.download_button("Download model file", data=_data, file_name=f"model{_modelName}.pkl",)

    def selectModel(self, problem_type):
        with self.tab2:
            st.header(f"the Problem is {problem_type}")
            with st.spinner('Wait for finishing data preprocessing'):
                while (self.startEDA == False):
                    st.stop()
            st.success('Lets choose the model')
            st.header("Split the data as percentage")
            # split the data for training and testing
            _, _, col2, _, _ = st.columns(5)
            with col2:
                self.train_split = st.number_input("Train split")

            _, _, col2 = st.columns(3)

            with col2:
                state_split = st.toggle("split")

            if state_split:
                x_train, x_test, y_train, y_test = train_test_split(
                    self.X_whole, self.Y_whole, train_size=self.train_split)
                st.info("done spliting✅")
                scalerOption = st.selectbox(
                    "select your scaler", ("StandardScaler", "MaxAbsScaler", "MinMaxScaler"))
                if scalerOption == "StandardScaler":
                    scaler = StandardScaler()
                elif scalerOption == "MaxAbsScaler":
                    scaler = MaxAbsScaler()
                elif scalerOption == "MinMaxScaler":
                    scaler = MinMaxScaler()
                _, _, col2 = st.columns(3)
                with col2:
                    scaling = st.toggle("start scaling")
                if scaling:
                    x_train_scaled = scaler.fit_transform(x_train)
                    x_test_scaled = scaler.transform(x_test)
                    st.info("done scaling✅")
                else:
                    x_train_scaled = x_train
                    x_test_scaled = x_test

            choice = problem_type
            # select regression type
            if choice == "Regression":
                st.subheader("Select type of regression")
                self.modelType = st.selectbox(
                    "type of regressor", ('Linear', 'Polynomial', "SVR", "SGD Regressor", "Gradient Boost Regressor"))

                if self.modelType == "Gradient Boost Regressor":
                    param = {"learning_rate": [0.01, 0.001, 0.0001, 0.00001],
                             "loss": ['squared_error', 'absolute_error', 'huber', 'quantile'],
                             "n_estimators": range(100, 1000, 100),
                             "criterion": ['friedman_mse', 'squared_error'],
                             "min_samples_split": range(1, 7),
                             "tol": [1e-2, 1e-3, 1e-5, 1e-6, 1e-7, 1e-8],
                             "warm_start": [True, False]
                             }
                    brain = GradientBoostingRegressor()
                    _, _, col2 = st.columns(3)
                    with col2:
                        stratTrain = st.toggle(
                            f"start {self.modelType} training")
                    if stratTrain:
                        grained_brain = GridSearchCV(brain, param_grid=param)
                        with st.spinner("training.."):
                            grained_brain.fit(x_train_scaled, y_train)
                        st.info("done Training🔥")
                        st.balloons()
                        best_model = grained_brain.best_estimator_
                        Y_predict = best_model.predict(x_test_scaled)
                        preformance = {
                            "the r2 score of the model is ": r2_score(y_test, Y_predict)}
                        st.dataframe(preformance, use_container_width=True)
                        data = self.pickle_model(best_model)
                        self.dowloadTheModel(data, self.modelType)

                if self.modelType == "SGD Regressor":
                    param = {"penalty": ['l2', 'l1', 'elasticnet'],
                             "loss": ["squared_error", "huber"],
                             "alpha": [0.01, 0.001, 0.0001],
                             "fit_intercept": [True, False],
                             "max_iter": [1000, 10000, 100000, -1],
                             "tol": [1e-2, 1e-3, 1e-5, 1e-6, 1e-7, 1e-8],
                             "warm_start": [True, False]
                             }
                    brain = SGDRegressor()
                    _, _, col2 = st.columns(3)
                    with col2:
                        stratTrain = st.toggle(
                            f"start {self.modelType} training")
                    if stratTrain:
                        grained_brain = GridSearchCV(brain, param_grid=param)
                        with st.spinner("training.."):
                            grained_brain.fit(x_train_scaled, y_train)
                        st.info("done Training🔥")
                        st.balloons()
                        best_model = grained_brain.best_estimator_
                        Y_predict = best_model.predict(x_test_scaled)
                        preformance = {
                            "the r2 score of the model is ": r2_score(y_test, Y_predict)}
                        st.dataframe(preformance, use_container_width=True)
                        data = self.pickle_model(best_model)
                        self.dowloadTheModel(data, self.modelType)

                # adjust the dataset on the polynomial feature and train polynomial regression
                if self.modelType == "Polynomial":
                    brain = LinearRegression()
                    polynomaialDegree = st.number_input(
                        "enter the degree of the regressor")
                    if int(polynomaialDegree) != 0:
                        polyfet = PolynomialFeatures(degree=int(
                            polynomaialDegree), include_bias=True)
                        x_train_scaled = polyfet.fit_transform(x_train_scaled)
                        x_test_scaled = polyfet.transform(x_test_scaled)
                    _, _, col2 = st.columns(3)
                    with col2:
                        stratTrain = st.toggle(
                            f"start {self.modelType} training")
                    # start training model
                    if stratTrain:
                        brain.fit(x_train_scaled, y_train)
                        Y_predict = brain.predict(x_test_scaled)
                        preformance = {
                            "the r2 score of the model is ": r2_score(y_test, Y_predict)}
                        st.dataframe(preformance, use_container_width=True)
                        data = self.pickle_model(brain)
                        self.dowloadTheModel(data, self.modelType)

                # train linear regression model
                if self.modelType == "Linear":
                    brain = LinearRegression(fit_intercept=True)
                    _, _, col2 = st.columns(3)
                    with col2:
                        stratTrain = st.toggle(
                            f"start {self.modelType} training")
                    # start training model
                    if stratTrain:
                        brain.fit(x_train_scaled, y_train)
                        Y_predict = brain.predict(x_test_scaled)
                        preformance = {
                            "the r2 score of the model is ": r2_score(y_test, Y_predict)}
                        st.dataframe(preformance, use_container_width=True)
                        data = self.pickle_model(brain)
                        self.dowloadTheModel(data, self.modelType)

                # Train an svm regressor
                if self.modelType == "SVR":
                    param = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                             "degree": [1, 2, 3, 4, 5, 6, 7, 8],
                             "gamma": ["scale"],
                             "coef0": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                             "tol": [1e-2, 1e-3, 1e-5, 1e-6, 1e-7, 1e-8]}
                    brain = SVR()
                    _, _, col2 = st.columns(3)
                    with col2:
                        stratTrain = st.toggle(
                            f"start {self.modelType} training")
                    if stratTrain:
                        grained_brain = GridSearchCV(brain, param_grid=param)
                        with st.spinner("training.."):
                            grained_brain.fit(x_train_scaled, y_train)
                        st.info("done Training🔥")
                        st.balloons()
                        best_model = grained_brain.best_estimator_
                        Y_predict = best_model.predict(x_test_scaled)
                        preformance = {
                            "the r2 score of the model is ": r2_score(y_test, Y_predict)}
                        st.dataframe(preformance, use_container_width=True)
                        data = self.pickle_model(best_model)
                        self.dowloadTheModel(data, self.modelType)

            # select classification type
            if choice == "Classification":
                st.subheader("Select type of Classification")
                self.modelType = st.selectbox("type of classificator", (
                                              'KNN', 'SVM', "Descision Tree"))

                if self.modelType == "KNN":
                    param = {"n_neighbors": range(1, 20),
                             "weights": ['uniform', 'distance'],
                             "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                             "leaf_size": range(1, 20)}
                    brain = KNeighborsClassifier()
                    _, _, col2 = st.columns(3)
                    with col2:
                        stratTrain = st.toggle(
                            f"start {self.modelType} training")
                    # start training model
                    if stratTrain:
                        grained_brain = GridSearchCV(brain, param_grid=param)
                        with st.spinner("training.."):
                            grained_brain.fit(x_train_scaled, y_train)
                        st.info("done Training🔥")
                        st.balloons()
                        best_model = grained_brain.best_estimator_
                        Y_predict = best_model.predict(x_test_scaled)
                        tableOfpreformance = {"the accuracy of the model is ": accuracy_score(
                            y_test, Y_predict)}
                        st.dataframe(tableOfpreformance,
                                     use_container_width=True)
                        data = self.pickle_model(best_model)
                        self.dowloadTheModel(data, self.modelType)

                if self.modelType == "SVM":
                    param = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                             "degree": range(1, 15),
                             "gamma": ["scale"],
                             "coef0": range(1, 15),
                             "tol": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]}
                    brain = SVC()
                    _, _, col2 = st.columns(3)
                    with col2:
                        stratTrain = st.toggle(
                            f"start {self.modelType} training")
                    # start training model
                    if stratTrain:
                        grained_brain = GridSearchCV(brain, param_grid=param)
                        with st.spinner("training.."):
                            grained_brain.fit(x_train_scaled, y_train)
                        st.info("done Training🔥")
                        st.balloons()
                        best_model = grained_brain.best_estimator_
                        Y_predict = best_model.predict(x_test_scaled)
                        tableOfpreformance = {"the accuracy of the model is ": accuracy_score(
                            y_test, Y_predict)}
                        st.dataframe(tableOfpreformance,
                                     use_container_width=True)
                        data = self.pickle_model(best_model)
                        self.dowloadTheModel(data, self.modelType)

                if self.modelType == "Descision Tree":
                    param = {"criterion": ['gini', 'entropy', 'log_loss'],
                             "splitter": ['best', 'random'],
                             "max_depth": range(1, 20),
                             "max_features": ['sqrt', 'log2'],
                             "max_leaf_nodes": range(1, 20)}

                    brain = DecisionTreeClassifier()
                    _, _, col2 = st.columns(3)
                    with col2:
                        stratTrain = st.toggle(
                            f"start {self.modelType} training")
                    # start training model
                    if stratTrain:
                        grained_brain = GridSearchCV(brain, param_grid=param)
                        with st.spinner("training.."):
                            grained_brain.fit(x_train_scaled, y_train)
                        st.info("done Training🔥")
                        st.balloons()
                        best_model = grained_brain.best_estimator_
                        Y_predict = best_model.predict(x_test_scaled)
                        tableOfpreformance = {"the accuracy of the model is ": accuracy_score(
                            y_test, Y_predict)}
                        st.dataframe(tableOfpreformance,
                                     use_container_width=True)
                        data = self.pickle_model(best_model)
                        self.dowloadTheModel(data, self.modelType)

    def dataPreprocessing(self):
        '''
        input:
            none
        functinality:
            this function is used to preprocess and clean the data using the pandas 
        return:
            None
        '''
        with self.tab1:
            self.file = st.file_uploader(
                "dataset must be uploaded in csv extensiom")
            self.data = self.LoadData(self.file)
            st.subheader("Data sample")
            st.dataframe(self.data.head(), use_container_width=True)
            # Plot the corelation between the data
            st.subheader("Corelation between dataset")
            corr_matrix = self.data.corr(numeric_only=True)
            fig = px.imshow(corr_matrix, text_auto=True)
            st.plotly_chart(fig)
            st.header("Data Cleaning and preprocessing")
            self.headers = self.data.columns.to_list()
            if self.file is not None:
                self.notImportant = st.multiselect(
                    'What is not important', options=tuple(self.headers))
                out_model = st.selectbox(
                    "select the output of the model", tuple(self.headers))
                # check if the column countain number or string
                if str(self.data[out_model].dtype) == "object" or str(self.data[out_model].dtype) == "boolean" or str(self.data[out_model].dtype) == "int64":
                    modeltype = "Classification"
                else:
                    modeltype = "Regression"

            _, col2, _ = st.columns(3)

            with col2:
                self.startEDA = st.toggle("  Start Data Preprocessing  ")
            if self.startEDA or st.session_state:
                for i in self.notImportant:
                    # remove the non important column from the data form
                    self.data.pop(i)
                    self.headers.remove(i)
                # show the data after removing the
                st.dataframe(self.data.head(), use_container_width=True)
                # Replace any nan wittin dataframe with 0
                self.data = self.data.replace(np.nan, '0', regex=True)
                # list to save column name with strings element
                columnString = list()
                # check for any column that have string elements
                for i in self.headers:
                    element = str(self.data[i].dtype)  # check the type of the
                    if element == "object" or element == "boolean":
                        # append the string
                        columnString.append(i)
                # give label unique Id for each string element
                for i in columnString:
                    self.UniqueDic[i] = self.data[i].unique()
                    pass
                # replace string element with special Id
                dict = {}
                for i in self.UniqueDic.keys():
                    for (Id, types) in enumerate(self.UniqueDic[i]):
                        self.data = self.data.replace(types, Id)
                        dict[types] = Id
                        pass
                st.subheader("Data after preprocessing")
                st.dataframe(self.data, use_container_width=True)
                st.subheader("Encoded labels")
                st.dataframe(dict, use_container_width=True)
                # split the data to input and output data
                self.X_whole = self.data.drop(out_model, axis=1)
                self.Y_whole = self.data[out_model]
                x, y = st.columns(2)
                with x:
                    st.subheader("the input data")
                    st.dataframe(self.X_whole, use_container_width=True)
                with y:
                    st.subheader("the output data")
                    st.dataframe(self.Y_whole, use_container_width=True)
        return modeltype

    def MainApp(self):
        problem_type = self.dataPreprocessing()
        self.model_requied = self.selectModel(problem_type)


Web = WebApp()
