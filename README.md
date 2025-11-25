Project Report (No Tables)

Breast Cancer Classification Using Supervised Machine Learning

1. Introduction

Breast cancer is one of the most common diseases affecting women worldwide. Early detection through accurate classification of breast tumors as benign or malignant plays a critical role in improving patient outcomes. Machine learning techniques can support medical diagnosis by analyzing patterns in tumor features and predicting the likelihood of cancer.

The objective of this project is to build and evaluate supervised machine learning models using the Breast Cancer Wisconsin Diagnostic Dataset. Two models—Logistic Regression and Random Forest—were implemented and compared to determine which one performs better in identifying malignant and benign tumors.

2. Dataset Description

The dataset used for this project contains 569 entries, each representing a breast tumor sample. Each sample is described using 30 numerical features derived from digital images of cell nuclei. These features include measurements such as radius, texture, perimeter, area, symmetry, compactness, smoothness, concavity, and fractal dimension.

The dataset includes one target column called "diagnosis."
It contains two labels:

M for malignant tumors

B for benign tumors

Before modeling, unnecessary columns ("id" and an empty column named "Unnamed: 32") were removed. The remaining numerical features were retained for training.

3. Methodology

The project follows a structured machine learning workflow involving data preprocessing, model development, training, and evaluation.

3.1 Data Preprocessing

The following steps were performed:

Removing unnecessary columns: The "id" column and the empty "Unnamed: 32" column were deleted because they provide no useful information.

Label encoding: The diagnosis values were converted into numeric format:

Malignant = 1

Benign = 0

Splitting the dataset: The dataset was divided into training and testing sets, with 75% of the data used for training and 25% for testing.

Feature scaling: StandardScaler was applied to normalize the features. This step helps improve the performance of models such as Logistic Regression.

3.2 Models Used

Two machine learning models were used in the project:

Logistic Regression

A classical linear classification model that is simple, fast, and easy to interpret. It is often used as a baseline model in binary classification problems. It works best when the decision boundary between classes is approximately linear.

Random Forest Classifier

A more advanced ensemble model made of many decision trees. It can handle complex, non-linear relationships in the data. Random Forest is often more accurate and robust than simple models because it reduces overfitting and improves generalization.

4. Results and Evaluation

The trained models were evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. These metrics help assess how well each model identifies malignant and benign tumors.

4.1 Logistic Regression Performance

Logistic Regression showed strong performance, achieving an accuracy of around 94–96%. It correctly classified most of the tumor samples and produced good precision and recall scores. Although it is a linear model, it still provided reliable predictions for this dataset.

4.2 Random Forest Performance

Random Forest performed even better, achieving an accuracy between 96–98%. It demonstrated superior ability to capture complex patterns and minimize misclassifications. It also showed stronger recall for malignant tumors, which is very important in medical diagnosis because false negatives can be dangerous.

4.3 Comparison of Models

Between the two models, Random Forest consistently achieved higher accuracy and more reliable predictions. Logistic Regression was faster and easier to interpret, but Random Forest provided higher diagnostic reliability. Therefore, Random Forest is considered the better model for this dataset.

5. Discussion

The project demonstrates how machine learning techniques can be effectively applied to real-world healthcare datasets. The performance of both models was strong, but Random Forest’s ability to handle non-linear relationships and reduce misclassification makes it more suitable for this task.

Early detection of breast cancer requires highly accurate prediction systems. A model that minimizes false negatives is critical because failing to detect a malignant tumor can delay treatment. Random Forest’s high sensitivity to malignant cases makes it a valuable choice.

6. Conclusion

This project successfully built and evaluated two supervised machine learning models for breast cancer classification. The Random Forest Classifier outperformed Logistic Regression and achieved near-perfect accuracy in some test runs.

Key conclusions:

The dataset was clean, well-organized, and well-suited for machine learning.

Preprocessing steps such as scaling and label encoding were essential.

Logistic Regression provided a solid baseline model.

Random Forest delivered the best performance and is the recommended model for this type of classification task.

Machine learning has great potential in medical diagnosis, and this project demonstrates a practical application of algorithms to support healthcare decision-making.

7. Future Enhancements

The project can be improved by:

Performing hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Using advanced models like SVM, XGBoost, or Neural Networks

Adding feature importance analysis to identify the most influential tumor characteristics

Deploying the model using a web application such as Flask or Streamlit

Using cross-validation for more reliable performance measurement

8. References

Kaggle: Breast Cancer Wisconsin Diagnostic Dataset

Scikit-Learn Documentation

UCI Machine Learning Repository
