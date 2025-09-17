## Deep_learning

# Titanic Dataset
Here's a breakdown of what my code does:
1.  **Load Data:** It begins by loading the well-known "titanic" dataset using the `seaborn` library.
2.  **Initial Exploration:** The notebook then performs a preliminary analysis of the data by:
    * Displaying the first 5 rows.
    * Showing the data types of each column.
    * Calculating and displaying the number of missing values for each column.
3.  **Separate Features and Target:** The "survived" column, which is the variable to be predicted, is separated from the rest of the features.
4.  **Identify Column Types:** The code programmatically identifies which columns are numerical (like `age`, `fare`) and which are categorical (like `sex`, `embarked`).
5.  **Create Preprocessing Pipelines:** It sets up two distinct pipelines using `scikit-learn` to handle the different data types:
    * **Numerical Pipeline:** For numerical features, it first fills any missing values using the median of the column and then scales the data using `StandardScaler`.
    * **Categorical Pipeline:** For categorical features, it fills missing values with the most frequent value in the column and then converts the categories into a numerical format using `OneHotEncoder`.
6.  **Combine Pipelines:** A `ColumnTransformer` is used to apply the correct pipeline to the corresponding columns of the dataset.
7.  **Split Data:** The dataset is split into a training set (80%) and a testing set (20%).
8.  **Apply Preprocessing:** The combined preprocessing steps are applied to both the training and testing sets. The number of features increases from 14 to 28 after this step, mainly due to the `OneHotEncoder` creating new columns for each category.
9.  **Convert to Tensors:** Finally, the preprocessed data (which is in NumPy format) is converted into `TensorFlow` tensors, making it ready to be fed into a neural network for training.
This focuses exclusively on the crucial steps of cleaning, transforming, and preparing a real-world dataset before building and training a model.
