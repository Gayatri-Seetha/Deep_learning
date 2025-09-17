# Deep_learning

## Titanic Dataset
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


## Image Classification
The goal is to classify images into 10 different categories of recyclable waste, such as "aluminum_can," "plastic_bottle," and "cardboard." It cleverly uses the standard **CIFAR-10** image dataset but assigns it new, custom labels related to recycling.
**1. Data Preparation**
* **Loading Data**: The notebook loads the built-in CIFAR-10 dataset, which consists of 50,000 training images and 10,000 test images, each being 32x32 pixels in color (`32, 32, 3`).
* **Normalization**: The pixel values of the images (originally from 0 to 255) are scaled down to a range between 0 and 1 by dividing by 255.0. This is a standard practice that helps the model train more effectively.
* **One-Hot Encoding**: The integer labels (0 through 9) are converted into a binary format (one-hot encoding) using `to_categorical`. This is necessary for training with the `categorical_crossentropy` loss function. 
**2. Model Architecture**
* A **Convolutional Neural Network (CNN)** is defined using a `Sequential` model in TensorFlow Keras.
* The architecture consists of:
    * Two blocks of `Conv2D` (convolutional), `BatchNormalization`, and `MaxPooling2D` layers to extract features from the images.
    * A `Flatten` layer to convert the 2D feature maps into a 1D vector.
    * A `Dense` (fully connected) layer with 256 neurons and a `relu` activation function.
    * A `Dropout` layer (with a rate of 0.5) is included to help prevent overfitting.
    * The final `Dense` output layer has 10 neurons (one for each class) with a `softmax` activation function to produce a probability distribution across the categories.
**3. Training and Evaluation**
* **Compilation**: The model is compiled with the `adam` optimizer, `categorical_crossentropy` loss function, and is set to track `accuracy` as its performance metric.
* **Training**: The model is trained for a maximum of 30 epochs. It uses an **EarlyStopping** callback, which monitors the validation accuracy and stops the training process if it doesn't improve for 5 consecutive epochs. This prevents overfitting and saves computation time. The training actually stopped early at **epoch 14**.
* **Results**: After training, the model achieves a final **test accuracy of approximately 72.03%** on the unseen test data.
**4. Performance Analysis**
* **Classification Report**: A detailed report is generated showing the **precision, recall, and f1-score** for each of the 10 waste categories. This shows that the model performs better on some classes (like "plastic\_bottle" and "e\_waste") than others ("paper\_carton").
* **Confusion Matrix**: A confusion matrix is plotted to visualize which categories the model tends to confuse with one another.
* **Learning Curves**: The notebook concludes by plotting the **accuracy and loss curves** for both the training and validation sets over the epochs. This helps to visually assess the model's learning progress and identify signs of overfitting.

It's example of a deep learning image classification project, demonstrating data preprocessing, model building, training, and detailed performance evaluation.
