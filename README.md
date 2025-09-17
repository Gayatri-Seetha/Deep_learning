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


# Image Classification
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

# Sentiment Analysis
 **1. Data Loading and Preparation**
  * **Load IMDb Dataset**: The notebook starts by loading the pre-processed IMDb movie review dataset directly from Keras. The data is already tokenized, meaning each word is represented by an integer. It limits the vocabulary to the 10,000 most common words.
  * **Inspect Data**: It shows that the reviews are sequences of numbers (e.g., `[1, 14, 22, 16, ...]`) and then decodes one review back into plain English to demonstrate what the data represents ("*\<START\> this film was just brilliant...*").
  * **Padding**: Since neural networks require inputs of a uniform size, and reviews have varying lengths, all sequences are **padded** (or truncated) to a fixed length of 200 words using `pad_sequences`. Shorter reviews get zeros added, and longer ones are cut short.
**2. Model Architecture (LSTM Network)**
The notebook defines a simple but powerful RNN using a `Sequential` model from Keras. The architecture consists of three key layers:
1.  **Embedding Layer**: This is the first layer. It takes the integer-encoded vocabulary and learns a dense vector representation (an "embedding") for each word. In this case, each of the 10,000 words is mapped to a 32-dimensional vector. This allows the model to capture semantic relationships between words.
2.  **LSTM Layer**: The core of the model is a **Long Short-Term Memory (LSTM)** layer with 64 units. LSTMs are a special type of RNN excellent at learning patterns from sequential data, like the order of words in a sentence, which is crucial for understanding context and sentiment.
3.  **Dense Output Layer**: The final layer is a single `Dense` neuron with a **sigmoid** activation function. This squashes the output to a value between 0 and 1, representing the probability that the review is positive.

**3. Training and Evaluation**

  * **Compilation**: The model is compiled using the `adam` optimizer and the `binary_crossentropy` loss function, which is standard for a two-class (binary) classification problem.
  * **Training**: The model is trained for **5 epochs** on the padded training data. The output logs show the training and validation accuracy improving with each epoch.
  * **Visualization**: After training, it plots the **accuracy and loss curves** for both the training and validation sets to visualize the model's performance over time and check for overfitting.
  * **Final Result**: The model is evaluated on the unseen test data and achieves a final **test accuracy of approximately 84.5%**. 👍👎

In summary,it is a complete and practical demonstration of how to use an LSTM network for a text classification task, covering all the essential steps from data preparation to final evaluation.
