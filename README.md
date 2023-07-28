As part of the process of preparing the data, we performed a variety of actions, we will explain and detail them here:
cleaning irrelevant columns:
When working with a data set, in many cases there are columns
or various characteristics that may not be relevant to the task at hand, in this case, diagnosis
intracranial hemorrhage. These irrelevant columns may contain information that does not contribute to the process
The classification may even introduce noise into the data. Removing such columns optimizes the dataset
and makes it more focused on the essential information needed to train the network (CNN).
Changing column names:
It is important to have clear and consistent names for the columns of the data set. On
By changing the names of the columns, we recognize that the data is well organized and easier to understand. This step
Also helps maintain data integrity and avoid confusion during data preparation steps
and the training.
Reading all the pictures of all the patients:
Since the goal of the project is to diagnose internal bleeding
Cranial using medical images, reading all available images of each patient is essential.
These images are used as the main input data for the CNN model. By reading all the pictures,
The researchers can build a comprehensive and diverse data set that captures the different cases of bleeding
Intracranial, its different types, and potentially different stages of the condition.
Deleting mapped images that do not belong:
In the process of preparing the data set, we wanted to ensure
Consistency in the data and avoid entering irrelevant data. We have removed images that are not relevant
for the target task of the model (diagnosis of intracranial bleeding). In doing so, we ensured that the paired dataset
(images with appropriate labels) clean and suitable for CNN training.
Creating a paired data set (image-label association):
in supervised learning tasks such as classification images, it is essential to have a data set in which each image is associated with its corresponding label (that is,
Does it represent an intracranial hemorrhage and its specific type if any (data set creation
Such a pair allows CNN to learn the connections between the images and their corresponding classes during training. This is the basis of the learning process, as the model adjusts its parameters
Based on the input-image pairs and the labels to make accurate predictions during inference.
The project is divided into two phases, one is the identification of whether there is bleeding and in which we applied the CNN model, and the second
Classified into a certain type of bleeding out of 5 possible.


deep learning approach:
Data Preprocessing:
The code starts by reading a CSV file (hemorrhage_diagnosis.csv) containing information about the patients and their corresponding images. 
The images are organized in subfolders for each patient. The code initializes lists to store images (images) and labels for hemorrhage diagnosis (labels_hemorrhage). 
It iterates over each row in the DataFrame to retrieve the image path, check if the image file exists, read the image, 
convert it to grayscale, resize it to the desired image_size (64x64 pixels), and then append the preprocessed image and hemorrhage label to the respective lists.
Normalization and Train-Test Split:
After preprocessing the data, the grayscale images are normalized by dividing each pixel value by 255.0 to scale the pixel values to the range [0, 1]. 
The dataset is then split into training and testing sets using an 80-20 split ratio. 80% of the images are used for training the model (train_images and train_labels_hemorrhage), 
while the remaining 20% are reserved for testing (test_images and test_labels_hemorrhage).
CNN Model for Hemorrhage Classification:
The convolutional neural network (CNN) model for hemorrhage classification is built using the Keras Sequential API. The model consists of the following layers:

a. Conv2D layers: Three Convolutional layers with 32, 64, and 128 filters, respectively, each using a 3x3 kernel and the ReLU activation function. 
These layers extract important features from the input images.

b. MaxPooling2D layers: Three MaxPooling layers with a 2x2 pooling window, which reduces the spatial dimensions and retains the most important features.

c. Flatten layer: Flattens the 2D feature maps into a 1D vector, preparing them for the fully connected (Dense) layers.

d. Dense layers: Two Dense layers, one with 64 neurons and the ReLU activation function and the other with a single neuron and the sigmoid activation function. 
The final Dense layer outputs the probability of hemorrhage presence (binary classification) based on the sigmoid activation function.

Optimization Method:
The optimization method used is the Adam optimizer. 
Adam stands for Adaptive Moment Estimation and is a popular optimization algorithm for training deep learning models. 
It combines ideas from RMSprop and Momentum optimization to adaptively adjust the learning rates of each parameter. 
Adam helps in achieving faster convergence and is well-suited for a wide range of deep learning tasks.

Loss Function:
The loss function used for training the model is 'binary_crossentropy'. 
This loss function is commonly used for binary classification problems,
where the output is a single value representing the probability of the positive class (in this case, hemorrhage present). 
The binary cross-entropy loss measures the difference between the predicted probabilities and the ground truth labels, 
penalizing larger errors more heavily.
After that, the model is trained on the training set for 35 epochs and batch_size=32 
In the second step we take the output of the convolutional network from the first step and we classify the
The type of bleeding according to Algorithm Means - K.
