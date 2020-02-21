#!/usr/bin/env python
# coding: utf-8

# # Image Classification to find   PavBhaji's image or Not .
# * I have to find the given image is pav bhaji or not 
# * Name "1" folder contain images of pav bhaji
# * Naem "0" folder contain images of non pav bhaji

# ### Actual  Image of pav bhaji
# ![](https://www.nicepng.com/png/detail/256-2564363_keemapavse-pav-bhaji-hd-png.png)

# In[1]:


### Import all necessary lib


# In[2]:


import pandas as pd
 
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ### cv2 -used  how to read an image, how to display it and how to save it back
# Read an image
# Use the function cv2.imread() to read an image. The image should be in the working directory or a full path of image should be given.
# 
# Second argument is a flag which specifies the way image should be read.
# 
# * cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
# * cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
# * cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel

# * cv2 lib convert  image to some matrix form according to the HSV which are ranges between 0 to 255 

# ### Libraries Load for training our model 
# * using Machine Laerning 
# * Using Deep Learning (ANN,CNN)
#     * ANN- Artificial Neural Networks (ANN) are multi-layer fully-connected neural nets .They consist of an input layer, multiple hidden layers, and an output layer.
#     * CNN - Convolutional Neural Networks (CNN) is one of the variants of neural networks used heavily in the field of Computer Vision.The hidden layers of a CNN typically consist of convolutional layers, pooling layers, fully connected layers, and normalization layers. 

# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier



from keras.layers.recurrent import LSTM,SimpleRNN
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# #### Read json file using pandas.read_json()

# In[4]:


data = pd.read_json('pavbhaji.json',orient='slice')


# In[5]:


data


# ####  print first row of URLs columns in Json file

# In[6]:


data['display_url'][0]


# * URL signature expired message deliever when search this link

# #### find text present in json file

# In[7]:



data['edge_media_to_caption'][0]


# #### Read first image of '1' folder (pav bhaji image) using cv2.imread() method

# In[8]:


img = cv2.imread('/home/psspl_sarfaraz/environments/ml_jupyter/Tensorflow_practice_book/dataset (1)/1/16228666_180901469054785_6854217108004274176_n.jpg', cv2.IMREAD_UNCHANGED)
 


# #### Print no of rows and columns of first image

# In[9]:


print('Original Dimensions : ',img.shape)


# #### Scaling and Resized of the Original image

# In[10]:


scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)


# #### Display an image
# * Use the function cv2.imshow() to display an image in a window. The window automatically fits to the image size.
# 
# * First argument is a window name which is a string. second argument is our image. You can create as many windows as you wish, but with different window names.

# In[11]:


# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[12]:


# cv2.imshow("Original image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #### Globe 
# * In Python, the glob module is used to retrieve files/pathnames matching a specified pattern. 
# * With glob, we can also use wildcards ("*, ?, [ranges]) apart from exact string search to make path retrieval more simple and convenient.

# In[13]:


import glob


# #### Retrieve all images from folder '1' (pav bhaji's image) then read image using cv2 and store this  in images variable

# In[14]:


images = [cv2.imread(file) for file in glob.glob("/home/psspl_sarfaraz/environments/ml_jupyter/Tensorflow_practice_book/dataset (1)/1/*.jpg")]


# In[15]:


#### Total images in folder '1'


# In[16]:


np.array(images).shape


# #### Retrieve all images from folder '0' (others images) then read image using cv2 and store this  in images_2 variable

# In[17]:


images_2 = [cv2.imread(file) for file in glob.glob("/home/psspl_sarfaraz/environments/ml_jupyter/Tensorflow_practice_book/dataset (1)/0/*.jpg")]


# #### Total images in folder '0'

# In[18]:


np.array(images_2).shape


# ######### Pav Bhaji data

# * Create empty list name mera_dat ,
# * In data '1' consist 183 images , 
# * used for loop resize image and store in that empty list  using append() method

# In[19]:


mera_dat = []

for i in range(183):
    desired_size = 368
    
    im = images[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/Rahul/Desktop/a.jpg'.format(i), new_im) 
    mera_dat.append(new_im)



# In[20]:


# print first rows of images data


# In[21]:


mera_dat[:1]


# ############### Not a  Pav Bhaji data

# * Create empty list name mera_dat ,
# * In data '0' consist 269 images , 
# * used for loop upto max range 269 resize image and store in that empty list  using append() method

# In[22]:


mera_dat_2 = []

for i in range(269):
    desired_size = 368
    
    im = images_2[i]
    old_size = im.shape[:2] # old_size is in (height, width) format
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
#    cv2.imshow("image", new_im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imwrite('C:/Users/Rahul/Desktop/a.jpg'.format(i), new_im) 
    mera_dat_2.append(new_im)


# #### print first rows of non pav bhaji image

# In[23]:


mera_dat_2[:1]


# ###  Transform list to  Numpy array and Image Reshaping

# In[24]:


arr = np.array(mera_dat)
arr = arr.reshape((183, 406272))


# In[25]:


arr


# In[26]:



ar1 = np.array(mera_dat_2)
ar1 = ar1.reshape((269, 406272))


# ### Normalize image

# In[27]:


arr = arr / 255
ar1 = ar1 / 255


# ### PCA- Principal Component Analysis
# * PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.
# * As our data has  very hugh  features, so i am  using PCA  technique as it reduce our dimensions upto 16 as i mention, most of the  time it is very useful for very large features columns 

# In[28]:


dataset = pd.DataFrame(arr)
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
dataset_pca = pca.fit_transform(dataset)
dataset_pca = pd.DataFrame(dataset_pca,columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16'])

#create new columns name label or target to store '1' value as define pav bhaji image
dataset_pca['label'] = np.ones(183)


# In[29]:


dataset_pca.head()


# In[30]:


dataset_2 = pd.DataFrame(ar1)
pca = PCA(n_components=16)
dataset_pca2 = pca.fit_transform(dataset_2)
dataset_pca2 = pd.DataFrame(dataset_pca2,columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16'])

#create new columns name label or target to store '0' value as define other  images 
dataset_pca2['label'] = np.zeros(269)


# In[31]:


dataset_pca2.head()


# In[32]:


#### Combine both dataframe pav bhaji' dataset(1) and other image dataset(0) after using pca


# In[33]:


dataset_master = pd.concat([dataset_pca, dataset_pca2])


# In[34]:


### print top 5 rows of combine dataframe 


# In[35]:


dataset_master.head()


# ### Labelling the data (i.e. X_label- contain features columns, y-label- contain target columns)

# In[36]:


X = dataset_master.iloc[:, 0:16].values
y = dataset_master.iloc[:, -1].values


# In[37]:


#### No of rws and col in X


# In[38]:


X.shape


# ## Visualization

# #### Find no of data consist of pav bhaji image or non pav-bhaji image

# In[39]:


# extracting the number of examples of each class
Real_len = dataset_master[dataset_master['label'] == 1].shape[0]
Not_len = dataset_master[dataset_master['label'] == 0].shape[0]


# In[40]:


# bar plot of the 2 classes
plt.rcParams['figure.figsize'] = (7, 5)
plt.bar(10,Real_len,3, label="Real pav bhaji", color='blue')
plt.bar(15,Not_len,3, label="Not real", color='red')
plt.legend()
plt.ylabel('Number of examples')
plt.title('Propertion of examples')
plt.show()


# #### Scatter diag 

# In[41]:


# giving a larger plot 
plt.figure(figsize =(8, 6)) 
  
plt.scatter( dataset_master['f1'],dataset_master['f2'], c = y, cmap ='plasma') 
  
# labeling x and y axes 
plt.xlabel('First Principal Component') 
plt.ylabel('Second Principal Component')
plt.legend(['pca1','pca2'])


# #### Histogram 

# In[42]:


dataset_master.hist()
plt.title("Ranges and frequecies of our data")


# #### Correlation betwwen features columns and target(label)

# In[43]:


dataset_master.corr()


# #### Correlation diag using seaborn.heatmap
# * positive correlation
# * negative correlation
# * no correlation

# In[44]:


plt.figure(figsize=[10,5])
import seaborn as sns
sns.heatmap(dataset_master.corr(),annot=True)


# ### Range of data

# In[45]:


sns.violinplot(dataset_master)


# ### Find any outlier present in our data

# In[46]:


plt.boxplot(dataset_master['f1'])
cols = dataset_master.columns


# In[47]:


plt.figure(figsize=[10,5])
plt.boxplot(dataset_master.iloc[:16])


# #### Splitting Training and Testing data

# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                                                random_state=40)


# In[49]:


from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth =2)
dtf.fit(X_train, y_train)


# In[50]:


dtf.score(X_train, y_train)


# In[51]:


dtf.score(X_test,y_test)


# ### Training our model using Other  Classifier
# * Fitting Machine learning  algo and compare there training and testing score
# * To check Data underfitting or Overfitting the model by comparing there score.
#      * In underfitting condition- model accuracy is very low.
#      * In Overfitting condtion- training  accuracy is high but testing accuracy is exponentially low as compare to training. 
# * Chose the best algo suited for our model
# * Find Accuracy score , confusion matrix, classsification report

# In[52]:


lr = LogisticRegression()
rfc =RandomForestClassifier(n_estimators=10,max_depth=20)
lsvc = LinearSVC(C=0.01)
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=35)


# In[53]:


ml_model = []
ml_model.append(("LogisticRegression",lr))
ml_model.append(('RandomForestClassifier',rfc))
ml_model.append(('LinearSVC',lsvc))
ml_model.append(('GaussianNB',gnb))
ml_model.append(('KNN',knn))


# In[54]:


for name, algo in ml_model:
    algo.fit(X_train,y_train)
    train_score=algo.score(X_train,y_train)
    test_score = algo.score(X_test,y_test)
    msg = "%s = (training score): %f (testing score:) %f"%(name,train_score,test_score)
    print(msg)


# ## Deep Learning usin ANN

# In[55]:


# Check the shape of training data
X_train.shape


# In[56]:


model = Sequential()
model.add(Dense(1000,activation='elu',input_shape = (16,)))
model.add(Dense(128,activation='elu'))
model.add(Dense(100,activation='elu'))
model.add(Dense(10,activation='softmax'))


# In[57]:


model.summary()


# In[58]:


model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics =['accuracy'])


# In[59]:


history = model.fit(X_train,y_train,epochs=1000)


# ### Visualize the training loss and training validation accuracy to see if the model is overfitting
# * Accuracy for training dataset = 100%
# * Loss reduce upto =1.1128e-07

# In[60]:


pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim()
plt.show()


# In[87]:


test_acc =model.evaluate(X_test,y_test)


# In[88]:


print("Loss and Accuracy of testing data =",test_acc)


# In[89]:


predict = model.predict_classes(X_test)


# In[90]:


predict


# In[91]:


y_test


# ### Confusion Matrix define-:
# * True Negative
# * False Positive
# * False Negative
# * True Positive
#         * All diagonal element are correct prediction rest are incorrect 
#         * Sum of all correct prediction(diagonal data) divided by sum of all the data(correct or wrong prediction data)

# In[66]:


from sklearn import metrics
metrics.confusion_matrix(y_test,predict)


# #### Define 
# * precision-:precision is the fraction of relevant instances among the retrieved instances
# * recall:-while recall is the fraction of the total amount of relevant instances that were actually retrieved
# * f1-score-:F1 is an overall measure of a model’s accuracy that combines precision and recall, in that weird way that addition and multiplication just mix two ingredients to make a separate dish altogether

# In[67]:


print(metrics.classification_report(y_test,predict))


# ### Implement CNN using Keras

# ## Convolutional Neural Network
# * CNN is used for image classification, object detection
# 
# * ![](https://preview.ibb.co/nRkBpp/gec2.jpg)

# ### What is Convolution Operation?
# * We have some image and feature detector(3*3)
# * Feature detector does not need to be 3 by 3 matrix. It can be 5 by 5 or 7 by 7.
# * Feature detector = kernel = filter
# * Feauture detector detects features like edges or convex shapes. Example, if out input is dog, feature detector can detect features like ear or tail of the dog.
# * feature map = conv(input image, feature detector). Element wise multiplication of matrices.
# * feature map = convolved feature
# * Stride = navigating in input image.
# * We reduce the size of image. This is important bc code runs faster. However, we lost information.
# * We create multiple feature maps bc we use multiple feature detectors(filters).
# * Lets look at gimp. Edge detect: [0,10,0],[10,-4,10],[0,10,0]

# ![](https://image.ibb.co/m4FQC9/gec.jpg)
# 
# * After having convolution layer we use ReLU to break up linearity. Increase nonlinearity. Because images are non linear.
# ### Same Padding
# 
# 
# * As we keep applying conv layers, the size of the volume will decrease faster than we would like. In the early layers of our network, we want to preserve as much information about the original input volume so that we can extract those low level features.
# * input size and output size are same.
# ![](https://preview.ibb.co/noH5Up/padding.jpg)

# ### Max Pooling
# * It makes down-sampling or sub-sampling (Reduces the number of parameters)
# * It makes the detection of features invariant to scale or orientation changes.
# * It reduce the amount of parameters and computation in the network, and hence to also control overfitting.
# * ![](https://preview.ibb.co/gsNYFU/maxpool.jpg)

# ### Flattening
# ![](https://image.ibb.co/c7eVvU/flattenigng.jpg)

# ### Full Connection
# * Neurons in a fully connected layer have connections to all activations in the previous layer
# * Artificial Neural Network

# ![](https://preview.ibb.co/evzsAU/fullyc.jpg)

# ### Implement Deep Learning  using Keras
# * conv => max pool => dropout => conv => max pool => dropout => fully connected (2 layer)
# * Dropout: Dropout is a technique where randomly selected neurons are ignored during training

# In[68]:


X_train.shape


# In[69]:


X_test.shape


# #### Reshaping 2_dim to 4_dim so that our convolutinal 2D layer can work 

# In[70]:


X_train = X_train.reshape(361,4,4,1)
X_test = X_test.reshape(91,4,4,1)


# In[93]:


X = X.reshape(452,4,4,1)


# In[71]:


X_train.shape


# #### Define the Model

# In[72]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (4,4,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# ### Set the optimizer and annealer
# Once our layers are added to the model, we need to set up a score function, a loss function and an optimisation algorithm.
# 
# We define the loss function to measure how poorly our model performs on images with known labels. It is the error rate between the oberved labels and the predicted ones. We use a specific form for categorical classifications (>2 classes) called the "categorical_crossentropy".
# 
# The most important function is the optimizer. This function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss.
# 
# I choosed RMSprop (with default values), it is a very effective optimizer. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. We could also have used Stochastic Gradient Descent ('sgd') optimizer, but it is slower than RMSprop.
# 
# The metric function "accuracy" is used is to evaluate the performance our model. This metric function is similar to the loss function, except that the results from the metric evaluation are not used when training the model (only for evaluation).

# In[73]:


# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[74]:


# Compile the model
model.compile(optimizer = optimizer , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])


# In[75]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[76]:


epochs = 1000
batch_size = 100


# In[77]:


history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, y_test), verbose = 2)


# ### Visualize the training loss and training validation accuracy to see if the model is overfitting
# * Accuracy for training dataset = 100%
# * Loss reduce upto =1.4067e-07 
# * Val_accuracy = 0.7802
# * val_loss = 7.9796 

# In[78]:


pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim()
plt.show()


# ### Accuracy  and loss of all 452 dataset (Training + Testing)

# In[94]:


model.evaluate(X,y)


# * Our accuracy is pretty well for all data sample  which is 95.57% hence our model can work on new data 

# ### predict test data

# In[97]:


prediction =  model.predict_classes(X)


# ### Confusion Matrix define-:
# * True Negative
# * False Positive
# * False Negative
# * True Positive
#         * All diagonal element are correct prediction rest are incorrect 
#         * Sum of all correct prediction(diagonal data) divided by sum of all the data(correct or wrong prediction data)

# In[105]:


from sklearn import metrics
metrics.confusion_matrix(y,prediction)


# #### Define 
# * precision-:precision is the fraction of relevant instances among the retrieved instances
# * recall:-while recall is the fraction of the total amount of relevant instances that were actually retrieved
# * f1-score-:F1 is an overall measure of a model’s accuracy that combines precision and recall, in that weird way that addition and multiplication just mix two ingredients to make a separate dish altogether

# In[106]:


print(metrics.classification_report(y,prediction))


# In[107]:


### convert priction array to dataframe formate


# In[108]:


prediction = pd.DataFrame(prediction,columns=['Predict_Pav_Bhaji'])


# In[109]:


# Top 5 rows of prediction
prediction.head()


# ### Our final prediction submit using csv format

# In[110]:


prediction.to_csv("sample_prediction.csv")


# In[ ]:




