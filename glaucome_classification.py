#Optic

#Import Packages and functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model
import os



#STEP 1 : DATA EXPLORATION


#Part 1 : df

#Load Data 
df = pd.read_csv('full_df.csv')
print(f"df.shape is {df.shape}")
   
#Columns  
df.columns
            #Identification (ID)
            #Age (Patient Age)
            #Sex (Patient Sex)
            
            #Left eye (Left-Fundus)
            #Right eye (Right-Fundus)
            #Left-Diagnostic Keywords (Left-Diagnostic Keywords)
            #Right-Diagnostic Keywords (Right-Diagnostic Keywords)
            
            #Normal (N)
            #Diabetes (D)
            #Glaucoma (G)
            #Cataract (C)
            #Age related Macular Degeneration (A)
            #Hypertension (H)
            #Pathological Myopia (M)
            #Other diseases/abnormalities (O)
            
            #Filepath (filepath)
            #Final label (labels)
            #Target (target)
            #Filename (filename)

#Explore first fives rows ==> df values
df_head = df.head()

# Look at the data type of each column and whether null values are present
df.info()


#Part 2 : X and Y
    
#load X set
X = df[['ID','Patient Age','Patient Sex' ,'filename']]
#Explore first fives rows ==> X values
X_head = X.head()
print(f"X.shape is {X.shape}")

#load Y set
Y = df[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]
#Explore first fives rows ==> Y values
Y_head = Y.head()
print(f"Y.shape is {Y.shape}")


#Part 3 : Y distribution
    
#The labels in our dataset
class_labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

#Extract the labels (Y)
y = Y[class_labels].values

#Histogram of the number of samples for each label in the dataset:
plt.xticks(rotation=90)
plt.bar(x = class_labels, height= y.sum(axis=0))

#Create a list of the names of each patient condition or disease
columns = Y.keys()
columns = list(columns)
print(columns)

# Print out the number of positive labels for each class
for column in columns:
    print(f"The class {column} has {Y[column].sum()} samples")  
        
        
#Part 4 : Image Vizualisation
    
# Extract numpy values from Image column in data frame
images = df['filename'].values

# Extract 9 random images from it
random_images = [np.random.choice(images) for i in range(9)]

# Location of the image dir
img_dir = 'ODIR-5K/ODIR-5K/Training Images/'

print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(20,10))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()
    
    
#Part 5 : Get the first image that was listed in the df
sample_img = df.filename[0]
raw_image = plt.imread(os.path.join(img_dir, sample_img))
plt.imshow(raw_image, cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')
print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")
  
      
#Part 6 : Plot up the distribution of pixel values in the image
sns.distplot(raw_image.ravel(),label=f'Pixel Mean {np.mean(raw_image):.4f} & Standard Deviation {np.std(raw_image):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
 
              
#Part 7 : Sex and Age distribution
    
#Histogramme gender
X['Patient Sex'].hist()
#Replace Female by 0 and Male by 1
df['Patient Sex'] = df['Patient Sex'].replace(['Female','Male'],[0,1])
#Histogramme gender
X['Patient Age'].hist()
 

    
#STEP 2 : TEST TRAIN AND VALID SETS
 
    
#Part 1 : Separated in train and test set
    
#Separation
from sklearn.model_selection import train_test_split    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, random_state=0)

#X_train and Y _train shape
print(f"X_train.shape is {X_train.shape}")
print(f"Y_train.shape is {X_train.shape}")

#X_test and Y _test shape
print(f"X_test.shape is {X_test.shape}")
print(f"Y_test.shape is {X_test.shape}")

#Sex distribution
X_train['Patient Sex'].hist()
X_test['Patient Sex'].hist()

#Age distribution
X_train['Patient Age'].hist()
X_test['Patient Age'].hist()
  
       
#Part 2 : Separated in valid and train set
    
#Separation
from sklearn.model_selection import train_test_split    
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, train_size=0.85, random_state=0)

#X_train and Y _train shape
print(f"X_train.shape is {X_train.shape}")
print(f"Y_train.shape is {X_train.shape}")

#X_test and Y _test shape
print(f"X_valid.shape is {X_valid.shape}")
print(f"y_valid.shape is {X_valid.shape}")

#Sex distribution
X_train['Patient Sex'].hist()
X_test['Patient Sex'].hist()

#Age distribution
X_train['Patient Age'].hist()
X_test['Patient Age'].hist()

#Concatened 
XY_train = pd.concat([X_train, Y_train], axis=1, sort=False)
XY_test = pd.concat([X_test, Y_test], axis=1, sort=False)
XY_valid = pd.concat([X_valid, Y_valid], axis=1, sort=False)
 
        
#Part 3 : Image Preprocessing in Keras
    
# Import data generator from keras
from keras.preprocessing.image import ImageDataGenerator

# Normalize images
image_generator = ImageDataGenerator(
    samplewise_center=True, #Set each sample mean to 0.
    samplewise_std_normalization= True # Divide each input by its standard deviation
)

# Flow from directory with specified batch size and target image size
generator = image_generator.flow_from_dataframe(
        dataframe= XY_train,
        directory="ODIR-5K/ODIR-5K/Training Images/",
        x_col='filename', # features
        y_col= ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'], # labels  
        class_mode="raw",
        batch_size= 1, # images per batch
        shuffle=False, # shuffle the rows or not
        target_size=(320,320) # width and height of output image
)

# Plot a processed image
sns.set_style("white")
generated_image, label = generator.__getitem__(0)
plt.imshow(generated_image[0], cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')
print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
 
   
#Part 4 :Comparison of the distribution of pixel values in the new pre-processed image versus the raw image 
    
# Include a histogram of the distribution of the pixels
sns.set()
plt.figure(figsize=(10, 7))

# Plot histogram for original iamge
sns.distplot(raw_image.ravel(), 
             label=f'Original Image: mean {float(np.mean(raw_image)):.4f} - Standard Deviation {float(np.std(raw_image)):.4f} \n '
             f'Min pixel value {float(np.min(raw_image)):.4} - Max pixel value {float(np.max(raw_image)):.4}',
             color='blue', 
             kde=False)

# Plot histogram for generated image
sns.distplot(generated_image[0].ravel(), 
             label=f'Generated Image: mean {float(np.mean(generated_image[0])):.4f} - Standard Deviation {float(np.std(generated_image[0])):.4f} \n'
             f'Min pixel value {float(np.min(generated_image[0])):.4} - Max pixel value {float(np.max(generated_image[0])):.4}', 
             color='red', 
             kde=False)

# Place legends
plt.legend()
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixel')    


               
#STEP 3 : TRAIN, TEST AND VALID SET PREPOCESSING
 
    
#Part 1 : Build a generator for train set
def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))

    return generator


#Part 2 : Build a separate generator for valid and test sets
def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col=x_col, 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    
    return valid_generator, test_generator


#Part 3 : Make one generator for our training data and one each of our test and validation datasets.
IMAGE_DIR = "ODIR-5K/ODIR-5K/Training Images/"
labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
train_generator = get_train_generator(XY_train, IMAGE_DIR, 'filename', labels)
valid_generator, test_generator= get_test_and_valid_generator(XY_valid, XY_test, XY_train, IMAGE_DIR, 'filename', labels)



#STEP 4 : MODEL DEVELOPMENT
    
    
#Part 1 : Class Imbalance
    
    #Addressing Class Imbalance
plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()

    #Computing Class Frequencies
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    
    # total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies

    #Compute frequencies for our training data.
freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos

#Visualize these two contribution ratios next to each other for each of the pathologies
data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)
 

#Part 2 : Balanced class
    
#Balancing the contribution of positive and negative label
pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights

#Verify this by graphing the two contributions next to each other again
data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

#Weighted Loss
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss_pos = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += loss_pos + loss_neg
        
        return loss
    
        
    return weighted_loss
    
#Part 3 : Use a pre-trained DenseNet121 model 

#Create the base pre-trained model
base_model = DenseNet121(weights='imagenet', include_top=False)

x = base_model.output

#Add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

#Add a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x) 
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))

    
    
#STEP 5 : MODEL TRAINING
    
#Training the model
history = model.fit_generator(train_generator, 
                          validation_data=valid_generator,
                          steps_per_epoch=100, 
                          validation_steps=25, 
                          epochs = 3)

#Plot Training Loss Curve
plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()
  
#Plot Training Accuracy Curve
plt.plot(history.history['accuracy'])
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title("Training Accuracy Curve")
plt.show()

    
    