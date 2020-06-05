# FDA  Submission

**Your Name:** Marcelo G.

**Name of your Device:** Pneumonia assistant

## Algorithm Description 

### 1. General Information

**Intended Use Statement:** Assisting a radiologist with classifying pneumonia disease through X-ray tests. 

**Indications for Use:** indicated for assisting in screening detection of Pneumonia through X-rays studies in people of `ages 20-65`. Not indicated for use in people with other diseases with the exception of `Edema` and `Infiltration`. 

**Device Limitations:** The algorithm performs very poorly on the accurate detection of pneumonia in the presence of `Atelectasis`

**Clinical Impact of Performance:** Predicting patients with pneumonia as healthy should not be acceptable. This algorithm is designed to minimize FN. Hence would be of great use for screening assistance. 

### 2. Algorithm Design and Function


**DICOM Checking Steps:**

The checking steps performed in the DICOM images were 3. However, we took more to print a status of the DICOM image:

* Patient ID
* Patient position or Image position which could be AP or PA

<img src='https://medicforyou.in/wp-content/uploads/2015/09/AP-vs-PA-view-of-Chest-Xray.jpg' width=400px>

* Patient Age
* Body part examined. Must be `Chest`
* Image Type. Must be `DX` which is an X-ray image.
* Study description (label)

Examples:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure13.JPG?raw=1' width=400px>


**Preprocessing Steps:**

The preprocessing steps were the following:
* Normalization of data to be between 0 and 1.
* Then to train the network some augmentation was performed.

To perform image augmentation we shall have in mind that not all augmentation techniques are adequate for images in the healthcare field.


<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure1.jpg?raw=1' width=600px>

From the above images, we can say that:
* Vertical flip is not suitable at all since we will never get upside down images
* Horizontal flip is feasible since the image could be taken in a different perspective AP vs PA
* Shift can be presented but in a low percentage
* Rotation only make sense up to 20 degrees
* Shear and zoom can also be presented in a low percentage

Therefore, considering these factors we selected certain parameters for each technique that can be seen in the **Parameters** section below.

**CNN Architecture:**
I decided to apply transfer learning to a VGG16 pre-trained network with a final architecture as shown below:
<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure3.jpg?raw=1' width=800px>

### 3. Algorithm Training

**Parameters:**
* Augmentation parameters:
    * horizontal_flip=True,
    * vertical_flip=False,
    * height_shift_range=0.1,
    * width_shift_range=0.1,
    * rotation_range=15,
    * shear_range=0.1,
    * zoom_range=0.05   
* Batch size:
    * Training = 16
    * Validation = 32
* Optimizer:
    * Adam
    * learning rate = 5e-6
        * decaying rate per epoch: `0.5`
* Layers of pre-existing architecture that were frozen:
    * Layer Name: input_1, Trainable? False
    * Layer Name: block1_conv1, Trainable? False
    * Layer Name: block1_conv2, Trainable? False
    * Layer Name: block1_pool, Trainable? False
    * Layer Name: block2_conv1, Trainable? False
    * Layer Name: block2_conv2, Trainable? False
    * Layer Name: block2_pool, Trainable? False
    * Layer Name: block3_conv1, Trainable? False
    * Layer Name: block3_conv2, Trainable? False
    * Layer Name: block3_conv3, Trainable? False
    * Layer Name: block3_pool, Trainable? False
    * Layer Name: block4_conv1, Trainable? False
    * Layer Name: block4_conv2, Trainable? False
    * Layer Name: block4_conv3, Trainable? False
    * Layer Name: block4_pool, Trainable? False
    * Layer Name: block5_conv1, Trainable? False
    * Layer Name: block5_conv2, Trainable? False
* Layers of pre-existing architecture that were fine-tuned
    * Layer Name: block5_conv3, Trainable? True
    * Layer Name: block5_pool, Trainable? True
* Layers added to pre-existing architecture
    * add(Flatten())
    * add(Dropout(dropout_prob))
    * add(Activation("relu"))
    * add(Dense(1000))
    * add(Activation("relu"))
    * add(Dropout(dropout_prob))
    * add(Dense(1))
    * add(Activation("sigmoid"))

#### Algorithm training performance visualization 
<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure5.JPG?raw=1' width=800px>

#### ROC

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure6.JPG?raw=1' width=800px>

#### P-R curve

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure7.JPG?raw=1' width=800px>

**Final Threshold and Explanation:**

For pneumonia detection, it is crucial that we find all the patients that have pneumonia. Predicting patients with pneumonia as healthy is not acceptable. In other words, having a high FN score is not acceptable. This is measured by recall. Therefore, my approach will be to weight recall over precision. For that reason `F2-score` will be taken.

Therefore, We iterated through all thresholds in the P-R curve and select the one that maximizes the F2-score. Which in our case was: 
* `Best f2 score = 0.6837606837606838, Best threshold = 0.6507165431976318`

### 4. Databases

**Description of Training Dataset:** 

For the Training dataset, we apply image augmentation as data for Pneumonia was very low as shown below:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure8.JPG?raw=1' width=600px>

Initial data distribution was as follows:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure9.JPG?raw=1' width=600px>

Since data was very uneven I decided to cut random samples of negative cases to get a distribution of 50% positive and 50% negative cases as shown below:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure10.JPG?raw=1' width=600px>

A good practice is to perform random spot checks of the data by choosing several random images and visualizing them. Hence, I took random images for Pneumonia and plot their intensity histogram to get an intuition of the data, as shown below:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure4.JPG?raw=1' width=800px>

Comparing them to `No findings` or negative cases below:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure14.JPG?raw=1' width=600px>

I concluded that based on intensity histograms for `No finding` and `Pneumonia` cases, we can say that images seem to have peals at lower intensity values while No findings at higher ones. The reason might be that presence of infiltration in lungs with `Pneumonia` have sort of a lower intensity tendency than `No Pneumonia` cases

Also another check I did was to compare Pneumonia against the top 1 comorbidities related to Pneumonia, which in this case turned to be Infiltration as shown below:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure15.JPG?raw=1' width=600px>

The intensity histrogram values for infiltration were:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure16.JPG?raw=1' width=600px>


**Conclusion** Since Infiltration is very related to Pneumonia as seen in the histograms of comorbidities above, we will compare both of them as well. Based on intensity values we can conclude that infiltration intensity histograms have peaks at similar intensity levels. The reason could be that as Pneumonia thrives, the presence of infiltration is more a more notorious, therefore the peak is even more notorious in the same intensity values. This 2 diseases seem to be very related.

Besides the mean of infiltration tend to be very close to the Pneumonia cases.

**Description of Validation Dataset:** 

For the Validation dataset we haven't performed augmentation but we do apply normalization. As in the training, dataset data was very unbalanced as shown below

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure11.JPG?raw=1' width=600px>

Therefore I decided to rearrange it to 75% negative cases and 25% positive cases ratio, as it is more similar to real-life distribution. The illustration is shown below:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure12.JPG?raw=1' width=600px>

#### Some other considerations in the EDA were the basic demographics

* AGE: for age we found that most of the data was between 25 and 65, as shown below:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure17.JPG?raw=1' width=600px>

* Sex: it was found that for all the diseases the prevalence of gender is masculine with a ratio closed to 60/40%. However, the whole population in the dataset tends to have the same ratio which could indicate that there are just more test cases of Masculine diseases rather than a tendency of masculine gender to acquire a certain disease. For Pneumonia cases the ratio is 58.5% Masculine and 41.5% Femenine, as shown below: 

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure18.JPG?raw=1' width=600px>

### 5. Ground Truth

Ground truth was taken from NIH ChestX-ray dataset that comprises of 112,120 frontal-view X-ray images of 30,805 unique patients with the text-mined fourteen disease image labels (where each image can have multi-labels), mined from the associated radiological reports using natural language processing.

The text-mined disease labels are expected to have accuracy >90%

### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

* People in the age range of `20-65` since the algorithm has not enough data to train outside these ranges to be considered statistically confident about them, as shown below:

 <img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure18.JPG?raw=1' width=400px>

* Not indicated for use in people with other diseases with the exception of `Edema` and `Infiltration`. 

* It is known that the presence of `Atelectasis` is a limitation of the algorithm so we must avoid people with that disease since the algorithm performed very poorly in this case, as shown below:

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure19.JPG?raw=1' width=400px>

* Gender distribution could be even since model was trained on a ratio of 58.5% Masculine and 41.5% Femenine

**Ground Truth Acquisition Methodology:** Labeled by 3 radiologists and vote (Silver Ground Truth). The silver standard approach of using several radiologists would be more optimal for our algorithm since data from NIH was taken from various radiologists as well.

**Algorithm Performance Standard:** Sensitivity or recall metric should be selected since for pneumonia detection, it is crucial that we find all the patients that have pneumonia. Predicting patients with pneumonia as healthy is not acceptable. In other words having high FN is not acceptable. This is measured by recall. Therefore, my approach will be to weight recall over precision.

<img src='https://github.com/CheloGE/ML-AI_for_Healthcare-Pneumonia_detection_Chest_X_rays/blob/master/Figures/figure20.JPG?raw=1' width=400px>
