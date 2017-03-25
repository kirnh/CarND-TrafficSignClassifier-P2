#**Traffic Sign Recognition** 

---

#####Goals

The goals / steps of this project are the following:

  * Load the data set
  * Explore, summarize and visualize the data set
  * Design, train and test a model architecture
  * Use the model to make predictions on new images
  * Analyze the softmax probabilities of the new images
  * Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/labels_distribution.jpg "Visualization"
[image2]: ./examples/original.jpg "No entry image"
[image3]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./examples/hist_equalized.jpg "Histogram equalization"
[image5]: ./examples/non_transformed.jpg "Original image"
[image6]: ./examples/transformed.jpg "Augmented image"
[image7]: ./examples/ahead_only.png "Traffic Sign 1"
[image8]: ./examples/children_crossing.png "Traffic Sign 2"
[image9]: ./examples/dangerous_curve_to_the_left.png "Traffic Sign 3"
[image10]: ./examples/go_straight_or_right.png "Traffic Sign 4"
[image11]: ./examples/slippery_road.png "Traffic Sign 5"


Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

####Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading the writeup, and here is a link to my [project code](https://github.com/kirnh/CarND-TrafficSignClassifier-P2)

---

####Data Set Summary & Exploration

#####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the native python and numpy methods to calculate summary statistics of the traffic
signs data set which is as follows:

  * The size of training set is 34799
  * The size of validation set is 4410
  * The size of test set is 12630
  * The shape of a traffic sign image is (32, 32, 3) 
  * The number of unique classes/labels in the data set is 43

#####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a chart showing how the data is distributed across the 43 different classes. As we can see, the distribution is clearly non uniform.

![Distribution of images across classes][image1]

---

####Design and Test a Model Architecture

#####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the thirteenth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because in the classification of traffic signs, the color channels don't really help a lot as features. Grayscale images also make the model smaller with less parameter requirements.

I then increased the contrast of the image using histogram equalization to better identify the patterns in the image.

As a last step, I normalized the image data to a range [-1, 1] because having equally scaled features helps the learning process by increasing the chances of a faster convergence with randomly initialized parameters. This also helps us avoid getting stuck at local minimas to an extent.

Here is an example of a traffic sign image through the preprocessing steps.

![Original image][image2]

![Grayscaled image][image3]

![Histogram equalized image][image4]

#####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The initial dataset used contains three different files containing 34799 training images, 4410 validation images and 12630 testing images. Since the training images are unevenly distributed across the different classes, I augmented the training dataset so that each class contains atleast 2000 images. This was done while using affine transformations like random shear, rotation and translation. Also, random brightness was applied during transformations. These methods were used keeping in mind the possible variations that can be anticipated with traffic sign images.

This augmentation of the training dataset which was done in the fifth code cell helps us avoid the model to become biased towards predicting classes having more images in the training set. Also, the random transformations will make our model more robust to variations in images. 

After augmentation, my final training set contained 86010 images that are evenly distributed across the 43 classes. 

Here is an example of an original image and an augmented image:

![Original image][image5]
![A randomly augmented image][image6]

#####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 16th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5   	| 1x1x1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2    	| 1x2x2x1 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5   | 1x1x1x1 stride, valid padding, outputs 10x10x16		|
| RELU					|												|
| Max pooling 2x2    	| 1x2x2x1 stride, valid padding, outputs 5x5x16		|
| Flatten						| outputs 400										|
| Fully connected		| outputs 120									|
| RELU                                           |
| Dropout						| keep_prob = 0.9    |
|	Fully connected		| outputs 84												|
| Dropout						| keep_prob = 0.9												|
| Fully connected			| outputs 43 												|
| Softmax				| outputs 43 probabilities   									|
 

#####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 20th cell of the ipython notebook. 

To train the model, I used cross entropy as the loss function, and Adam optimizer to find the choice of parameters that minimized the loss. I used a learning rate of 0.0022 for training, with a batch size of 128 for 25 epochs. 

#####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 20th cell of the Ipython notebook for training set and validation set. While for the test set, the accuracy is calculated in the 23rd code cell.

Initially, I took a small sample of my training set of around 2000 images and overfit a model with LeNet architecture. The overfitting was done by iterating over a number of learning rates and batch sizes, and selecting the best choice. Then, to conteract the overfitting, I added regularization using dropouts over the first two fully connected layers in my model with a keep_prob of 0.9 which worked well at regularizing overfitting! Then I used all my training data to train the model. The number of epochs to be used was decided by looking at the convergence of the loss and the validation accuracy. 

The choice of LeNet architecture as a starting point for the classification problem at hand was made because of its good performance on image classification. This in turn, is because of the convolutional layers used in the architecture which works well on image data.

My final model results were:
  * training set accuracy of 93.2%
  * validation set accuracy of 95.8% 
  * test set accuracy of 92.6%
 
--- 
 
####Test a Model on New Images

#####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![ahead only][image7] 
![children crossing][image8] 
![dangerous curve to the left][image9]
![go straight or right][image10] 
![slippery road][image11]

The second image and the fifth image might be difficult to classify because they involve very intricate shapes inside of them. If our model is to classify these correctly, it would have to have learned these intricate features while training which is a hard task in itself.

However, the remaining 3 images must be easily classified by the model.

#####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model for the above images is located in the 27th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only     		| Ahead only   									| 
| Children crossing | Slippery road 										|
| Dangerous curve to the left | Dangerous curve to the left 											|
| Go straight or right | Go straight or right					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Although the accuracy of 80% is far compared to the accuracy on the test set of 92.6%, I believe this is because of the very small number (5) of examples from the web that we have used to make predictions. With increase in the number of test images, I hope that the accuracy might reach the test set accuracy. 

#####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Ahead only 									| 
| .00     				| Go straight or left						|
| .00					| Turn left ahead						|
| .00	      			| Right-of-way at the next intersection			 				|
| .00				    | Road work	|


For the second image, the model is relatively sure that this is a slippery road sign (probability of 0.5), but the image actually contains a children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .50         			| Slippery road 									| 
| .15     				| No entry						|
| .12					| General caution						|
| .10	      			| Vehicles over 3.5 metric tons prohibited	 				|
| .03				    | Dangerous curve to the right	|

For the third image, the model is relatively sure that this is a dangerous curve to the left sign (probability of 0.99), and the image actually contains the same sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Dangerous curve to the left 									| 
| .00     				| Slippery road					|
| .00					| Bumpy road   |
| .00	      			| Dangerous curve to the right 				|
| .00				    | No passing	|

For the fourth image, the model is relatively sure that this is a go straight or right sign (probability of 0.95), and the image actually contains the same sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Go straight or right									| 
| .04     				| Right-of-way at the next intersection					|
| .00					| Ahead only   |
| .00	      			| Roundabout mandatory 				|
| .00				    | End of no passing by vehicles over 3.5 metric tons	|

For the fifth image, the model is relatively sure that this is a slippery sign (probability of 0.94), and the image actually contains the same sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| Slippery road									| 
| .05     				| Double curve					|
| .00					| Right-of-way at the next intersection   |
| .00	      			| Wild animals crossing 				|
| .00				    | Bumpy road	|

---