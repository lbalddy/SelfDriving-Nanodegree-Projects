#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/CAPTURA.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./New_traffic_signs/100limit.jpg "Traffic Sign 1"
[image5]: ./New_traffic_signs/children_crossing.jpg "Traffic Sign 2"
[image6]: ./New_traffic_signs/Roundabout.jpg "Traffic Sign 3"
[image7]: ./New_traffic_signs/stop.jpg "Traffic Sign 4"
[image8]: ./New_traffic_signs/yield.jpg "Traffic Sign 5"
[image9]: ./examples/roundabout.png "Round about"

---

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


Here is the writeup. 

Code for the project is splitted so you can check it and get a clear idea about it. 



###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in 01 - Exploring dataset.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in 01 - Exploring dataset.  

As we see below, traffic sign data set isn't evenly distributed. This means, that our traffic sign recognition NN will have a bias towards images that have major representation in the dataset. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in 03 - Preprocessing.

Yan Lecunn paper states that http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf using color channels doesn't improve the neuronal network, so we avoid using 3 color channels, reducing therefore the size of the neuronal network. 

Moreover, we have a applied localized histogram equalization. That may improve feature extraction, enhancing the image and making it easier to recognize. In the image below, we can see that image is well separated from the rest of the picture. Therefore, it will be easier for a convolutional NN to generalise over the data.

![alt text][image2]

Features were scaled from 0 to 1. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Validation and testing data are unaltered, and remain as Udacity has given them for this project. However, training data has been augmentated. In a neuronal network, using a big chunk of data, which is related to your problem, is the most important task, and we can say, after this project that your NN will be as good as it is your training dataset. 

The training data has been augmatated in the next manner. 
* Random translation, from -2 to 2 pixels. 
* Random rotation. From -15 to 15 degrees. This is tricky, because we can't flip traffic signs at our will. The meaning of some traffic signs is changed or lost when they flip. For example, stop signal doesn't have sense when it's flipped 90 degrees. This can be addresed properly during augmentation, but it's something where we didn't want to enter. So we preferred to keep it simple. 
* Blurring. Using a 5x5 blurring kernel. 

I have incremented by 3 the original training dataset, using 139196 distributed in the same way as the original data, because no consideration has been taken over those traffic signs that are less representated in the dataset. Is obvious, that using an evenly distributed dataset is far more superior, and we could achieve that augmatating data wisely. 


![alt text][image3]



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started using Lenet original architecture with the dataset, and I improved the architecture enough to reach the goal of the project, but considering I'll run on a laptop, without GPU. 

With the original Lenet architecture, and the data augmentated I reached a 93.2% on the test dataset. Through fine tuning, and some modifications on the NN I've finally reached a 96.2% on the test dataset. 

Last architecture includes dropout, and more depth on the first layer. Training time increased, but was short enough I can tune the parameters correctly. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y channel image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x14 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x14				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  					|
| RELU         		|         									|
| Convolution 5x5				| 1x1 stride, valid padding, outputs 5x5x16       									|
|Fully connected|		Input 400. Output 120										|
|				RELU		|												|
| Fully connected| Input 120  -  Output 84|
|RELU||
|Dropout| Keep prob = 0.5|
|Fully connected| Input 84 - Output 43|

As you can see, this is a modified Lenet architecture which has performed well for this dataset. Achieving the dataset accuracy needed to pass. 

I think, that the NN is big enough to solve the problem without overfitting. An could be tuned better with a larger dataset. For me, one of the most important things was to keep it small to play with it enough, something where I think I have performed well. Because it's light an simple to solve our problem. 
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model I used Adam Optimizer. Adam Optimizer is a computionally efficient method for stochastic optimization, with little memory requirementes and well suited for problems with large datasets.

- Batch size = 128
- Epoch = 30
- Initial learning rate = 0.001. Learning rate has a little algorithm integrated into the training code, in order to divide by 10 whenever we find ourselves worsening the training results. Therefore, we have a quick training until we find the optimum. We could have even have implemented an stop for the training when it gets stabilized. However, this wasn't done. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 98.7%
* validation set accuracy of 97.5%
* test set accuracy of 95.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

*   I started with LeNet5 architecture, and I increased the depth of the first layer. Moreover I added dropout, which demonstrated a correct working and tried regularization which made the NN act poorly.

* What were some problems with the initial architecture? 

Initially LeNet5 architecture didn't achieved the goal, but achieved it quickly as soon I augmentated the data. I was sure that with small modifications I could improve the accuracy on the test dataset over 95%.

* How was the architecture adjusted and why was it adjusted? Typicwial adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. 

It was adjusted by tiny modifications of Lenet5. First I noticed that LeNet maybe was too simple for this problem, so I incremented graph in the first two layers so many more features were taken in account. 

I saw improvement in the NN when added dropout. I've think a lot about dropout, and I think that could be resembled as increasing the dataset. Basically, you "blank out" some features during the training, as they never existed. 

I didn't have any overfitting problem as I went improving the NN bit by bit. But it took me a while to increase the performance of the NN. 

* Which parameters were tuned? How were they adjusted and why? 

Dropout was added -> Keep probability is a parameter that needs tune. I keep everything static and I changed it. I manage to go from 94% to 96%. 
Batch size        -> I used one bigger so I could train the NN quick in my laptop. 
Learning rate     -> It starts at a value 0.001 but while the NN is converging I reduce it by a factor of ten. The algorithm to decide whether or not reduce it uses a counter of how many times we dimish the result of the NN instead of improving it in an epoch. Doing this I reach a good performance.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  A dropout layer adds some "noise" by removing features, and that's always welcomed because we are basically making the NN more solid to changes. As we can find different traffic signs in the road. Convolutional layer are the best NN to apply to image problems because they are invariant to rotations and translations. 

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The five images should be easy to classify. They're centered, only one is distorted because of the picture. However, when it comes to image number 3, we find that the roundabout is classified as a 100 kmh traffic signal. The reason is easy, during the preprocessing and image resizing, so much information is lost and the image is difficult to classify for our neural network. 
![alt text][image9]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in 04.-Testing in new images & in dataset.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Roundabout     			|100 km/h										|
| Children crossing					| Children crossing												|
| 100 km/h	      		|100 km/h					 				|
| Yield		|Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the accuracy on the test set. As we saw, we have a bias towards some classes because training dataset isn't balanced. 

More

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in 04.-Testing in new images & in dataset.The model is always 100% sure of the decission, even when it gets wrong. That's tricky, and a sign that the model needs some improvement, however, as I am in my "month of grace" for personal reasons I think I can stop here. 

####4. Improvements

* More balanced dataset
* Using a different NN
* Implement different optimizer
* Less aggresive preprocessing of the images

