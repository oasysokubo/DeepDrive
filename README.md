# DeepDrive

## Introduction
DeepDrive simulates a self-driving car that is entirely fuelled by a Convolutional Neural Network (CNN) using Tensorflow and Computer Vision using OpenCV. There are many processes that I have gone through to make this project go through succesfully. Here are the specific tasks that I have done:

* Trained a Deep Learning Model that can identify between 43 different Traffic Signs
* Applied Computer Vision and Deep Learning techniques to build automotive-related algorithms
* Built and trained Convolutional Neural Networks with Keras
* Used essential Computer Vision techniques to identify lane lines on a road

## Background

### My solution to training the model:
The output is a distribution of steering angles that are much more uniform, viewed data in relative proportion. There are significant left and right steering angles, eliminating the extreme bias to driving straight all the time. This extreme bias resulted from the training as we drove the car as smooth as possible in the middle of the road, and thereby, recording the ideal steering angle at all times. We fix this by cropping most of the extraneous training data

### Alternate Solution:
Let the car continuously steer in either direction drifting to the edge of the road, but not actually hitting the edges and have it recover back to the middle before it crashes
Because simply driving down the middle of the road and recording that, is not enough to train model to drive properly

Imagine the car wanders off to the side of the road in such a case it wouldn't be able to recover back to the middle as it wouldn't be able to predict the appropriate left steering measurements therefore an alternative to manipulating the data in this way is to record a recovery laps where we actually make a separate recording of the car constantly steering back from the sides.

So what you would do is turn off the recording of the car or wandering off to the side since that's not the behavior that we want to train the model on but only record once we're going to steer back to the middle.

And now since having gotten rid of the zero angle bias not completely since we still want our car to favor driving down the middle let's load our image and steering data into arrays so we're able tomanipulate them and split them into training and validation data.


## In Action
### Output of terminal
![](https://github.com/oasysokubo/DeepDrive/blob/master/img/output.gif)
The first column of numbers represent the steering angle for the vehicle to follow, next the second, middle column represents the throttle rate, and finally the third and last column represents the speed of the vehicle.

## Installation

1) First things first, we have to install Udacity's Self-Driving Car Simulator to get this thing going. You can do that by visiting the repository and downloading the Version 2, 2/07/17 for any of the Operating Systems (Linux, Mac, or Windows)

[Link to the Simulator](https://github.com/udacity/self-driving-car-sim "Udacity Simulator")

2) Next, please clone my repository to any directory of your choosing. 
```bash
git clone https://github.com/oasysokubo/DeepDrive.git
```

## Usage

Linux/Unix Systems
1) First lets go into the main repository, you will find all my files.
```bash
cd DeepDrive
```
2) Now lets finally deploy the autonomous car simulation!
```bash
cd behavior_cloning
```

3) Run the program by typing

```bash
python drive.py
```
for Python version 1, and 
```bash
python3 drive.py
```
for Python version 3 users

4) Finally after getting this ran, open up the simulation program and the configurations settings will appear. You may choose to your liking, however for optimal experience and responsiveness, choose the lowest screen resolution and fastest graphics quality. Then click Play!

5) Now, click on autonomous mode and see the car moving autonomously. 
