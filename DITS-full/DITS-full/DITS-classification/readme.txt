This readme file is attached to the Dataset on Italian Traffic Sign classification subset and is valid for both the test and the training set.
Along with this file you should have a classes.txt file containing all the classes currently available for the dataset, a training and test folder
containing all the samples.

Particularly the training set may not contains the same number of classes as for the test.
You are free to use this data package for any purpose you like.
This dataset come as it is, withouth any warranty.

For further details or question write to dario.albani@gmail.com.

For the classification subset, 58 classes are identified and are grouped in 3 superset: warning, indication, prohibitory. 
The supersets are chosen according to the shape of each sign inside it (triangle, square, circle) and to the majority of the signs present in each 
superset.
Each class is named according to UK names and by taking in consideration the meaning of each sign in Italy. Correspondences between names and signs 
are not ensured.
Each class has at least one(1) track containing 15 images of the same traffic sign instance. A track is a set of images of the same istance taken while
approaching the sign with the car. Speed (thus distance between each sample) vary.

NOTE = With respect to previous dataset as the German Traffic Sign Recognition Benchmark, the superclasses are different. Particularly there is the 
indication superclass that is completely new while the "german prohibitory" and "german mandatory" are here merged to form the prohibitory superclass.
Please consider these two differences when making comparisons.

The mapping is as follow # = className (superclass) 
