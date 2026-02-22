This readme file is attached to the Dataset on Italian Traffic Sign detection subset and is valid for both the test and the training set.
Along with this file you should have a training and test folder containing all the samples.
You are free to use this data package for any purpose you like.
This dataset come as it is, withouth any warranty.

The training set contains three folders, one for every identified superclass. There are three superclasses chosen according to the shape of the sign 
present. I.e. prohibitory, warning, indication (circle, triangle, square).

Particularly the test set is composed by a set of full size images taken in different conditions (fog, day, night) and divided per folder. Each
of these folder contains a annotation.txt file containing all the annotation for every images. 
Annotations are of the form:	imageName upperLeftCorner_X upperLeftCorner_Y bottomRightCorner_X bottomRightCorner_Y superclass
e.g. (0.png 814 43 968 197 warning)

NOTE = With respect to previous dataset as the German Traffic Sign Recognition Benchmark, the superclasses are different. Particularly there is the 
indication superclass that is completely new while the "german prohibitory" and "german mandatory" are here merged to form the prohibitory superclass.
Please consider these two differences when making comparisons.
