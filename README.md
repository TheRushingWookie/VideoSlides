# VideoSlides

This script takes in videos of powerpoint lectures and splits it into the individual slides by measuring the change in the number of corners in the image.
* The video file should be a .avi file.
* REQUIRES OPENCV AND NUMPY

## Installation guide for Mac OS X
* Install numpy > 1.9 with

	pip install numpy
* Use brew to install opencv

	brew tap homebrew/science
	brew install opencv

##Example Usage
	VidToSlides.py ./test.avi