# VideoSlides

This script takes in videos of powerpoint lectures and splits it into the individual slides by measuring the change in the number of corners in the image.
* The video file should be a .avi file.
* REQUIRES OPENCV AND NUMPY

## Installation guide for Mac OS X

* Install numpy > 1.9 with

		pip install numpy
* Use brew to install opencv

		brew tap homebrew/science
		brew install opencv --with-ffmpeg
Now link the opencv python bindings to your python site-packages

		ln -s /usr/local/Cellar/opencv/2.4.9/lib/python2.7/site-packages/cv.py $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
		ln -s /usr/local/Cellar/opencv/2.4.9/lib/python2.7/site-packages/cv2.so $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

##Example Usage
	VidToSlides.py ./test.avi
	