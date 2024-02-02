'''
While loading and preprocessing image data can be a bit tricky,Keras 
provides us with a few tools to make the process less burdensome. Of 
these, the ImageDataGenerator class is the most critical. We can use
ImageDataGenerators to load images from a file path, and to preprocess
them.

Beyond just loading our images, the ImageDataGenerator can also preprocess 
our data. We do this by passing additional arguments to the constructor.

There are a few ways to preprocess image data, but we will focus on 
the most important step: pixel normalization. Because neural networks 
struggle with large integer values, we want to rescale our raw pixel 
values between 0 and 1. Our pixels have values in [0,255], so we can 
normalize pixels by dividing each pixel by 255.0.

'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Creates an ImageDataGenerator:
training_data_generator = ImageDataGenerator(
  rescale = 1.0/255, #Perform pixel normalization

  zoom_range = 0.2, #Increase or decrease the size of the image by up to 20%

  rotation_range = 15, #Randomly rotates the image between [-15,15] degrees

  width_shift_range = 0.05, #Shift the image along its width by up to +/- 5%

  height_shift_range = 0.05 #Shifts the image along its height by up to +/- 5%
)

#Prints its attributes:
print(training_data_generator.__dict__)
