import numpy as np
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout

# Task 1
def light_pixels(image, lightness, channel):
     """Return a mask for each channel that identifies the pixels whose intensity is 
     above the given threshold. The mask of channel i is an array that has values 0 or 1.
     Assume that the image is stored as a numpy array with three channels for the colours
     'red', 'green', and 'blue'.

     Parameters:
     image (numpy.ndarray): Input image array with shape (height, width, 3).
     lightness (int): The intensity threshold.
     channel (str): The channel name ('red', 'green', 'blue').

     Returns:
     numpy.ndarray: Mask array with the same height and width as the input image, 
                    with values 0 or 1 indicating whether the pixel intensity is above the 
                    threshold.

     Examples:
     >>> image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]],
     ...                   [[  2,  20,  20], [250, 255, 255], [127, 127, 127]]])                          
     >>> light_pixels(image, 20, 'red')
     array([[1, 0, 0],
            [0, 1, 1]])
     >>> light_pixels(image, 20, 'green')
     array([[0, 1, 0],
            [0, 1, 1]])
     >>> light_pixels(image, 15, 'blue')
     array([[0, 0, 1],
            [1, 1, 1]])
     """
     # Dictionary maps each color channel name to its corresponding index in the image array
     channel_index = {'red' : 0, 'green' : 1, 'blue' : 2} 

     # Retrieve the index for the specified channel
     index = channel_index[channel]

     # Extract the data of specific channel using the index
     channel_data = image[:, :, index]

     # Create a boolean mask where each element is True if the pixel intensity in the specified 
     # channel is greater than the given threshold (lightness), otherwise False
     boolean_mask = channel_data > lightness

     # Convert the boolean mask to integer array
     # True values become 1, and False values become 0
     mask = boolean_mask.astype(int)

     return mask

# Task 2
def histogram(image, buckets, channel):
    """
    Return a histogram of the channel, where the image is represented as a 
    3-channel numpy array with values between 0 and 255. A histogram is
    an array of length `buckets` where the i-th element is the count of pixels
    in the range [i * (256 // buckets), (i + 1) * (256 // buckets)).

    This function should not use third-party functions such as np.linspace or
    np.histogram.

    Parameters:
    image (numpy.ndarray): Input image array with shape (height, width, 3).
    buckets (int): Number of histogram buckets.
    channel (str): The channel name ('red', 'green', 'blue').

    Returns:
    numpy.ndarray: Histogram array of length `buckets`.

    Examples:
    >>> image = np.array([[[250,   2,   2], [  0,   2, 255], [  0,   0, 255]], 
    ...                   [[  2,   2,  20], [250, 255, 255], [127, 127, 127]]])
    >>> histogram(image, 4, 'red')
    array([3, 1, 0, 2])
    >>> histogram(image, 5, 'green')
    array([4, 0, 1, 0, 1])
    >>> histogram(image, 6, 'blue')
    array([2, 0, 0, 1, 0, 3])
    """
    # Dictionary maps each color channel name to its corresponding index in the image array
    channel_index = {'red' : 0, 'green' : 1, 'blue' : 2}

    # Retrieve the index for the specified channel
    index = channel_index[channel]

    # Extract the data of specific channel using the index
    channel_data = image[:, :, index]

    # Initialize the histogram array with zeros
    histogram = [0] * buckets

    # Bucket size and // is used because return type is integer
    bucket_size = 256 // buckets

    # Iterate over each pixel value in the channel data
    # Edge case: if the calculated bucket_index is out of range
    # adjust it to point to the last bucket
    for row in channel_data:
        for value in row:
            bucket_index = value // bucket_size
            if bucket_index >= buckets:
                 bucket_index = buckets - 1
            histogram[bucket_index] += 1
            
    return histogram
     
# Task 3
def build_deep_nn(rows, columns, channels, layer_options):
       #num_hidden, hidden_sizes, dropout_rates,
       #           output_size, output_activation):
     """Return a Keras neural model that has the following layers:
     - a Flatten layer with input shape (rows, columns, channels)
     - as many hidden layers as the length of layer_options
     - layer_options is a list of layer options, such that:
       - hidden layer number i is of size layer_options[i][0] and activation
         layer_options[i][1]
       - if layer_options[i][2] > 0, then hidden layer number i is followed
         by a dropout layer with dropout rate layer_options[i][2]

     Parameters:
     rows (int): Number of rows in the input.
     columns (int): Number of columns in the input.
     channels (int): Number of channels in the input.
     layer_options (list of tuples): Each tuple contains three elements:
                                    (hidden_size, activation, dropout_rate).

     Returns:
     keras.models.Sequential: The constructed Keras model.
     
     Examples:
     >>> rows = 28
     >>> columns = 28
     >>> channels = 2
     >>> layer_options = [
     ...     (128, 'relu', 0.2),  # 128 neurons, relu activation, 20% dropout
     ...     (64, 'relu', 0),     # 64 neurons, relu activation, no dropout
     ...     (32, 'sigmoid', 0.4) # 32 neurons, sigmoid activation, 40% dropout
     ... ]
     >>> model = build_deep_nn(rows, columns, channels, layer_options)
     >>> model.summary()
     Model: "sequential"
     ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
     ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
     ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
     │ flatten (Flatten)                    │ (None, 1568)                │               0 │
     ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
     │ dense (Dense)                        │ (None, 128)                 │         200,832 │
     ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
     │ dropout (Dropout)                    │ (None, 128)                 │               0 │
     ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
     │ dense_1 (Dense)                      │ (None, 64)                  │           8,256 │
     ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
     │ dense_2 (Dense)                      │ (None, 32)                  │           2,080 │
     ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
     │ dropout_1 (Dropout)                  │ (None, 32)                  │               0 │
     └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
      Total params: 211,168 (824.88 KB)
      Trainable params: 211,168 (824.88 KB)
      Non-trainable params: 0 (0.00 B)
     >>> model.layers[1].get_config()['activation']
     'relu'
     >>> model.layers[3].get_config()['activation']
     'relu'
     >>> model.layers[4].get_config()['activation']
     'sigmoid'
     """
     # Initializing a Sequential model
     neural = Sequential()

     # Adding a Flatten layer 
     neural.add(Flatten(input_shape=(rows, columns, channels)))

     # Iterate through the layer options to add hidden layers
     # Add a Dense layer with specified number of neurons and activation function
     for (hidden_size, activation, dropout_rate) in layer_options:
         neural.add(Dense(hidden_size, activation = activation))
     # Add a Dropout layer if the dropout rate is greater than 0
         if dropout_rate > 0:
             neural.add(Dropout(dropout_rate))
             
     return neural

if __name__ == "__main__":
     import doctest
     doctest.testmod()
