from keras.models import Model
from keras.layers import Conv2DTranspose, Input

def reconstruction_net(input_shape=(None,3,None,None), r):
	input_img = Input(shape=input_shape)
	output = Conv2DTranspose(3, r, r, padding="valid", data_format="channels_first")(input_img)
	model = Model(input_img, output)

	return model
