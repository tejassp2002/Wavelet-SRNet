from keras.models import Model
from keras.layers import Conv2D, Input, add
from keras.layers import add
from keras.layers import BatchNormalization


def wavelet_prediction_net(input_shape=(None,None,3)):
	input_img = Input(shape=input_shape)

	output = Conv2D(32, (3,3), strides=(1, 1), padding="same")(input_img)
	output = BatchNormalization()(output)
	output = Conv2D(32, (3,3), strides=(1, 1), padding="same")(output)
	output = BatchNormalization()(output)
	block_1_output = add([output, input_img])

	output = Conv2D(64, (3,3), strides=(1, 1), padding="same")(output)
	output = BatchNormalization()(output)
	output = Conv2D(64, (3,3), strides=(1, 1), padding="same")(output)
	output = BatchNormalization()(output)
	block_2_output = add([output, block_1_output])

	model = Model(input_img, block_2_output)

	return model