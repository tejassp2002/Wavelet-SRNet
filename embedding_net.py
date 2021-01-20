from keras.models import Model
from keras.layers import Conv2D, Input, add
from keras.layers import add
from keras.layers import BatchNormalization



def embedding_net(input_shape=(None,None,3)):
	input_img = Input(shape=input_shape)

	output = Conv2D(128, (3,3), strides=(1, 1), padding="same")(input_img)
	block_1_output = BatchNormalization()(output)

	output = Conv2D(128, (3,3), strides=(1, 1), padding="same")(block_1_output)
	output = BatchNormalization()(output)
	output = Conv2D(128, (3,3), strides=(1, 1), padding="same")(output)
	output = BatchNormalization()(output)
	block_2_output = add([output, block_1_output])

	output = Conv2D(256, (3,3), strides=(1, 1), padding="same")(output)
	output = BatchNormalization()(output)
	output = Conv2D(256, (3,3), strides=(1, 1), padding="same")(output)
	output = BatchNormalization()(output)
	block_3_output = add([output, block_2_output])

	output = Conv2D(512, (3,3), strides=(1, 1), padding="same")(block_3_output)
	output = BatchNormalization()(output)
	output = Conv2D(512, (3,3), strides=(1, 1), padding="same")(output)
	output = BatchNormalization()(output)
	block_4_output = add([output, block_3_output])

	output = Conv2D(1024, (3,3), strides=(1, 1), padding="same")(block_4_output)
	output = BatchNormalization()(output)
	output = Conv2D(1024, (3,3), strides=(1, 1), padding="same")(output)
	output = BatchNormalization()(output)
	block_5_output = add([output, block_4_output])

	model = Model(input_img, block_5_output)

	return model