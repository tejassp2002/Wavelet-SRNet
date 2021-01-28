import numpy as np  
from wavelets import wavelet_packet
def wavelet_loss(w, w_hat, N_w):  #w is ground truth wavelet coefficients
	l_wavelet = 0			  #w_hat is inferred wavelet coefficients
	for i in range(N_w):
		for j in range(N_w):
			for k in range(N_w):
				l_wavelet += np.linalg.norm(w_hat(i,j,k)-wavelet_coeff(w,i,j,k))

	l_wavelet -= np.linalg.norm(w_hat(0,0,0)-wavelet_coeff(w,0,0,0))
	l_wavelet += 0.01*np.linalg.norm(w_hat(0,0,0)-wavelet_coeff(w,0,0,0))

	return l_wavelet

def texture_loss(w, w_hat, N_w):
	l_texture = 0
	k = 2 #start index
	eps = 0.001 #epsilon - slack value
	alpha = 1.0   #alpha - slack value
	gamma = 1.0   #gamma - balance weights
	for p in range(k, N_w):
		for q in range(k, N_w):
			for r in range(k, N_w):
				l_texture += gamma*max(
					alpha*np.linalg.norm(wavelet_coeff(w,i,j,k))
					+ epsilon - np.linalg.norm(w_hat(i,j,k)),0)
	return l_texture


def full_image_loss(y, y_hat):
	l_full_image = np.linalg.norm(y_hat-y)  #y is ground-truth HR image
	return l_full_image			  			#y_hat is estimated high-resolution image


def loss_function(w, w_hat, y, y_hat, N_w):
	mu = 1.0
	v = 0.1
	l_total = (wavelet_loss(w, w_hat, N_w) 
	          + mu*texture_loss(w, w_hat, N_w)+v*full_image_loss(y,y_hat))
	return l_total
