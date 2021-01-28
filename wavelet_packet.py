import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2

def wavelet_coeff(img, x, y, z):
	b,g,r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

	b_wavelet3 = pywt.WaveletPacket2D(b, 'haar', mode='periodization', maxlevel=3)
	g_wavelet3 = pywt.WaveletPacket2D(g, 'haar', mode='periodization', maxlevel=3)
	r_wavelet3 = pywt.WaveletPacket2D(r, 'haar', mode='periodization', maxlevel=3)

	bA3 = b_wavelet3.get_level(3,decompose=True)
	gA3 = g_wavelet3.get_level(3,decompose=True)
	rA3 = r_wavelet3.get_level(3,decompose=True)

	b_data = bA3[x].data
	g_data = gA3[y].data
	r_data = rA3[z].data


	b_data /= np.abs(b_data).max()
	g_data /= np.abs(g_data).max()
	r_data /= np.abs(r_data).max()


	rgb = np.dstack((b_data,g_data,r_data))

	return rgb
