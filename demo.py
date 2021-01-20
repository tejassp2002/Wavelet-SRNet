import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("lena.jpg").astype(np.float32) #float array
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b,g,r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

b_wavelet3 = pywt.wavedec2(b, 'haar', mode='periodization', level=3)
g_wavelet3 = pywt.wavedec2(g, 'haar', mode='periodization', level=3)
r_wavelet3 = pywt.wavedec2(r, 'haar', mode='periodization', level=3)
b_wavelet3[0] /= np.abs(b_wavelet3[0]).max()
g_wavelet3[0] /= np.abs(g_wavelet3[0]).max()
r_wavelet3[0] /= np.abs(r_wavelet3[0]).max()

bA3 = b_wavelet3[0]
gA3 = g_wavelet3[0]
rA3 = r_wavelet3[0]
rgb = np.dstack((bA3,gA3,rA3))/*
cv2.imshow("b",rgb)
cv2.waitKey(0)
cv2.destroyAllWindows() 
