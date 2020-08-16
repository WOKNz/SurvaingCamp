# Import
import numpy as np


def rad2dms(rad, tostr=None):
	deg = rad[0] * 180 / np.pi
	d = int(deg)
	m = np.abs(int((np.abs(deg) - np.abs(d)) * 60))
	s = np.abs((np.abs(deg) - np.abs(d) - m / 60) * 3600.0)

	if tostr == True:
		return np.array([d, m, s])
	else:
		return np.array([d, m, s])


def dms2rad(dms, tostr=None):
	if dms[0] < 0:
		deg = dms[0] - dms[1] / 60.0 - dms[2] / 3600
	else:
		deg = dms[0] + dms[1] / 60.0 + dms[2] / 3600

	if tostr == True:
		return deg * np.pi / 180.0
	else:
		return deg * np.pi / 180.0

# if __name__ == '__main__':
# 	test = np.array([1.34523455345])
# 	test_vect = np.array([[1.33333333333333355],[-1.3452345534555234555]])
# 	result_test = rad2dms(test)
# 	vec_f = np.vectorize(rad2dms(test_vect))
# 	result_test_vec = np.apply_along_axis(rad2dms,1,test_vect)
#
# 	dms_vec = np.array([[200,12,56],[-200,12,56]])
# 	rad = np.apply_along_axis(dms2rad,1,dms_vec).T
#
#
# 	print('done')
