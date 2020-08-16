import numpy as np
import pandas as pd
from angles import rad2dms, dms2rad

directions = pd.read_csv('data/mesurments.csv', header=None)
rf_np = np.apply_along_axis(dms2rad, 1, directions.iloc[:, 0:3])
lf_np = np.apply_along_axis(dms2rad, 1, directions.iloc[:, 3:6])
directions_rad_np = np.vstack((rf_np, lf_np)).T
direct_err_fixed = np.zeros((rf_np.shape[0], 4))
direct_err_fixed[:, 0:2] = directions_rad_np


def fl_fix(faces):
	if faces[1] > faces[0]:
		return (faces[0] + faces[1] + np.pi) / 2
	else:
		return (faces[0] + faces[1] - np.pi) / 2


def fix(faces):
	if faces[1] > faces[0]:
		return (faces[0] - faces[1] + np.pi) / 2
	else:
		return (faces[0] - faces[1] - np.pi) / 2


direct_err_fixed[:, 3] = np.apply_along_axis(fl_fix, 1, directions_rad_np)
direct_err_fixed[:, 2] = np.apply_along_axis(fix, 1, directions_rad_np)
erros = np.apply_along_axis(rad2dms, 1, direct_err_fixed[:, 2].reshape(rf_np.shape[0], 1))

A = np.zeros((6 * 2 + 4 * 6, 6 + 4))
Lb = pd.DataFrame(direct_err_fixed[:, 3].reshape(rf_np.shape[0], 1), columns=['LF_FIXED'])
Lb_no_o1 = Lb.drop([0, 5])
index = directions.iloc[:, 6]
index_no_o1 = index.drop([0, 5])
P = []
for row in range(0, 36, 6):
	set_indx = int(row / 6)
	A[row:row + 6, set_indx] = np.array([1, 1, 1, 1, 1, 1])
	A[row + 1, 6] = 1
	A[row + 2, 7] = 1
	A[row + 3, 8] = 1
	A[row + 4, 9] = 1
	P.extend([1, 1, 1, 1, 0.1, 1])
P = np.diagflat(P)

Lb = Lb.to_numpy()

N = np.dot(np.dot(A.T, P), A)
U = np.dot(np.dot(A.T, P), Lb)
x = np.dot(np.linalg.inv(N), U)
x_dms = np.apply_along_axis(rad2dms, 1, x)

v = np.dot(A, x) - Lb
tesssssssssssssst = np.dot(A, x)
sig_post = np.dot(v.T, v) / (36 - 10)

rms = np.diag(sig_post[0] * np.linalg.inv(N))
rms_dms = np.apply_along_axis(rad2dms, 1, rms.reshape(10, 1))
rms_dms_sqrt = np.apply_along_axis(rad2dms, 1, (rms ** 0.5).reshape(10, 1))

length = pd.read_csv('data/adj_lenth.csv')

directions_adj = []
directions_adj.extend(x[6] - x[0])
directions_adj.extend(x[7] - x[0])
directions_adj.extend(x[8] - x[0])
directions_adj.extend(x[9] - x[0])
e = 5.41
angles_adj = []
angles_adj.append(directions_adj[3] - directions_adj[0])
angles_adj.append(directions_adj[3] - directions_adj[1])
angles_adj.append(directions_adj[3] - directions_adj[2])


def fix_ang(xi, e, alpha, l):
	return xi + e * np.sin(alpha) / l


c_fix = []
c_fix.append(fix_ang(directions_adj[0], e, angles_adj[0], length.iloc[0, 1]))
c_fix.append(fix_ang(directions_adj[1], e, angles_adj[1], length.iloc[0, 1]))
c_fix.append(fix_ang(directions_adj[2], e, angles_adj[2], length.iloc[0, 1]))

zavit = []
zavit.append(c_fix[2] - c_fix[1])
zavit.append(c_fix[1] - c_fix[0])
zavit = np.apply_along_axis(rad2dms, 1, np.array(zavit).reshape((2, 1)))
zavit = pd.DataFrame(zavit, columns=['D', 'M', 'S'], index=['C4-C1', 'C2-C4'])

c_fix = np.apply_along_axis(rad2dms, 1, np.array(c_fix).reshape((3, 1)))
c_fix = pd.DataFrame(c_fix, columns=['D', 'M', 'S'], index=['C2', 'C4', 'C1'])
print('pause')
