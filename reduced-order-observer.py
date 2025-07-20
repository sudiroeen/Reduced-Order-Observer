#
# Author: Sudiro [sudiro@mail.ugm.ac.id]
#

import numpy as np 
import matplotlib.pyplot as plt 
import scipy
from scipy.signal import place_poles 

a21_ = 76.3 
a41_ = -0.36
b2_ = -3.7
b4_ = 0.49

W11mat_ = np.array([[0,0], [0,0]])
W12mat_ = np.array([[1,0], [0,1]])
W21mat_ = np.array([[a21_,0], [a41_, 0]])
W22mat_ = np.array([[0,0], [0,0]])
Wmat = np.vstack([
	np.hstack([W11mat_, W12mat_]),
	np.hstack([W21mat_, W22mat_])])

M11mat_ = np.array([[0,0]]).T 
M21mat_ = np.array([[b2_, b4_]]).T 
Mmat = np.vstack([M11mat_, M21mat_])

Pmat = np.array([
	[1, 0, 0, 0],
	[0, 1, 0, 0]
	])
Dmat = np.array([[0,0]]).T

cont_des_poles = [-2+1j*5, -2-1j*5, -4, -5]
obsv_des_poles = [-6, -8] 

print("Desired Poles controller: \n\t", cont_des_poles)
print("Desired Poles Observer: \n\t", obsv_des_poles)

Kcont_mat_ = scipy.signal.place_poles(Wmat, Mmat, cont_des_poles).gain_matrix
Kobs_mat_ = scipy.signal.place_poles(W22mat_, W12mat_, obsv_des_poles).gain_matrix

print("\n")
print("Kcont: \n", Kcont_mat_)
print("Kobs: \n", Kobs_mat_)

Gm1mat_ = W22mat_ - Kobs_mat_@W12mat_
Gm2mat_ = W21mat_ - Kobs_mat_@W11mat_ + (W22mat_ - Kobs_mat_@W12mat_)@Kobs_mat_

print("eigval Gm1mat_: \n")
print(np.linalg.eig(Gm1mat_)[0])

print("\n")
print("eigval Wmat-Mmat@Kc: ")
print(np.linalg.eig(Wmat - Mmat@Kcont_mat_)[0])

dt_ = 1e-4 # small enough to approximate digital control simulation as continuous-time framework

# actual system variables
vst_sys = np.array([[0.1,0.2,0,0]]).T 
yout_sys = np.array([[0.1, 0.2]]).T 

# observer system variables
# vbh_obs = np.array([[5,2]]).T 
gm_obs = np.array([[5,2]]).T

# actuator control effort
uinp_act = np.array([[0]])

# controller variables
vh_ct = np.array([[0,0,0,0]]).T

def actual_system(vst_, uinp_):
	vst_dot_ = Wmat @ vst_ + Mmat @ uinp_
	vst_ = vst_ + vst_dot_* dt_ 
	yout = Pmat @ vst_ + Dmat@uinp_ # sensor measurement
	return vst_, yout

def min_ord_observer(gm_, yout_, uinp_):
	gm_dot_ = Gm1mat_ @ gm_ + Gm2mat_ @ yout_ + M21mat_ @ uinp_
	gm_ = gm_ + gm_dot_*dt_
	vbh_ = gm_ + Kobs_mat_@yout_
	vh_ = np.vstack([yout_, vbh_])
	return gm_, vh_ 

def controller(vh_):
	uinp_ = -Kcont_mat_ @ vh_ 
	return uinp_

# for simulation
tm_ = np.arange(0, 4, step=dt_)
lvst_sys = list()
lvh_obs = list()
luinp_act = list()

for t in tm_:
	uinp_act = controller(vh_ct)
	vst_sys, yout_sys = actual_system(vst_sys, uinp_act)
	gm_obs, vh_ct = min_ord_observer(gm_obs, yout_sys, uinp_act)

	lvst_sys.append(vst_sys)
	lvh_obs.append(vh_ct)
	luinp_act.append(uinp_act)

lvst_sys = np.hstack(lvst_sys)
lvh_obs = np.hstack(lvh_obs)
luinp_act = np.hstack(luinp_act)

fig, ax = plt.subplots(5)
ax[0].plot(tm_, lvst_sys[0], '-r', label='v1')
ax[0].legend(loc='lower right')

ax[1].plot(tm_, lvst_sys[1], '-g', label='v2')
ax[1].legend(loc='lower right')

ax[2].plot(tm_, lvst_sys[2], '-k', label='v3 - unknown')
ax[2].plot(tm_, lvh_obs[2], '--r', label='v3h')
ax[2].legend(loc='lower right')

ax[3].plot(tm_, lvst_sys[3], '-k', label='v4 - unknown')
ax[3].plot(tm_, lvh_obs[3], '--r', label='v4h')
ax[3].legend(loc='lower right')

# fig, ax = plt.subplots(1)
ax[4].plot(tm_, luinp_act[0], '-k', label='controller effort')
ax[4].legend(loc='lower right')

plt.show()
