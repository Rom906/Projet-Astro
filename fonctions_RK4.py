import numpy as np
import matplotlib.pyplot as plt

#==== RK4 implementation ====
def RK4(rp, vp, t, dt, Nsteps, q, m, B, ROdip, mu):
    for i in range(1, Nsteps):
        rp1 = rp[i-1,:]
        vp1 = vp[i-1,:]
        ap1 = q/m* np.cross(vp1, B(rp1,ROdip,mu))

        rp2 = rp[i-1,:] + 0.5*vp1*dt
        vp2 = vp[i-1,:] + 0.5*ap1*dt
        ap2 = q/m * np.cross(vp2, B(rp2,ROdip,mu))

        rp3 = rp[i-1,:] + 0.5*vp2*dt
        vp3 = vp[i-1,:] + 0.5*ap2*dt
        ap3 = q/m* np.cross(vp3, B(rp3,ROdip,mu))

        rp4 = rp[i-1,:] + vp3*dt
        vp4 = vp[i-1,:] + ap3*dt
        ap4 = q/m * np.cross(vp4, B(rp4,ROdip,mu))

        rp[i] = rp[i-1,:]+(dt/6.0)*(vp1+2*vp2+2*vp3+vp4)
        vp[i] = vp[i-1,:] + (dt/6.0)*(ap1+2*ap2+2*ap3+ap4)
        t[i] = dt*i
    return rp, vp, t

#=== Trajectory visualization ===

def plot_trajectory(rp):
    plt.plot(rp[:,0], rp[:,1], rp[:,2])
    plt.xlabel('X [Earth Radii]')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rp[:,0], rp[:,1], rp[:,2])
    ax.set_xlabel('X [Earth Radii]')
    ax.set_ylabel('Y [Earth Radii]')
    ax.set_zlabel('Z [Earth Radii]')