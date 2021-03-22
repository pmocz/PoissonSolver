import matplotlib.pyplot as plt
import numpy as np

"""
Simple Poisson Solver
Philip Mocz (2021), Princeton University

Assumes cubic box, periodic BCs

"""


def getpotential( rho, rhobar, Lbox, G ):
	
	N = rho.shape[0]

	# fourier space variables
	klin = (2*np.pi/Lbox) * np.arange(-N/2,N/2)
	kx, ky, kz = np.meshgrid(klin, klin, klin)
	kSq = kx**2 + ky**2 + kz**2
	kSq = np.fft.fftshift(kSq)
	kSq[kSq==0]=1

	# Poisson solver
	V = -(np.fft.fftn( 4*np.pi*G * (rho-rhobar) )) / kSq  # (Vhat)
	V = np.fft.ifftn(V)

	# normalize so mean potential is 0
	V -= np.mean(V)
	
	V = np.real(V)
	#print(np.min(V))
	#print(np.max(V))
	
	return V



def main():
	
	N = 128
	Lbox = 20
	G = 1
	
	# Set up a density field
	xlin = np.linspace(0,Lbox,N+1)[0:N]
	
	xx, yy, zz = np.meshgrid(xlin, xlin, xlin)
	
	rho = 1 + np.sin( 2*np.pi/Lbox * xx * 2 )**2 * np.sin( 2*np.pi/Lbox * yy * 1 )**2 + 0.1*np.sin( 2*np.pi/Lbox * xx * 4 ) + 0.1*np.sin( 2*np.pi/Lbox * yy * 8 )
	rhobar = np.mean(rho)
	
	# Calculate Potential
	V = getpotential( rho, rhobar, Lbox, G )
	
	# Plot
	fig, (ax1, ax2) = plt.subplots(2, 1)
	fig.suptitle('Density and Potential')
	
	im1 = ax1.imshow(rho[:,:,40], vmin=1, vmax=3, cmap='viridis', aspect=1)
	cbar = fig.colorbar(im1, ax=ax1)
	
	im2 = ax2.imshow(V[:,:,40], vmin=-12, vmax=12, cmap='RdBu', aspect=1)
	cbar = fig.colorbar(im2, ax=ax2)
	
	plt.savefig('potential.pdf', bbox_inches='tight')
	plt.show()
	
	
	

if __name__ == "__main__":
    main()
