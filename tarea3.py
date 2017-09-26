from skimage import io
from skimage.color import rgb2gray
from skimage.filters.rank import entropy, gradient
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import sys

def get(y, x, t):
    m = len(t) #rows
    n = len(t[0]) #cols
    if x < 0 or x >= n: return sys.maxsize
    if y < 0 or y >= m: return sys.maxsize
    return t[y][x]
    
## Transform the image to gray
## Calculate the entropy or gradient
## Get the path with less energy
## Return the energy as an integer, and the 
## path with less energy as a list of (x,y)

def test(img):
    # filas = m, columnas = n
    m, n = img.shape
    #print(m,n)
    energy_matrix = img
    tot = np.zeros((m,n))
    tot[0] = energy_matrix[0]
    for i in range(1, m):
        tot[i][0] = min(tot[i-1][0], tot[i-1][1]) + energy_matrix[i][0]

        tot[i][n-1] = min(tot[i-1][n-1], tot[i-1][n-2]) + energy_matrix[i][n-1]
        for j in range(1, n-1):
            tot[i][j] = min(tot[i-1][j], tot[i-1][j-1], tot[i-1][j+1]) + energy_matrix[i][j]

    #print(energy_matrix)
    return tot

def energy(img,tot):
	m,n=img.shape
	mi=tot[-1][0]
	ans = [] #matriz --> 1pixel,energia)
	    
	for x in range(n):
	   	if tot[-1][x]<=mi:
	   		mi=tot[-1][x]
	   		pos=x
	ans.append((n-1,pos))
	e=img[-1][pos]
	pond=e
	for x in range(n):
	   	if tot[n-(x+1)][x]<=mi:
	   		mi=tot[n-(x+1)][x]
	   		pos=x
	for x in range(m-2,0,-1):	
		if (tot[x-1][pos])-e==img[x][pos]:
			ans.append((x,pos))
			e=img[x][pos]
			pond+=e
			continue

		if pos!=0:
			if (tot[x-1][pos-1])-e==img[x][pos]:
				ans.append((x,pos-1))
				e=img[x][pos]
				pos=pos-1
				pond+=e
				continue

		if pos!=n-1:
			if (tot[x-1][pos+1])-e==img[x][pos]:
				ans.append((x,pos+1))
				e=img[x][pos]
				pos=pos+1
				pond+=e
				continue

	if tot[1][pos]-img[1][pos]==tot[0][pos]:
		ans.append((0,pos))
		pond+=tot[0][pos]
	if pos!=(n-1):
		if tot[1][pos]-img[1][pos]==tot[0][pos+1]:
			ans.append((0,pos+1))
			pond+=tot[0][pos+1]
	if pos!=0:
		if tot[1][pos]-img[1][pos]==tot[0][pos-1]:
			ans.append((0,pos-1))
			pond+=tot[0][pos-1]
	return e, ans



def togray(image):
    image_bw = rgb2gray(image)
    # using the entropy
    #image_e = entropy(image_bw,disk(1))
    #return energy(image_e)
    
    # using gradient
    image_g = gradient(image_bw,disk(1))
    return image_g

## Remove one pixel per row... the one
## in the path min energy
def remove(image, pixels):
    ## Debe remover el camino con menor energia
    m,n,r= image.shape
    aux=[]
    for x in range(n):
        for y in range(m):
            aux.append((y,x))
    for x in range(len(aux)):
        if aux[x%m][0] and aux[x%m][y]==pixels[x]:
            del aux[x%m]
    image=np.array(aux)
    

if __name__=='__main__':
    import sys
    
    image = io.imread('image.png')
    img_gray = togray(image)
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(img_gray)
    plt.colorbar()
    plt.show()
    imgplot=plt.imshow(img_gray)
    a=test(img_gray)
    energy(img_gray,a)

    percent = 0.75
    
    m,n,_ = image.shape
    new_n = int(n * percent)
    
    img = image
    ims = []
    for i in range(n-new_n):
        print("Iteracion numero {}/{}".format(i+1, n-new_n))
        img_gray = togray(img)
        e, p = energy(img_gray,a)
        img_new = remove(img, p)
        
        img = img_new
  
    plt.figure()
    plt.imshow(image) # imagen original
    plt.figure()
    plt.imshow(img) # imagen escalada
    plt.show()
