import numpy as np
import matplotlib.pyplot as plt

img1 = plt.imread("tigre.png")
img = img1[:,:,0].copy() #normalni img
img2= img1[:,:,0].copy() #posvijetlit
print(img.shape)
(h,w) = img.shape
lightPlus = 0.5
for x in range(h):
        for y in range(w):
            if (img[x,y] + lightPlus) >1:
                img2[x,y] = 1
            else:
                img2[x,y]=img[x,y] + lightPlus

plt.imshow(img, cmap="gray")
plt.figure() #normalna img
# plt.imshow(img2, cmap="gray")
# plt.figure() # svijetla img


# img_rot= np.zeros((w,h))
# for i in range(0,h):
#     img_rot[:,h-1-i] = img[i,:]
# plt.imshow(img_rot, cmap="gray")
# plt.figure() 
# # rotirana img

# img_mirror = img_rot.transpose()
# plt.imshow(img_mirror, cmap="gray") 
# plt.figure()
# #mirrorana img
# rez=7
# img_lowRez = img[::rez,::rez]
# plt.imshow(img_lowRez, cmap="gray") 

# img_part= np.zeros((h,w))
# img_part[:,240:481] = img[:,240:481]
# plt.imshow(img_part, cmap="gray")
# 1/4 2.dijela img

def check(kvad,red,stup):
    blackGrid[:,:,:] = (0,0,0) # rgb grid black
    whiteGrid[:,:,:] = (255,255,255) # rgb grid black
    red0= np.stack[[blackGrid,whiteGrid]*(stupci//2)]
     
img_part= np.zeros((h,w))
plt.imshow(img_part, cmap="gray")
#nedovrseno board
plt.show()
#prikazi sve slike

    

