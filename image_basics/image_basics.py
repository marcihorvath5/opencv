import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve


# def download_and_unzip(url, save_path):
#     print(f"Downloading and extracting assets....", end="")
#
#     # Downloading zip file using urllib package.
#     urlretrieve(url, save_path)
#
#     try:
#         # Extracting zip file using the zipfile package.
#         with ZipFile(save_path) as z:
#             # Extract ZIP file contents in the same directory.
#             z.extractall(os.path.split(save_path)[0])
#
#         print("Done")
#
#     except Exception as e:
#         print("\nInvalid file.", e)
#
# URL = r"https://www.dropbox.com/s/qhhlqcica1nvtaw/opencv_bootcamp_assets_NB1.zip?dl=1"
#
# asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_NB1.zip")
#
# if not os.path.exists(asset_zip_path):
#     download_and_unzip(URL, asset_zip_path)


#reading images
# img = cv.imread("checkerboard_18x18.png")
# cv.imshow('kép', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

cb_img = cv.imread("image_basics/checkerboard_18x18.png", 0)
print(cb_img)
print({f"Image size is:{cb_img.shape}"})
print(f"Data type of image is {cb_img.dtype}")

# plt.imshow(cb_img)
# plt.show()

# plt.imshow(cb_img,cmap='gray')
# plt.show()

cb_img_fuzzy = cv.imread("image_basics/checkerboard_fuzzy_18x18.jpg", 0)
print(cb_img_fuzzy)
# plt.imshow(cb_img_fuzzy,cmap='gray')
# plt.show()

coke_img = cv.imread("image_basics/coca-cola-logo.png", 1)
print(f"Image size is(H,W,C):{coke_img.shape}")
print(f"Data type of image is {coke_img.dtype}")
# plt.imshow(coke_img)
# plt.show()

coke_img_channels_reversed = coke_img[:,:,::-1]
# plt.imshow(coke_img_channels_reversed)
# plt.show()

img_NZ_bgr = cv.imread("image_basics/New_Zealand_Lake.jpg", cv.IMREAD_COLOR)
b,g,r = cv.split(img_NZ_bgr)

plt.figure(figsize=[20, 5])
plt.subplot(141); plt.imshow(r, cmap="gray");plt.title("Red Channel")
plt.subplot(142); plt.imshow(g, cmap="gray");plt.title("Green Channel")
plt.subplot(143); plt.imshow(b, cmap="gray");plt.title("Blue Channel")
img_Merged = cv.merge([b,g,r])
plt.subplot(144)
plt.imshow(img_Merged[:,:,::-1])
plt.title("Merged Output")
plt.show()

img_NZ_rgb = cv.cvtColor(img_NZ_bgr, cv.COLOR_BGR2RGB)
plt.imshow(img_NZ_rgb)

img_hsv = cv.cvtColor(img_NZ_bgr, cv.COLOR_BGR2HSV)

# Split the image into the B,G,R components
h,s,v = cv.split(img_hsv)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel")
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel")
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel")
plt.subplot(144);plt.imshow(img_NZ_rgb);   plt.title("Original")

plt.show()

h_new = h + 10
img_NZ_merged = cv.merge((h_new, s, v))
img_NZ_rgb = cv.cvtColor(img_NZ_merged, cv.COLOR_HSV2RGB)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel")
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel")
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel")
plt.subplot(144);plt.imshow(img_NZ_rgb);   plt.title("Modified")

plt.show()