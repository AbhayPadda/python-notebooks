
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import glob
import os
import imutils

# For reading text from images
from PIL import Image
import pytesseract


# In[2]:


rotated_path  = 'test/0-'
denoised_path = 'test/1-'
bilateral_path = 'test/2-'
denoised_text_path = 'test/3-'
bilateral_text_path = 'test/4-'


# In[3]:


def getDenoisedImage(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # adjust contrast
    img = cv2.multiply(img, 1.2)
    
    # create a kernel for the erode() function
    kernel = np.ones((1, 1), np.uint8)

    # erode() the image to bolden the text
    img = cv2.erode(img, kernel, iterations=5)
    
    # Remove noise from Image
    denoised = cv2.fastNlMeansDenoising(img, None, 28, 11, 21)

    # Decrease the contrast
    denoised = cv2.multiply(denoised, 0.9)
    
    #erode() to further refine the characters
    denoised = cv2.erode(denoised, kernel, iterations=5)
    
    #dilate() to increase character intensity
    denoised = cv2.dilate(denoised, kernel, iterations=5)

    return denoised

def rotateImage(image):
    edges = cv2.Canny(image, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    count = 0
    for line in lines:
        for rho,theta in line:
            if (theta > 1.5):
                break
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            
            if abs(y2-y1) > abs(x2-x1):
                count = count + 1
    if count > (0.40 * len(lines)):
        image = imutils.rotate_bound(image, 90)
    return image
    
def writeToFile(fileText, filePath):
    file = open(filePath,'w') 
    file.write(fileText)
    file.close()


# In[4]:


for path in glob.iglob('test/original/*', recursive=True):
    file_name = os.path.splitext(os.path.basename(path))[0]
    
    denoised = getDenoisedImage(path)
    rotated_image = rotateImage(denoised)
    
    # save denoised and rotated Image
    cv2.imwrite((denoised_path + file_name + '.jpg'), rotated_image)
    
    # Apply Bilateral filter to the Image and dilate it again to further bold the characters
    kernel = np.ones((1, 1), np.uint8)
    bilateral = cv2.bilateralFilter(denoised, 9, 80, 80)
    bilateral = cv2.dilate(bilateral, kernel, iterations=2)

    cv2.imwrite((bilateral_path + file_name + '.jpg'), bilateral)
    
    # Read text from denoised Image
    denoised_image_text = pytesseract.image_to_string(Image.open(denoised_path + file_name + '.jpg'))
    file_path = denoised_text_path + file_name + '.txt'
    writeToFile(denoised_image_text, file_path)
    
    bilateral_image_text = pytesseract.image_to_string(Image.open(bilateral_path + file_name + '.jpg'))
    file_path = bilateral_text_path + file_name + '.txt'
    writeToFile(bilateral_image_text, file_path)

