import cv2
import numpy as np

#Kontrast Germe:
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = 255 * ((image - min_val) / (max_val - min_val))
    return stretched.astype(np.uint8)

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
stretched_image = contrast_stretching(image)
cv2.imwrite('output_contrast_stretched.jpg', stretched_image)

#Histogram Dengeleme:
def histogram_equalization(image):
    equalized = cv2.equalizeHist(image)
    return equalized

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
equalized_image = histogram_equalization(image)
cv2.imwrite('output_histogram_equalized.jpg', equalized_image)


#Alçak Geçiren Filtreleme (Gauss Filtre):
def gaussian_filter(image, kernel_size=5):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
filtered_image = gaussian_filter(image)
cv2.imwrite('output_gaussian_filtered.jpg', filtered_image)


#Medyan Filtreleme:
def median_filter(image, kernel_size=3):
    median_filtered = cv2.medianBlur(image, kernel_size)
    return median_filtered

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
median_filtered_image = median_filter(image)
cv2.imwrite('output_median_filtered.jpg', median_filtered_image)


#Görüntü Türevi ile Keskinleştirme:
def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
sharpened_image = sharpen_image(image)
cv2.imwrite('output_sharpened.jpg', sharpened_image)

