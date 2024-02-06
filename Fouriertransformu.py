import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_low_pass_filter(image, cutoff_frequency=30):
    # Fourier transform'u uygula
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Fourier transformunun görüntüyü temsil ettiği frekans uzayında bir alanı sıfırla
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    f_transform_shifted[crow - cutoff_frequency:crow + cutoff_frequency,
                        ccol - cutoff_frequency:ccol + cutoff_frequency] = 0

    # Inverse Fourier transform'u uygula
    f_transform_inverse_shifted = np.fft.ifftshift(f_transform_shifted)
    image_restored = np.fft.ifft2(f_transform_inverse_shifted).real

    return image_restored

# Gürültülü görüntüyü oku
image_noisy = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Dosya okuma başarısız olduysa hatayı işle
if image_noisy is None:
    print("Dosya okuma hatası: image_noisy.jpg bulunamadı.")
else:
    # Veri tipini kontrol et ve gerekiyorsa dönüştür
    if image_noisy.dtype != np.float64:
        image_noisy = image_noisy.astype(np.float64)

    # Alçak geçiren filtre ile düzelt
    restored_image = apply_low_pass_filter(image_noisy)

    # Sonuçları göster
    plt.subplot(1, 2, 1), plt.imshow(image_noisy, cmap='gray'), plt.title('Gürültülü Görüntü')
    plt.subplot(1, 2, 2), plt.imshow(restored_image, cmap='gray'), plt.title('Alçak Geçiren Filtre Uygulandı')
    plt.show()
