import cv2
import numpy as np
from matplotlib import pyplot as plt

def plothistogram(image) :

    plt.hist(image.ravel(), bins=256, range=(0.0, 255), fc='k', ec='k')
    print(plt.hist(image.ravel(), bins=256, range=(0.0, 255), fc='k', ec='k'))
    plt.show()

def lowpassfilter(image , dft_shift) :
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.float32)
    d_0 = 150
    _power = 2
    for i in range(0, rows):
        for j in range(0, cols):
            d = np.power(i - crow, 2) + np.power(j - ccol, 2)
            # mask[i, j, 0] = mask[i, j, 1] = 1 / np.exp( (d / ( 2*np.power(d_0, 2))))
            mask[i, j, 0] = mask[i, j, 1] = 1 / np.power((1 + (np.sqrt(d) / d_0)), 2 * _power)
        print(i)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, alpha=0, beta=252, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(r'butterwidth_lowpass_result.jpg', img_back)
    plt.figure(figsize=(14,9))
    plt.subplot(121)
    plt.imshow(image, cmap = 'gray')
    plt.title('Filtered by Median Filter')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(img_back, cmap = 'gray')
    plt.title('Butterwidth Lowpass Image')
    plt.axis('off')
    plt.savefig(r'blowpassstacked.jpg')
    plt.show()
    return img_back

def gussianlowpassfilter(image , dft_shift) :
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.float32)
    d_0 = 150
    _power = 2
    for i in range(0, rows):
        for j in range(0, cols):
            d = np.power(i - crow, 2) + np.power(j - ccol, 2)
            mask[i, j, 0] = mask[i, j, 1] = 1 / np.exp( (d / ( 2*np.power(d_0, 2))))
            #mask[i, j, 0] = mask[i, j, 1] = 1 / np.power((1 + (np.sqrt(d) / d_0)), 2 * _power)
        print(i)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, alpha=0, beta=252, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(r'gussian_lowpass_result.jpg', img_back)
    plt.figure(figsize=(14,9))
    plt.subplot(121)
    plt.imshow(image, cmap = 'gray')
    plt.title('Filtered by Median Filter')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(img_back, cmap = 'gray')
    plt.title('Gussian Lowpass Image')
    plt.axis('off')
    plt.savefig(r'glowpassstacked.jpg')
    plt.show()
    return img_back

def highpassfilter(image , dft_shift) :
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            d = np.sqrt(pow(i - crow, 2) + pow(j - ccol, 2))
            try:
                mask[i, j, 0] = mask[i, j, 1] = 1 / (1 + np.float_power(25 / d, 2 * 1))
            except ZeroDivisionError:
                mask[i, j, 0] = mask[i, j, 1] = 0
        print(i)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, alpha=0, beta=252, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(r'highpass_result.jpg', img_back)
    plt.figure(figsize=(14, 9))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Filtered by Median Filter')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(img_back, cmap='gray')
    plt.title('Highpass Image')
    plt.axis('off')
    plt.savefig(r'highpassstacked.jpg')
    plt.show()
    return img_back

def medblur(image) :
    blurred_img = cv2.medianBlur(image, 13)
    return blurred_img

def magnitude_spectrum(image_float32, img) :
    dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    fig, (ax1, ax2) = plt.subplots(figsize=(14, 9), nrows=1, ncols=2)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Input Image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.set_title('Magnitude Spectrum')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.savefig(r'magnitude_spectrum.jpg')
    plt.show()
    dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

if __name__ == '__main__':

    main_img = cv2.imread(r'Text1_Mesh.tif',0)
    float_img = np.float32(main_img)
    magnitude_spectrum(float_img, main_img)
    blured_img = medblur(main_img)
    image_float32 = np.float32(blured_img)
    dff_shift = magnitude_spectrum(image_float32, blured_img)
    img_back_low = lowpassfilter(image_float32, dff_shift)
    image_back_gussian = gussianlowpassfilter(main_img, dff_shift)
    img_back_high = highpassfilter(image_float32, dff_shift)
    final_image = main_img + img_back_low + img_back_high
    cv2.imwrite(r'final_image.jpg', final_image)
    print("final image saved")
