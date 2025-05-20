# Ex. No. 01 Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** Krithick Vivekananda  
- **Register Number:** 212223240075



#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
img = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_GRAYSCALE)
```

#### 2. Print the image width, height & Channel.
```python
image = cv2.imread('Eagle_in_Flight.jpg')
print("Height, Width and Channel:", image.shape)
```
![image](https://github.com/user-attachments/assets/c4ec201e-503f-48b1-b14a-d5535e46ba95)

#### 3. Display the image using matplotlib imshow().
```python
import matplotlib.pyplot as plt
plt.imshow(img)
```
![image](https://github.com/user-attachments/assets/3c8c2261-8cdc-417e-8cf3-0cff7de8677f)

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight.png', image)
```
![image](https://github.com/user-attachments/assets/85628686-588e-4801-b548-b50e71140479)

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img = cv2.imread('Eagle_in_Flight.png')
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(color_img)
```
![image](https://github.com/user-attachments/assets/6e033e21-12a4-4e77-b0a1-bea8ef750e38)

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(color_img)
color_img.shape
```
![image](https://github.com/user-attachments/assets/3eeb88cc-83cb-42b7-866e-79b2bd3d81f5)
![image](https://github.com/user-attachments/assets/6c1c6820-7461-4767-8d7b-5737f244a54e)

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
cropped = color_img[10:450, 150:570]
plt.imshow(cropped)
plt.axis("off")
```
![image](https://github.com/user-attachments/assets/d057ad63-535a-416e-bb5c-3084c2eed0db)

#### 8. Resize the image up by a factor of 2x.
```python
height, width = image.shape[:2]
resized_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
plt.imshow(resized_image)
```
![image](https://github.com/user-attachments/assets/a8318582-0c89-4150-9e6d-f5029894d604)

#### 9. Flip the cropped/resized image horizontally.
```python
flipped = cv2.flip(resized_image, 1)
plt.imshow(flipped)
```
![image](https://github.com/user-attachments/assets/dfa9e3ae-f12f-406e-8e08-146f9314a6d2)

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
img_apollo = cv2.imread('Apollo-11-launch.jpg')
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
cv2.putText(img_apollo, 'Apollo 11 Saturn V Launch, July 16, 1969', (50, img_apollo.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
plt.imshow(img_apollo)
```
![image](https://github.com/user-attachments/assets/a10543e6-0c48-48eb-856e-107b0fc44e65)

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
cv2.rectangle(img_apollo, (400, 30), (750, 600), (255, 0, 255), 3)
```
![image](https://github.com/user-attachments/assets/be70d0d9-15b9-4460-b96a-2c273b66f85f)

#### 13. Display the final annotated image.
```python
plt.imshow(img_apollo)
```
![image](https://github.com/user-attachments/assets/50de615a-ff67-4845-90e4-60f502633998)

#### 14. Read the image ('Boy.jpg').
```python
boy_img = cv2.imread('Boy.jpg')
```

#### 15. Adjust the brightness of the image.
```python
# Create a matrix of ones (with data type float64)
import numpy as np
matrix_ones = np.ones(boy_img.shape, dtype='uint8') * 50
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(boy_img, matrix)
img_darker = cv2.subtract(boy_img, matrix)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize=(10, 3))
for i, img in enumerate([boy_img, img_darker, img_brighter]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```
![image](https://github.com/user-attachments/assets/2b19e2bc-d517-4d98-9536-413fdfd09cc6)

#### 18. Modify the image contrast.
```python
# Create two higher contrast images using the 'scale' option with factors of 1.1 and 1.2 (without overflow fix)
matrix1 = np.ones(boy_img.shape, dtype='uint8') * 25
matrix2 = np.ones(boy_img.shape, dtype='uint8') * 50
img_higher1 = cv2.addWeighted(boy_img, 1.1, matrix1, 0, 0)
img_higher2 = cv2.addWeighted(boy_img, 1.2, matrix2, 0, 0)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize=(10, 3))
for i, img in enumerate([boy_img, img_higher1, img_higher2]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```
![image](https://github.com/user-attachments/assets/7cdd89c0-1e51-441a-9c7a-ad81aa929cb9)

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(boy_img)
plt.figure(figsize=(10, 3))
for i, channel in enumerate([b, g, r]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channel, cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/a4dd29f6-248e-46f0-919d-5335846bd7af)

#### 21. Merged the R, G, B , displays along with the original image
```python
merged = cv2.merge([b, g, r])
plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
plt.show()
```
![image](https://github.com/user-attachments/assets/84c3b456-fcc9-4879-99f4-5b0f01f3c170)

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv = cv2.cvtColor(boy_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
plt.figure(figsize=(10, 3))
for i, channel in enumerate([h, s, v]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channel, cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/2e2dc6fb-6436-4c66-b5b9-3b7c150c38bf)

#### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.merge([h, s, v])
plt.imshow(cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB))
plt.show()
```
![image](https://github.com/user-attachments/assets/da94bbac-6869-493d-8f94-57a940f1d8b1)

#### 24. Image Generation by bitwise operations
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images
boy_img = cv2.imread('Boy.jpg')
apollo_img = cv2.imread('Apollo-11-launch.jpg')

# Resize images to the same dimensions
boy_img = cv2.resize(boy_img, (500, 500))
apollo_img = cv2.resize(apollo_img, (500, 500))

# Perform bitwise AND operation
bitwise_and_img = cv2.bitwise_and(boy_img, apollo_img)

# Display images
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(boy_img, cv2.COLOR_BGR2RGB))
plt.title('Boy Image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(apollo_img, cv2.COLOR_BGR2RGB))
plt.title('Apollo Image')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(bitwise_and_img, cv2.COLOR_BGR2RGB))
plt.title('Bitwise AND')

plt.show()
```
![image](https://github.com/user-attachments/assets/0663e1e5-dd64-414b-9077-705b69d06a91)

## Output:
- **i)** Read and Display an Image.
   ![image](https://github.com/user-attachments/assets/f5a133f2-dcff-421d-879e-1bce6ca0a0e5)
- **ii)** Adjust Image Brightness.
  ![image](https://github.com/user-attachments/assets/2db61cae-676d-46a6-871e-c21dad4b02f2)
- **iii)** Modify Image Contrast.
  ![image](https://github.com/user-attachments/assets/46aa72ac-9f14-4122-aca0-c5ec1ec30795)
- **iv)** Generate Third Image Using Bitwise Operations.
  ![image](https://github.com/user-attachments/assets/2ea7e09f-3382-4f28-a72c-395e78f184d4)

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

