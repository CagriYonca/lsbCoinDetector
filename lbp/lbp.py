from skimage import feature
from sklearn.svm import LinearSVC
from imutils import paths, resize

import cv2
import numpy as np

training_image_path = "/home/user/cs231/examples/pyimage/lbp/training_images"
test_image_path = "/home/user/cs231/examples/pyimage/lbp/test_images"

# LBP Feature Extractor
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-9):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        
        return hist

desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# Creating histograms for training data
for imagePath in paths.list_images(training_image_path):
    image = cv2.imread(imagePath)
    image = cv2.medianBlur(image, 5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = resize(gray, width=64)
    hist = desc.describe(gray)
    image_name = imagePath.split("-")[-2]
    labels.append(image_name.split("/")[-1])
    data.append(hist)

# Initialize and feed Support Vector Classifier with training data histograms
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

# Create histograms for test data
for imagePath in paths.list_images(test_image_path):
    total = 0
    image = cv2.imread(imagePath)
    if (image.shape[0] > image.shape[1]) and (image.shape[0] > 1000):
        image = resize(image, width = 1000)
    if (image.shape[0] < image.shape[1]) and (image.shape[1] > 1000):
        image = resize(image, height = 1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (image.shape[0] or image.shape[1]) == 1000:
        gray = cv2.medianBlur(gray, 9)
    else:
        gray = cv2.medianBlur(gray, 5)
    
    # Detect Circles in test data
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 5, param1=120, param2=30, minRadius=int(rows / 14), maxRadius=int(rows / 6))
    
    if circles is not None:
        circles = np.int32(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = (circle[0], circle[1], circle[2])
            circleROI = gray[ x - r : x + r, y - r : y + r ]
            if (circleROI.shape[0] != 0) and (circleROI.shape[1] != 0):
                if circleROI.shape[0] > 256:
                    circleROI = resize(circleROI, width=256)
                circleROI = cv2.equalizeHist(circleROI)
                hist = desc.describe(circleROI)
                prediction = model.predict(hist.reshape(1, -1))
                cv2.circle(image, (x, y), r, (255, 0, 255), 3)
                cv2.circle(image, (x, y), 1, (0, 100, 100), 3)
                cv2.putText(image, prediction[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # Calculate total money value
            if prediction == "lirabir":
                total += 100
            if prediction == "kuruselli":
                total += 50
            if prediction == "kurusyirmibes":
                total += 25
    lira = int(total / 100)
    kurus = total % 100
    if (lira == 0):
        if (kurus != 0):
            cv2.putText(image, str(kurus) + " kurus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        if (kurus != 0):
            cv2.putText(image, str(lira) + " TL " + str(kurus) + " kurus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(image, str(lira) + " TL ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
