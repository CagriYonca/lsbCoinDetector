from skimage import feature
from sklearn.svm import LinearSVC
from imutils import paths, resize

import cv2
import numpy as np

training_image_path = "/home/user/cs231/examples/pyimage/lbp/training_images"

# LBP Feature Extractor
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        
        return hist

desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# Create histograms for training data
for imagePath in paths.list_images(training_image_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 3)
    
    hist = desc.describe(gray)
    image_name = imagePath.split("-")[-2]
    labels.append(image_name.split("/")[-1])
    data.append(hist)

# Initialize Support Vector Classifier and feed with training hstograms
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

print("Model has trained")

# Start webcam capturing
cap = cv2.VideoCapture(0)

while True:
    total = 0
    _, frame = cap.read()
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gframe = cv2.medianBlur(gframe, 3) 
    
    # Detect circles in frame
    rows = gframe.shape[0]
    circles = cv2.HoughCircles(gframe, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=120, param2=30, minRadius=int(rows / 14), maxRadius=int(rows / 3))

    # Identify images in circles
    if circles is not None:
        circles = np.int32(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = (circle[0], circle[1], circle[2])
            circleROI = gframe[x - r : x + r, y - r : y + r]
            if (circleROI.shape[0] != 0) and (circleROI.shape[1] != 0):
                circleROI = cv2.equalizeHist(circleROI)     # Equalize test data
                hist = desc.describe(circleROI)
                prediction = model.predict(hist.reshape(1, -1))
                cv2.circle(frame, (x, y), 1, (0, 100, 100), 3)
                cv2.circle(frame, (x, y), r, (255, 0, 255), 3)
                cv2.putText(frame, prediction[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
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
            cv2.putText(frame, str(kurus) + " kurus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        if (kurus != 0):
            cv2.putText(frame, str(lira) + " TL " + str(kurus) + " kurus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(frame, str(lira) + " TL ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("Image", frame)
    k = cv2.waitKey(33)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
