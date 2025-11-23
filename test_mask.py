from keras.models import load_model
import cv2
import numpy as np
import os

# 1️⃣ Correct absolute path to your model
model_path = r"C:\Users\Sankar\OneDrive\Desktop\mas\model\mask_detector.h5"

# Check if file exists
if not os.path.exists(model_path):
    print("Error: mask_detector.h5 not found! Check the path.")
    exit()

# 2️⃣ Load model
model = load_model(model_path)
print("Model loaded successfully ✅")

# 3️⃣ Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4️⃣ Preprocess frame for model
    img = cv2.resize(frame, (150, 150))  
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
   


    # 5️⃣ Prediction
    pred = model.predict(img)[0][0]
    if pred > 0.5:
        label = "Mask 😷"
        color = (0, 255, 0)
    else:
        label = "No Mask ❌"
        color = (0, 0, 255)

    # 6️⃣ Display result
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

