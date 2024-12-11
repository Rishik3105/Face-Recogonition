# 😊 Face Recognition Using OpenCV and LBPH Face Recognizer

Welcome to my **Face Recognition Project** powered by OpenCV! 🖼️📸 This project uses the **Local Binary Pattern Histogram (LBPH) Face Recognizer** to identify faces from a trained dataset and detect them in new images. 🤖

---

## 🚀 Features
✨ **Face Training**: Train the LBPH face recognizer on a custom dataset of images.  
🧩 **Face Detection**: Use a Haar Cascade classifier to detect faces in an image.  
🎯 **Face Recognition**: Recognize faces from the trained model and display the name of the detected person.  
⚙️ **Customizable Parameters**: Adjust `scaleFactor` and `minNeighbors` for face detection based on image quality and size.  
💾 **Model Persistence**: Save and load trained models for future predictions.

---

## 🛠️ Installation 🐍
Make sure you have the following dependencies installed:

- **Python 3.x** 🐍
- **OpenCV** 🖥️
- **NumPy** 🔢

Install required libraries using pip:
```bash
pip install opencv-python
pip install numpy
```

---

## 📂 Project Files Overview 🗂️
Here is the structure of the project:

- 📝 `face_trained.yml` - The trained face recognizer model.
- 💡 `features.npy` - Saved features (face regions).
- 🏷️ `labels.npy` - Saved labels (person identifiers).
- 🖼️ `haar_face.xml` - Haar Cascade Classifier XML file for face detection.

---

## 🧩 Code Walkthrough 📝

### 1. **Dataset and Training** 🎓
The dataset contains folders for each person, where each folder has images of the respective individual. Here’s an example structure:
```
Train/
    Ben Afflek/
        1.jpg
        2.jpg
    Elton John/
        1.jpg
        2.jpg
```
The code loops through each folder, detects faces using Haar Cascade, and stores face regions as **features** and their corresponding labels.

**Haar Cascade Classifier** 🖥️:
```python
haar_cascade = cv.CascadeClassifier('D:\Python\opeancv\haar_face.xml')
```
- The `haar_face.xml` file is a pre-trained model used for detecting objects (in this case, faces) based on features extracted from a large dataset.
- 🔧 **scaleFactor**: Reduces the image size at each scale. Start with `1.1` and adjust based on your image.
- 🧩 **minNeighbors**: Defines how many neighboring rectangles should be considered a valid face. Adjust for fewer false positives.

Face detection loop during training:
```python
faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)
for (x, y, w, h) in faces_rect:
    faces_roi = gray_img[y:y+h, x:x+w]
    features.append(faces_roi)
    labels.append(label)
```

### 2. **Training the Recognizer** 🏋️‍♂️
Once the features and labels are collected, the LBPH face recognizer is trained:
```python
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
```
The trained model is saved for future use:
```python
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
```

### 3. **Face Recognition** 🧠
For recognition, the trained model predicts the face label and its confidence score:
```python
label, confidence = face_recognizer.predict(faces_roi)
print(f'Label = {people[label]} with a confidence of {confidence}')
```
Detected faces are highlighted with rectangles and labeled with names:
```python
cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

---

## 🎨 Example Output 📷
- **Training Phase**: Images are processed, and features are saved. ✅
- **Recognition Phase**:

    ![Detected Face Placeholder](https://via.placeholder.com/300x200.png?text=Detected+Face)

- Output in terminal:
    ```bash
    Label = Ben Afflek with a confidence of 57.34
    ```

---

## 📝 Notes 📌
1. **Haar Cascade XML File**: The `haar_face.xml` file is essential for face detection. It is a pre-trained classifier that uses Haar-like features to detect faces efficiently.
2. **Parameter Tuning**:
   - 🔧 **scaleFactor**: Start with `1.1`. Increase it to speed up detection but reduce accuracy.
   - 🧩 **minNeighbors**: Start with `4`. Higher values reduce false positives but might miss faces.
3. **LBPH Face Recognizer**: LBPH works by analyzing pixel patterns in localized grids of the image, making it robust to lighting variations and simple image changes.

---

## 🧑‍💻 About Me 🙋‍♂️
👋 Hi, I’m **Nimmani Rishik**, a passionate developer working on **AI, Deep Learning, and Computer Vision**. This project is a stepping stone toward advanced face recognition systems. Let’s connect:

📧 [Email](mailto:nimmanirishik@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/nimmani-rishik-66b632287)  
📷 [Instagram](https://instagram.com/rishik_3142)  

Feel free to reach out for collaborations or improvements! 💬🎯

---

## ⭐ Contribution 🤝
If you find this project helpful, don’t forget to **star ⭐** the repository. Contributions are welcome! Fork, improve, and submit a pull request.

---

## 📜 License 🛡️
This project is licensed under the MIT License.

---

**Happy Coding! 💻🎉**
