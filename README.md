# Fabric-Pattern-Sense-Classifier
Pattern Sense - Fabric Pattern Classification Using Deep Learning
Pattern Sense is a deep learning-based project focused on classifying five types of fabric patterns using image data. This project is part of the SmartInternz Internship program and uses transfer learning with MobileNetV2 to achieve accurate classification.

🔗 Demo Link:
Watch Demo : https://drive.google.com/file/d/1FlGOm3sOVZfmPp7waAXLGsfHOuMbZF5T/view

📌 Project Overview

*Project Title: Pattern Sense - AI-Powered Fabric Pattern Classification Using Transfer Learning
*Domain: Computer Vision, Deep Learning
*Frameworks: TensorFlow, Keras, Flask
*Model: MobileNetV2 (Transfer Learning)
*Dataset: TFD Textile Dataset

📂 Dataset Info
The dataset contains images of 5 fabric pattern categories:

*Floral

*Checks


*Stripes


*Dots
*Plain/Solid
*Each class contains ~160 high-resolution images of fabric patterns.

🧠 Model Architecture

We use MobileNetV2 with transfer learning for feature extraction and a custom fully connected head for classification.
Input shape: 128x128x3
Output: 5-class softmax layer
Optimizer: Adam
Loss: Categorical Crossentropy
Metrics: Accuracy

🚀 How to Run
1.Clone the repository:

git clone https://github.com/afreenparveenshaik3335/Fabric_Patternsence_project.git
cd Fabric_Patternsence_project

2.Install dependencies:

pip install -r requirements.txt

3.Run the Flask app:

python app.py

4.Open in browser:
Go to 👉 http://localhost:5000


- Contributed by Meghana9160
- Contributed by Niharika7346
- Contributed by Shaik-Kabeer-max

