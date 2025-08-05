🧵 Fabric-Pattern-Sense-Classifier

Pattern Sense – Fabric Pattern Classification Using Deep Learning

Pattern Sense is a deep learning-based project focused on classifying five types of fabric patterns using image data.
It is built using transfer learning with MobileNetV2 to achieve high classification accuracy.

✅ Part of: SmartInternz Internship Program
✅ Tech Stack: TensorFlow, Keras, Flask
✅ Model: MobileNetV2 (pretrained, fine-tuned)
✅ Dataset: TFD Textile Dataset

🔗 Demo Link:
📽️ https://drive.google.com/file/d/1UrIZTjKhMTg9V_ktF3MM3wJ9PVCXUaDJ/view?usp=drive_link




📁 Dataset Info

The dataset contains images of 5 fabric pattern categories:

🌸 Floral

🧵 Checks

🎞️ Stripes

⚪ Dots

🔳 Plain / Solid


Each category contains ~160 high-resolution images.




🧠 Model Architecture

* Base Model: MobileNetV2 (Transfer Learning, pretrained on ImageNet)

* Input Shape: 128x128x3

* Classification Head: Custom Fully Connected Layers

*Output: 5-Class Softmax Layer

* Optimizer: Adam

* Loss Function: Categorical Crossentropy

* Metric: Accuracy





🚀 How to Run the Project Locally

1. Clone the repository:

git clone https://github.com/afreenparveenshaik3335/Fabric_Patternsence_project.git
cd Fabric_Patternsence_project

2. Install dependencies:

pip install -r requirements.txt

3. Run the Flask app:

python app.py

4. Open in browser:

Go to 👉 http://localhost:5000




📂 Project Structure

Fabric_Patternsence_project/
│
├── static/
│   ├── css/
│   │   └── style.css             # Frontend CSS styling
│   └── uploads/                  # Uploaded images for prediction
│
├── templates/
│   ├── index.html                # Homepage template
│   └── result.html               # Results page after prediction
│
├── model/
│   └── fabric_classifier.h5      # Trained MobileNetV2 model
│
├── app.py                        # Flask backend application
├── train_model.ipynb            # Jupyter Notebook for training
├── requirements.txt             # Python dependencies
├── README.md                    # Project README
└── .gitignore                   # Git ignored files




👨‍💻 Contributors

👩‍💻 Meghana9160

👩‍💻 Niharika7346

👨‍💻 Shaik-Kabeer-max
