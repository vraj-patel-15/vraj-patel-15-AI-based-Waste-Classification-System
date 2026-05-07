# ♻️ AI Waste Classifier
An AI-powered web application that classifies waste materials into categories like plastic, paper, glass, etc., and provides recycling guidance. Built using TensorFlow and Streamlit, this project promotes smart waste segregation and environmental awareness.

## 🚀 Features
* 📷 Upload an image of waste
* 🤖 AI-based classification using deep learning
* 🧠 Model trained on 6 categories:
  * Cardboard
  * Glass
  * Metal
  * Paper
  * Plastic
  * Trash
* 📊 Displays:
  * Predicted class
  * Confidence score
  * Probability breakdown (bar chart)
* ♻️ Provides recycling category (Recyclable / Non-Recyclable)
* 💡 Shows helpful recycling tips

## 🖥️ Tech Stack
* Frontend/UI: Streamlit
* Backend/Model: TensorFlow / Keras
* Model Architecture: MobileNetV2 (Transfer Learning)
* Libraries:
  * NumPy
  * PIL (Python Imaging Library)
  * Matplotlib / Seaborn (for visualization)
  * Scikit-learn (evaluation metrics)

## 📂 Project Structure
├── app.py                 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Streamlit UI<br>
├── model.keras            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;Trained model<br>
├── dataset/               &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;Dataset directory<br>
├── training_script.ipynb  &emsp;&emsp;&emsp;&ensp;Model training code<br>
└── README.md              &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;Project documentation<br>

## ⚙️ How It Works
1. User uploads an image via the Streamlit UI
2. Image is preprocessed (resized to 224x224, normalized)
3. Model predicts probabilities for each class
4. Highest probability class is selected
5. Recycling category and tip are displayed

## 🧠 Model Details
* Base Model: MobileNetV2 (pre-trained on ImageNet)
* Input Size: 224 × 224 × 3
* Output Layer: Dense (6 units, Softmax)
* Training Setup:
  * Loss: Categorical Crossentropy
  * Optimizer: Adam
  * Epochs: 10
  * Validation Split: 20%

## 📊 Model Evaluation
* Accuracy and loss tracked during training
* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix visualization
* F1-score comparison across categories

## 📦 Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/vraj-patel-15/AI-based-Waste-Classification-System.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 📸 Usage
1. Open the web app in your browser
2. Upload an image (jpg/png)
3. View:
   * Predicted material
   * Confidence score
   * Recycling suggestion
   * Probability chart

## 📌 Dataset
* Based on TrashNet dataset (resized version)
* Organized into 6 folders (one per class)

## 🙌 Acknowledgements
* TensorFlow & Keras for deep learning framework
* Streamlit for easy UI deployment
* TrashNet dataset for training data
