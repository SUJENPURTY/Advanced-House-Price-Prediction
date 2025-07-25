# 🏡 Advanced-House-Price-Prediction
California House Price Prediction

This project aims to build an end-to-end machine learning pipeline to predict house prices using advanced regression techniques, data preprocessing, and model deployment.

---

## 📁 Project Structure

Advanced-House-Price-Prediction/
├── artifacts/ # Contains model files and datasets
├── Notebook/ # Jupyter notebooks (EDA, training, etc.)
├── src/ # Source code
│ ├── components/ # Core components: ingestion, transformation, training
│ └── pipeline/ # Training and prediction pipeline
├── templates/ # HTML templates for frontend
├── static/ # CSS/JS or static files for Flask
├── app.py # Flask application for model deployment
├── requirements.txt # Project dependencies
├── setup.py # Setup script for packaging
└── .gitattributes # Git LFS file tracking



---

## 🚀 Features

- Data ingestion and cleaning
- Feature engineering and preprocessing
- Model training (with Hyperparameter tuning)
- Evaluation metrics and visualization
- Model serialization using `.pkl`
- Web interface with Flask for predictions
- Git LFS used for managing large files (`.pkl`, `.csv`)

---

## 🔧 Installation

```bash
git clone https://github.com/SUJENPURTY/Advanced-House-Price-Prediction.git
cd Advanced-House-Price-Prediction
pip install -r requirements.txt

⚠️ Note: Ensure Git LFS is installed to fetch large files (models, datasets).

🧠 Tech Stack
Python 🐍

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn

Flask

Git LFS

VSCode / Jupyter


python app.py

Open http://127.0.0.1:5000/ in your browser to use the prediction interface.

📊 Example
Sample input fields:

Number of bedrooms, bathrooms, population

Location-based features


Model returns: Predicted House Price

📦 Deployment

You can deploy the app on:

Heroku

Render

or use Docker + AWS/GCP

📜 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Sujen Purty
Feel free to connect: GitHub








