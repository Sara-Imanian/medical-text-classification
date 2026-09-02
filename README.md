# Medical-text-classification

A Natural Language Processing (NLP) project that recommends the most appropriate medical category based on a patient's symptom description using TensorFlow/Keras and a Flask web application.

## Medical categories

- Cardiac
- Cosmetic
- Dental
- Eye
- Fertility
- Neurology
- Orthopedic

## Features

- Medical text preprocessing
- Multi-class text classification
- Word embeddings using the Keras Embedding layer
- Automatic medical package recommendation
- Flask REST API
- Simple web interface for real-time predictions

## Tech Stack

- Python
- TensorFlow / Keras
- Flask
- Scikit-learn
- Pandas
- NumPy

## Dataset

- 420 medical text samples
- 7 medical categories
- 60 samples per category

## Project Structure

```text
medical-text-classification/
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── medical_package_model.keras
├── tokenizer.pkl
├── encoder.pkl
├── medical_comments_dataset.csv
├── requirements.txt
└── README.md
```

## Web Application

Users can enter a medical description in the web interface, and the trained NLP model predicts the most appropriate medical category.

## API Example

### Request

```http
POST /predict
```

```json
{
    "message": "I have severe eye pain and blurry vision"
}
```

### Response

```json
{
    "category": "Eye"
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

## Run the Application

```bash
python app.py
```

Then open your browser and visit:

```text
http://127.0.0.1:5000
```

## Future Improvements

- Expand the dataset
- Improve text preprocessing
- Display prediction confidence scores
- Experiment with LSTM and Bidirectional LSTM
- Deploy the application
- Docker support
