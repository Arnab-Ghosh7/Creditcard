# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using a Random Forest classifier. This project includes a complete ML pipeline with data processing, model training, evaluation, and deployment options via FastAPI and Streamlit.

## 🎯 Features

- **Machine Learning Pipeline**: End-to-end ML workflow using DVC (Data Version Control)
- **Multiple Deployment Options**: 
  - FastAPI REST API for production deployments
  - Streamlit web application for interactive predictions
  - Docker containerization support
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, precision-recall curves, and confusion matrices
- **Feature Importance Visualization**: Automated visualization of model feature importance
- **Model Versioning**: DVC-based model and data versioning
- **Cloud Integration**: AWS S3 support for model storage and deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [ML Pipeline](#ml-pipeline)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda
- (Optional) Docker for containerized deployment

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Creditcard
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For development dependencies:
   ```bash
   pip install -r dev-requirements.txt
   ```

4. **Install the project in editable mode**
   ```bash
   pip install -e .
   ```

5. **Set up DVC** (if using the ML pipeline)
   ```bash
   dvc pull  # Pull data and models from remote storage
   ```

## 📁 Project Structure

```
Creditcard/
├── LICENSE
├── Makefile              # Makefile with common commands
├── README.md            # This file
├── requirements.txt     # Production dependencies
├── dev-requirements.txt # Development dependencies
├── setup.py             # Package setup configuration
├── params.yaml          # ML pipeline parameters
├── dvc.yaml             # DVC pipeline configuration
├── Dockerfile           # Docker configuration
├── app.py               # FastAPI application (standalone)
├── app_gunicorn.py      # FastAPI application (Gunicorn-ready)
├── app_streamlit.py     # Streamlit web application
├── model.joblib         # Trained model file
│
├── data/                # Data directory
│   ├── external/        # Third-party data sources
│   ├── interim/         # Intermediate data transformations
│   ├── processed/       # Final processed datasets
│   └── raw/             # Original immutable data
│
├── docs/                # Documentation (Sphinx)
│   ├── conf.py
│   ├── index.rst
│   └── getting-started.rst
│
├── models/              # Trained models directory
│   └── model.joblib
│
├── notebooks/           # Jupyter notebooks
│   └── dummy.ipynb
│
├── references/          # Data dictionaries and references
│
├── reports/             # Generated analysis reports
│   └── figures/        # Generated graphics
│
├── src/                 # Source code
│   ├── __init__.py
│   ├── data/            # Data processing scripts
│   │   └── make_dataset.py
│   ├── features/        # Feature engineering
│   │   └── build_features.py
│   ├── models/          # Model training and deployment
│   │   ├── train_model.py
│   │   └── push_model.py
│   └── visualization/   # Visualization scripts
│       └── visualize.py
│
└── dvc_plots/           # DVC-generated plots and metrics
    └── index.html
```

## 💻 Usage

### Running the ML Pipeline

The project uses DVC for managing the ML pipeline. To run the complete pipeline:

```bash
# Run the entire pipeline
dvc repro

# Run specific stages
dvc repro make_dataset
dvc repro train_model
dvc repro visualize
```

### Using Makefile Commands

```bash
# Install dependencies
make requirements

# Process data
make data

# Lint code
make lint

# Clean compiled files
make clean

# View all available commands
make help
```

### FastAPI Application

**Standalone mode:**
```bash
python app.py
```
The API will be available at `http://localhost:8080`

**Production mode with Gunicorn:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app_gunicorn:app
```

**API Endpoints:**
- `GET /` - Health check endpoint
- `POST /predict` - Make fraud detection predictions

### Streamlit Application

```bash
streamlit run app_streamlit.py
```

The application will open in your default web browser, typically at `http://localhost:8501`

## 🔄 ML Pipeline

The project uses DVC to manage the ML pipeline with the following stages:

1. **make_dataset**: Loads raw data and splits into train/test sets
   - Input: `data/raw/creditcard.csv`
   - Output: `data/processed/train.csv`, `data/processed/test.csv`
   - Parameters: `test_split`, `seed`

2. **train_model**: Trains a Random Forest classifier
   - Input: Processed training data
   - Output: `models/model.joblib`
   - Parameters: `n_estimators`, `max_depth`, `seed`

3. **visualize**: Generates evaluation metrics and visualizations
   - Input: Trained model and test data
   - Output: Metrics, ROC curves, precision-recall curves, confusion matrices, feature importance plots

### Pipeline Parameters

Edit `params.yaml` to configure pipeline parameters:

```yaml
make_dataset:
  test_split: 0.2
  seed: 2023
train_model:
  seed: 21
  n_estimators: 50
  max_depth: 8
```

## 📡 API Documentation

### FastAPI Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
"Working fine"
```

#### Predict Fraud
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Time": 0.0,
  "V1": -1.359807134,
  "V2": -0.072781173,
  "V3": 2.536346738,
  "V4": 1.378155224,
  "V5": -0.338261769,
  "V6": 0.462387778,
  "V7": 0.239598554,
  "V8": 0.098697901,
  "V9": 0.36378697,
  "V10": 0.090794172,
  "V11": -0.551599533,
  "V12": -0.617800856,
  "V13": -0.991389847,
  "V14": -0.311169354,
  "V15": 1.468176972,
  "V16": -0.470400525,
  "V17": 0.207971242,
  "V18": 0.02579058,
  "V19": 0.40399296,
  "V20": 0.251412098,
  "V21": -0.018306778,
  "V22": 0.277837576,
  "V23": -0.11047391,
  "V24": 0.066928075,
  "V25": -0.208253515,
  "V26": -0.108300452,
  "V27": 0.005273597,
  "V28": -0.190320519,
  "Amount": 149.62
}
```

**Response:**
```json
{
  "prediction": 0
}
```

**Prediction Values:**
- `0`: Legitimate transaction
- `1`: Fraudulent transaction

### Interactive API Documentation

When running the FastAPI application, visit:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## 🐳 Deployment

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t creditcard-fraud-detection .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:8080 creditcard-fraud-detection
   ```

### AWS S3 Model Deployment

The project includes functionality to push trained models to AWS S3:

```python
python src/models/push_model.py
```

Configure your AWS credentials before running:
```bash
aws configure
```

## 🛠 Technologies

- **Machine Learning**: scikit-learn, pandas, numpy
- **Web Framework**: FastAPI, Streamlit
- **API Server**: Uvicorn, Gunicorn
- **MLOps**: DVC (Data Version Control), DVC Live
- **Visualization**: Matplotlib
- **Model Serialization**: Joblib
- **Cloud**: AWS S3, boto3
- **Containerization**: Docker
- **Documentation**: Sphinx

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 30 features (Time, V1-V28, Amount)
- **Target**: Binary classification (0: Legitimate, 1: Fraud)
- **Evaluation Metrics**: 
  - ROC-AUC Score
  - Average Precision Score
  - Precision-Recall Curves
  - Confusion Matrix

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the terms specified in the LICENSE file.

## 👤 Author

**Pranjal**

---

## 🙏 Acknowledgments

- Project structure based on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template
- Credit card fraud detection dataset

---

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
