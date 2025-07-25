# Electricity Cost Prediction API

This project is an end-to-end machine learning application that predicts electricity costs based on building features. It demonstrates a complete workflow from data analysis and model training to the deployment of a live RESTful API on the cloud.

The core of the project is a multi-model strategy, where four independent Linear Regression models are trained to specialize in different building types, leading to more accurate and logical predictions.

---

### 🚀 Live Demo

The API is live and hosted on Render. You can interact with it using the automatically generated documentation.

*   **Live API Docs (Swagger UI):** **[https://electricity-cost-regression-predictor.onrender.com/docs](https://electricity-cost-regression-predictor.onrender.com/docs)**

---

## ✨ Features

*   **Custom Machine Learning Model:** Implements a Linear Regression model from scratch using NumPy and Gradient Descent.
*   **Multi-Model Strategy:** Deploys four specialized models, one for each building type (Residential, Commercial, Industrial, Mixed-use), for improved accuracy.
*   **Robust Data Pipeline:** A repeatable script (`train_and_save_models.py`) handles all data cleaning, feature engineering, scaling, and artifact generation.
*   **RESTful API:** A production-ready API built with FastAPI to serve predictions.
*   **Cloud Deployment:** Fully containerized and deployed on Render for public access.
*   **Large File Handling:** Uses Git LFS to manage model artifacts effectively.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.12
*   **Backend & API:** FastAPI, Uvicorn
*   **Data Science & ML:** Pandas, NumPy, Scikit-learn, Joblib
*   **Deployment & Infrastructure:** Render, Gunicorn, Git, Git LFS

---

## 📁 Project Structure

The repository is organized to separate concerns, making it clean and maintainable.

```
.
├── app/
│   ├── dataset/
│   │   └── electricity_cost_dataset.csv  # Raw data
│   ├── models/
│   │   ├── com_model.joblib            # Saved models and scalers
│   │   └── ...
│   └── main.py                         # FastAPI application source code
├── documentation/
│   └── ...                             # Project reports and formal docs
├── notebooks/
│   ├── EDA.ipynb                       # Exploratory Data Analysis
│   └── Preprocessing_and_Model.ipynb   # Initial modeling experiments
├── .gitattributes                      # Config for Git LFS
├── .gitignore
├── observations.md                     # Detailed data science observations
├── README.md                           # This file
├── requirements.txt                    # Python dependencies for deployment
└── train_and_save_models.py            # Script to generate all model artifacts
```

---

## ⚙️ Local Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Git LFS to pull model files**
    ```bash
    git lfs install
    git lfs pull
    ```
    This will download the large model files from LFS storage.

5.  **Generate Model Artifacts (Optional but Recommended)**
    To ensure you are using a consistent version of the models, you can re-generate them by running the training script from the project root:
    ```bash
    python train_and_save_models.py
    ```

6.  **Run the FastAPI Server**
    Run the server from the **project root directory**. This is crucial for the file paths to work correctly.
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will now be running locally at `http://127.0.0.1:8000`.

---

## 🔬 Project Methodology & Documentation

This project followed a structured data science workflow. Detailed observations and justifications for each step are documented in `observations.md`. The key stages are summarized below.

### 1. Exploratory Data Analysis & Key Insights
*(Summary: Initial analysis of the dataset revealed strong correlations for some features and led to the critical insight that a multi-model strategy based on `structure type` was necessary.)*

### 2. Feature Processing and Engineering
*(Summary: A pipeline was built to clean data, segregate it by structure type, perform type-specific feature selection, split for unbiased evaluation, and apply Min-Max scaling to prepare the data for Gradient Descent.)*

### 3. Model Selection and Rationale
*(Summary: A custom-built Linear Regression model was chosen as an excellent, interpretable baseline. Its manual implementation demonstrated a foundational understanding of ML principles, and its performance was optimized via hyperparameter tuning.)*

### 4. Model Performance and Evaluation
*(Summary: The final models achieved a strong performance with an overall Root Mean Squared Error (RMSE) of ~$205, which corresponds to an average error of about 8.2% relative to the mean electricity cost.)*

### 5. API Endpoints
*(Summary: The models are served via a `POST /predict` endpoint, which accepts a JSON payload and returns a cost prediction. The API includes robust error handling for invalid inputs and server-side issues.)*

---

## 💡 Challenges & Lessons Learned

Deploying this project from a local notebook to a live cloud service revealed several critical real-world challenges:
*   **The `joblib`/`pickle` Context Problem:** A model saved in one environment (e.g., a training script importing a class) contains a "recipe" that can fail in another environment (e.g., a server where the class is defined differently). The solution was to embed the class definition in both the final training script and the main application file, and use a patch (`setattr`) to make the class findable during loading.
*   **Local vs. Production Environment Parity:** The most common source of errors was the difference between running a script directly versus running it as part of a larger application from the project root. Adopting a `uvicorn app.main:app` command for local development was key to solving path-related `FileNotFoundError` issues.
*   **Dependency Management:** A bloated `requirements.txt` generated from a global environment can cause build failures. Maintaining a clean, project-specific virtual environment is essential for reliable deployments.

