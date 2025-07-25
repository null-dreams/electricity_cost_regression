
# Electricity Cost Prediction: A Deep Dive into Regression Modeling

This document details the end-to-end process of building a predictive model for electricity costs, from initial data exploration to the final deployment of a robust API.

## 1. Exploratory Data Analysis (EDA) & Key Insights

The project began with a thorough exploration of the `electricity_cost_dataset.csv` to understand its structure, identify key relationships, and formulate a modeling strategy.

### 1.1. Initial Dataset Observations
A preliminary look at the dataset revealed the following:
*   **Data Quality:** The dataset is clean with no missing values, allowing us to proceed directly to analysis without imputation.
*   **Categorical Features:** The `structure type` column contains only four unique values (`Residential`, `Commercial`, `Industrial`, `Mixed-use`). This makes it a prime candidate for categorical analysis.
*   **Feature Scaling:** A significant variance was observed in the ranges of numerical columns (e.g., `site area` in the thousands vs. `utilisation rate` from 0-100). This immediately highlighted the necessity for feature scaling (normalization) before training any distance-based or gradient-based models.

### 1.2. Correlation Analysis: A First Look
A correlation heatmap was generated for all numerical columns against the target variable, `electricity cost`. This provided the first major insight into feature relevance.

| Feature Column | Correlation | Remarks |
| :--- | :---: | :--- |
| `site area` | **0.87** | 🟢 Strong positive correlation |
| `water consumption` | **0.70** | 🟢 Strong positive correlation |
| `resident count` | 0.36 | 🟡 Mild positive correlation |
| `utilisation rate` | 0.21 | 🟡 Mild positive correlation |
| `issue resolution time` | 0.04 | 🔴 No linear correlation |
| `air quality index` | 0.02 | 🔴 No linear correlation |
| `recycling rate` | -0.01 | 🔴 No linear correlation |

**Conclusion:** The features `site area`, `water_consumption`, `utilisation_rate`, and `resident_count` showed the most promise for a linear model. The other columns demonstrated near-zero linear correlation and were identified as potential noise.

### 1.3. The "Resident Count" Anomaly and the Pivot to a Multi-Model Strategy

Plotting scatterplots of the significant features against `electricity cost` revealed a critical anomaly.
*   The plot for `resident count` vs. `electricity cost` showed a dense cluster of data points with very few residents (near zero) but an extremely wide range of electricity costs.
*   This suggested that for certain `structure type`s, `resident count` was not a meaningful predictor.

Further analysis confirmed this hypothesis: `Industrial` and `Commercial` structures had high average electricity costs even with a resident count near zero. This led to the most important strategic decision of the project:

> **Instead of building one generalized model, the problem would be split into four distinct sub-problems, with a specialized model trained for each `structure type`.**

This multi-model approach allows the unique data patterns of each structure type to be learned independently, leading to more accurate and logical predictions.

## 2. Feature Processing and Engineering

Based on the EDA insights, a robust data processing pipeline was created to prepare the data for the four specialized models. This pipeline is implemented in the `train_and_save_models.py` script.

### 2.1. Initial Data Cleaning and Feature Selection

First, columns identified as having no predictive value for this regression context were dropped globally from the dataset:
*   `recycling rate`
*   `air quality index`
*   `issue resolution time`

**Rationale:** Removing these irrelevant features reduces model complexity and noise, allowing it to focus on the more impactful signals in the data.

### 2.2. Type-Specific Feature Handling

The core of the strategy was to tailor the feature set for each model. After splitting the main DataFrame by `structure type`, the following logic was applied:

*   For **Commercial** and **Industrial** models, the `resident count` column was dropped.
*   For **Residential** and **Mixed-use** models, `resident_count` was kept as a feature.

**Rationale:** This ensures that each model is trained only on features that are logically relevant to its specific domain, preventing the model from learning irrelevant correlations.

### 2.3. Data Splitting for Unbiased Evaluation

To ensure an accurate and unbiased measure of performance, each of the four processed DataFrames was split into a training set (80%) and a testing set (20%) using `sklearn.model_selection.train_test_split` with a `random_state` for reproducibility.

**Rationale:** The model is trained exclusively on the training set and evaluated on the completely unseen testing set. This simulates how the model would perform on new, real-world data and is the gold standard for preventing model overfitting.

### 2.4. Feature Scaling (Normalization)

The final preprocessing step was to scale all numerical features (and the target variable) to a `[0, 1]` range using **Min-Max Scaling**.

**Rationale:**
1.  **For Gradient Descent:** The custom `LinearRegressionGD` model uses Gradient Descent, an algorithm that converges much faster and more reliably when features are on a similar scale.
2.  **Preventing Data Leakage:** The scaling parameters (`min` and `max` values) were calculated **only from the training data**. These same parameters were then applied to transform the test set, a crucial practice to prevent information from the test set from influencing the training process.

This meticulous processing pipeline results in four clean, scaled, and tailored datasets, perfectly prepared for training four specialized and high-performing regression models.

## 3. Model Selection and Rationale

With the data fully processed and prepared, the next step was to select an appropriate modeling algorithm. For this project, a custom-built **Linear Regression model using Gradient Descent (`LinearRegressionGD`)** was chosen as the primary algorithm.

### 3.1. Why Start with Linear Regression?

While more complex algorithms like Gradient Boosting or Random Forests might yield a lower final error, starting with Linear Regression was a deliberate strategic choice for several key reasons:

1.  **Interpretability and Debugging:** Linear Regression is a "white-box" model. The final weights assigned to each feature provide a clear and direct interpretation of its influence on the prediction. This makes it incredibly easy to debug and verify that the model is learning logical patterns.

2.  **Excellent Baseline:** It provides a strong, quantifiable performance baseline. Any future, more complex model must significantly outperform this simple model to justify its added complexity and computational cost. Our Linear Regression model achieved an average error (RMSE) of approximately 8.2%, which is a very respectable starting point.

3.  **Manual Implementation for Deeper Understanding:** A primary goal of this project was to demonstrate a foundational understanding of machine learning principles. By manually implementing the Gradient Descent algorithm from scratch, I was able to work directly with the core mechanics of model training, including cost functions, partial derivatives, and parameter updates. This hands-on approach is invaluable for building intuition.

4.  **Sufficient for the Problem:** The EDA revealed strong linear relationships between key features (like `site area`) and the target variable. This indicated that a linear model, despite its simplicity, was likely to capture a significant portion of the variance in the data and provide valuable predictions.

### 3.2. The Multi-Model Approach

As established during the EDA, a single Linear Regression model would not suffice. The final implementation consists of **four independently trained `LinearRegressionGD` models**, one for each `structure type`. This hybrid approach combines the simplicity of a linear model with the flexibility of specialization, allowing it to capture the different feature dynamics of residential, commercial, industrial, and mixed-use buildings.

### 3.3. Hyperparameter Tuning

To optimize the performance of each of the four models, a simple grid search was performed over the two main hyperparameters of the `LinearRegressionGD` class:

*   **`learning_rate`:** Controls the step size during Gradient Descent. Tested values included `[0.3, 0.1, 0.03, 0.01]`.
*   **`n_iterations`:** The number of training iterations. Tested values included `[5000, 10000, 20000]`.

For each of the four models, the combination of hyperparameters that resulted in the lowest Mean Squared Error (MSE) on the test set was selected and its corresponding artifacts (the model object and scalers) were saved for deployment.

## 4. Model Performance and Evaluation

After training and hyperparameter tuning, each of the four specialized models was evaluated on its respective unseen test set. The primary metrics used for evaluation were **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**.

### 4.1. Understanding the Metrics

*   **Mean Squared Error (MSE):** This metric calculates the average of the squared differences between the predicted and actual values. While it is excellent for optimizing the model during training (as it heavily penalizes large errors), its units are "squared dollars," making it difficult to interpret directly.
*   **Root Mean Squared Error (RMSE):** This is the square root of the MSE. Its primary advantage is that its units are the same as the target variable (in this case, dollars). The RMSE can be interpreted as the "typical" or "average" error of the model's predictions in dollars.

### 4.2. Overall Model Performance

To get a single, representative measure of the system's performance, the MSE was calculated across the predictions for all four test sets combined.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Overall MSE** | **~42,000** | The average squared error across all predictions. |
| **Overall RMSE** | **~$205** | On average, the model's predictions are off by approximately $205. |

Given that the average electricity cost in the dataset is around **$2,500**, an average prediction error of $205 represents a **relative error of approximately 8.2%**. For a baseline implementation using a custom-built Linear Regression model, this is a highly successful and valuable result.

### 4.3. Performance Breakdown by Structure Type

While the overall performance is strong, it's also insightful to see how each specialized model performed in its domain. The following table shows the final, best MSE achieved for each model after hyperparameter tuning.

| Structure Type | Best Learning Rate | Best Iterations | Final Test MSE | Final Test RMSE |
| :--- | :--- | :--- | :---: | :---: |
| **Residential** | 0.3 | 10,000 | ~42,068 | ~$205 |
| **Commercial** | 0.03 | 5,000 | ~47,283 | ~$217 |
| **Industrial** | 0.3 | 10,000 | ~38,183 | ~$195 |
| **Mixed-use** | 0.03 | 5,000 | ~44,802 | ~$211 |

All four models performed consistently, with no single model being a significant outlier. This validates the multi-model approach as a stable and effective strategy.

### 4.4. Conclusion on Performance

The evaluation results confirm that the chosen multi-model strategy, combined with careful feature processing and hyperparameter tuning, has produced a reliable and valuable predictive tool. The system's performance provides a strong baseline that could be further improved with more advanced modeling techniques in future iterations.

## 5. API Deployment and Endpoints

To make the predictive models accessible and usable, a web service was created using the **FastAPI** framework and deployed to a live environment on **Render**. This section details the API's structure, how to interact with it, and what to expect in return.

The live API documentation, automatically generated by FastAPI, is available at:
**[https://electricity-cost-regression-predictor.onrender.com/docs](https://electricity-cost-regression-predictor.onrender.com/docs)**

### 5.1. Endpoint: `POST /predict`

This is the primary and only endpoint of the API. It is designed to accept building feature data and return a prediction for its electricity cost.

*   **Method:** `POST`
*   **URL:** `/predict`
*   **Description:** Takes a JSON object containing the features of a building and returns a single predicted electricity cost.

### 5.2. Request Body Schema

The API expects a JSON object in the request body. The fields required depend on the `structure_type`, as detailed during the feature engineering phase.

**Unified Model Schema:**
The API uses a single, unified Pydantic model for simplicity. This means `resident_count` is technically required for all requests, but it is **ignored by the model** for `Commercial` and `Industrial` types.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :---: |
| `structure_type` | string | **Required.** The type of the building. Must be one of: `"Residential"`, `"Commercial"`, `"Industrial"`, `"Mixed-use"`. | `"Residential"` |
| `site_area` | float | **Required.** The total area of the site in square meters. | `2500.5` |
| `water_consumption` | float | **Required.** The total water consumption in cubic meters. | `5150.7` |
| `utilisation_rate` | float | **Required.** The rate of building utilization, from 0 to 100. | `85.5` |
| `resident_count` | integer | **Required.** The number of residents. This value is ignored for Commercial and Industrial types but must be included in the request (e.g., with a value of `0`). | `12` |

**Example `cURL` Request:**
```bash
curl -X 'POST' \
  'https://your-app-name.onrender.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "structure_type": "Residential",
  "site_area": 3100,
  "water_consumption": 6200,
  "utilisation_rate": 90,
  "resident_count": 15
}'
```

### 5.3. Response Schema

The API provides clear and simple responses in JSON format.

#### Successful Response (`200 OK`)
If the request is valid and the prediction is successful, the API will return a JSON object with a single key.

*   **Content:**
    ```json
    {
      "predicted_cost": 2754.88
    }
    ```
*   `predicted_cost`: The model's predicted electricity cost in dollars, rounded to two decimal places.

#### Error Responses

The API uses standard HTTP status codes to indicate errors.

*   **`422 Unprocessable Entity`**
    *   **Cause:** The request body is malformed. This could be due to a missing required field (e.g., `site_area` is not provided) or a field having the wrong data type (e.g., `site_area` is sent as a string instead of a number).
    *   **Example Response:**
        ```json
        {
          "detail": [
            {
              "loc": ["body", "site_area"],
              "msg": "field required",
              "type": "value_error.missing"
            }
          ]
        }
        ```

*   **`400 Bad Request`**
    *   **Cause:** The `structure_type` provided is not one of the four valid options.
    *   **Example Response:**
        ```json
        {
          "detail": "Invalid Structure Type. Available types: ['residential', 'commercial', 'industrial', 'mixed-use']"
        }
        ```
*   **`500 Internal Server Error`**
    *   **Cause:** An unexpected error occurred on the server during the prediction process (e.g., a shape mismatch between input data and a model's expectations). The robust error handling in the endpoint is designed to catch these issues.
    *   **Example Response:**
        ```json
        {
          "detail": "An internal error occurred in the prediction logic: Shape Mismatch: The model for 'Residential' expects 4 features, but 3 were provided."
        }
        ```