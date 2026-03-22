# 📉 Employee Absenteeism Prediction System
> **An End-to-End Machine Learning Pipeline for Workforce Optimization.**

This project focuses on predicting whether an employee will be "excessively absent" (defined as >3 hours) based on various personal and professional factors. By leveraging **Logistic Regression**, the system provides actionable insights for HR departments to minimize productivity loss.

---

## 🧠 Digital Logic: How the Algorithm "Thinks"

The system doesn't just process numbers; it deciphers human behavior patterns through a structured computational pipeline:

### 1. Feature Engineering & Data Distillation
The raw dataset contains noise and high-cardinality features. The "brain" performs the following operations:
*   **Categorical Grouping:** 28 different "Reasons for Absence" are mapped into **4 distinct logical clusters** (1: Serious Diseases, 2: Pregnancy, 3: Light Diseases, 4: Routine Check-ups). This reduces complexity and helps the model focus on severity.
*   **Temporal Extraction:** The algorithm extracts **Month** and **Day of the Week** from raw timestamps to identify cyclical trends (e.g., "Monday Effect").
*   **Binary Transformation:** To create a clear decision boundary, the target variable (Absenteeism hours) is transformed into a binary classification problem: `0` (Normal) and `1` (Excessive).

### 2. Standardization & Weight Balancing
To prevent features with large magnitudes (like `Distance to Work`) from dominating smaller but critical features (like `Number of Children`), the algorithm applies **Standard Scaling**. 
*   **Logic:** Every input is shifted to a mean of 0 and a standard deviation of 1, ensuring a "fair" competition during the learning phase.

### 3. Logistic Regression Engine
The model calculates the **Log-Odds** of absenteeism. It maps the weighted sum of features through a **Sigmoid Function**, producing a probability between `0` and `1`. 
*   **Threshold:** If $P(Absenteeism) > 0.5$, the computer flags the employee as a high-risk instance.

---

## 🛠️ Technical Stack & Architecture

*   **Language:** Python 3.x
*   **Core Libraries:** `Pandas` (Data Manipulation), `NumPy` (Matrix Calculus), `Scikit-Learn` (Model Implementation).
*   **Deployment Ready:** Includes a modular `absenteeism_module.py` for real-world integration.
*   **Visualization:** `Matplotlib` & `Seaborn` for coefficient analysis.

---

## 📈 Model Performance & Coefficients
The algorithm assigns a "Weight" to each feature. In this project, the most influential factors were:
1.  **Reason for Absence (Group 1):** The highest positive coefficient (as expected).
2.  **Transportation Expense:** Correlated with higher absenteeism rates.
3.  **Number of Children:** A significant predictor for family-related leaves.

---

## 📊 Business Intelligence & Insights
To transform raw predictions into actionable strategy, I utilized **Tableau** to visualize the key drivers of absenteeism. 

![Tableau Dashboard](./tableau.png)

*Key Insight:* The visualization reveals that distance from work and transportation expenses have a non-linear but significant impact on long-term absenteeism trends.

├── .idea/                  # Project configuration
├── Absenteeism_data.csv    # Raw source data
├── absenteeism_module.py   # Core Logic: The "Brain" class for preprocessing & prediction
├── main.py                 # Entry point of the application
├── model                   # Saved Logistic Regression model weights
├── scaler                  # Saved StandardScaler object for data normalization
├── tableau.png             # Visual insights & dashboard results
└── using-scaler-model.py   # Integration script demonstrating model deployment
