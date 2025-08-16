# California House Price Prediction – Linear Regression

A from-scratch implementation of Linear Regression to predict California housing prices, designed to demonstrate mastery over core ML algorithms, model evaluation, and deployment-ready code structuring without using ML libraries like `scikit-learn`.

---

## 🚀 Project Objective

This project aims to build a linear regression model **from scratch**, without relying on pre-built ML libraries, to predict median house values in California based on features such as location, population, and income levels. This serves as a foundational showcase for understanding the full ML pipeline — from data processing to model training, evaluation, and deployment-readiness.

---

## 📊 Features

- ✅ Cleaned and visualized real-world California housing dataset
- ✅ Engineered custom numerical pipeline
- ✅ Implemented **Linear Regression** from scratch using NumPy
- ✅ Trained and evaluated without any ML libraries (no scikit-learn)
- ✅ Custom implementation of **MSE**, **MAE**, **R² Score**
- ✅ Modularized codebase (ready for CI/CD integration)
- ✅ Compatible with both scripting and interactive environments

---

## 📊 Dataset Explanation

The dataset used is the **California Housing Dataset**, originally derived from the **1990 U.S. Census**. It is widely used as a benchmark dataset for regression tasks in machine learning.

- **Number of Instances:** ~20,640  
- **Number of Features:** 8 numerical + 1 target variable  
- **Target Variable:** `median_house_value` (median house price for a district in California)

### Feature Details
1. **longitude** – Longitude coordinate of the district.  
2. **latitude** – Latitude coordinate of the district.  
3. **housing_median_age** – Median age of houses in the district.  
4. **total_rooms** – Total number of rooms in all houses of the district.  
5. **total_bedrooms** – Total number of bedrooms in all houses of the district.  
6. **population** – Total population of the district.  
7. **households** – Number of households in the district.  
8. **median_income** – Median income of households in the district (scaled).  
9. **median_house_value (Target)** – Median house price in the district (USD).  

This dataset is useful for demonstrating **real-world regression tasks** since it contains:
- **Continuous target variable** (house price prediction).  
- **High-dimensional feature space** (geographic + socioeconomic).  
- **Noise and outliers**, mimicking real-world challenges.  

---

Custom Metrics

    Mean Squared Error (MSE)

    Mean Absolute Error (MAE)

    R² Score (Coefficient of Determination)


