# Customer-Churn-Prediction
This Python project predicts telecom customer churn using models like LightGBM and PyTorch NN. It preprocesses data, balances classes via SMOTE, and tunes for ~97% accuracy. Insights from feature importances and probability segments (e.g., low risk: rewards; high: outreach) enable targeted retention strategies to cut churn.

#### 1. **Project Purpose and Goals**
The core aim is to tackle customer churn, a common issue in telecom where users switch providers, causing financial losses. By analyzing historical customer data, the project predicts who might churn and suggests targeted retention actions. Key goals include:
- Building accurate predictive models to forecast churn.
- Uncovering data patterns (e.g., high usage or frequent support calls) that signal risk.
- Generating actionable strategies to improve retention, potentially increasing revenue through better customer loyalty.

This isn't just a technical exercise—it's geared toward real-world use, like integrating into a CRM system for timely interventions.

#### 2. **Data and Preparation**
The project uses a telecom dataset split into training (2,666 records) and testing (667 records) sets, with 20 features per record. These include:
- **Customer Attributes**: Account duration, subscription plans (international, voicemail), and voicemail usage.
- **Behavior Metrics**: Call minutes, calls, and charges across day, evening, night, and international categories.
- **Interaction Data**: Number of customer service calls.
- **Outcome**: A binary churn label (0 = stays, 1 = leaves).

Data was clean (no missing values), but imbalanced (only ~14% churn cases). Preparation steps involved:
- Encoding categories (e.g., plans as 0/1).
- Dropping low-value features (e.g., state codes).
- Standardizing numbers for consistency.
- Balancing classes with SMOTE to avoid model bias toward non-churn.

Exploratory analysis (EDA) used visualizations like histograms, pairplots, and heatmaps to spot trends, such as churned customers having higher daytime usage or more support calls.

#### 3. **Modeling Approach**
Multiple machine learning models were trained and fine-tuned to predict churn:
- **Logistic Regression**: A simple baseline, optimized for penalty strength.
- **Random Forest**: An ensemble method, tuned for tree count and depth; useful for ranking feature importance.
- **XGBoost and LightGBM**: Advanced boosting algorithms, adjusted for learning rate and tree depth; they excelled in handling complex patterns.
- **Neural Network (PyTorch)**: A deep learning model with layers for non-linear relationships, trained over epochs with dropout to prevent overfitting.

Models were evaluated on metrics like accuracy (up to 97%), precision, recall, F1-score, and ROC-AUC. Comparisons via bar charts highlighted LightGBM as often the top performer due to its speed and accuracy.

#### 4. **Retention Insights and Strategies**
A major focus is turning predictions into actions:
- **Feature-Driven Suggestions**: Based on model importances (e.g., from Random Forest), high-impact factors like "Total day minutes" or "Customer service calls" led to ideas such as discounts for heavy users or faster support for complainers.
- **Probability-Based Segmentation**: Churn probabilities (from the best model) were grouped into ranges (0-0.3 very low risk, 0.31-0.5 low, 0.51-0.8 medium, 0.81-1 high), with customized advice like loyalty rewards for low-risk customers or urgent discounts for high-risk ones.
- **Prediction Demo**: The code includes a function (using LightGBM) to predict on new data, outputting churn likelihood and tailored suggestions. Examples with 10 varied samples showed a mix of outcomes, illustrating real-world variability.

Average retention probability was around 85%, with visualizations confirming most customers are low-risk.

#### 5. **Results and Key Findings**
- **Performance**: Boosting models like LightGBM achieved 95-97% accuracy, minimizing false predictions (especially missing actual churners).
- **Insights**: Churn is linked to high costs/usage without supportive plans and poor service experiences. Probability distributions skewed toward retention, but high-risk segments need immediate attention.
- **Business Impact**: Implementing these could cut churn by 20-30% through targeted efforts, as suggested by the model's strong recall.
