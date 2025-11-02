# What's In My Bag: Behavioral Prediction Portfolio

An interactive data science investigation into predicting smoking behavior from multi-source behavioral data, presented as a forensic examination of daily life captured in data.

---

## Project Overview

This project explores the question: Can we predict tomorrow's cigarette purchase from today's behavioral signals?

Through an intimate, story-driven interface styled as an investigation, users explore financial patterns, emotional signals, cultural indicators, and physiological data to understand how machine learning can model human behavior—and the ethical implications of doing so.

The project uses a synthetic persona ("Jane Doe") to demonstrate technical capabilities while engaging deeply with questions of privacy, consent, and algorithmic surveillance.

---

## Technical Skills Demonstrated

### Data Engineering
- Multi-source data integration with temporal alignment
- Feature engineering including lag-1 transformations and rolling averages
- Data leakage detection and prevention
- Missing data imputation strategies
- Class imbalance handling with balanced weighting

### Machine Learning
- Binary classification for behavioral prediction
- Temporal modeling (today's features predict tomorrow's outcome)
- Model comparison (Logistic Regression vs. Decision Tree)
- Feature importance analysis and coefficient interpretation
- Comprehensive evaluation (ROC-AUC, precision, recall, F1, confusion matrices)

### Visualization & Communication
- Interactive web application development using Streamlit
- Dynamic visualizations with Plotly, Matplotlib, and Seaborn
- Data-driven narrative design
- Complex information architecture made accessible

### Ethical AI
- Privacy considerations in behavioral prediction systems
- Informed consent and algorithmic transparency
- Harm identification and mitigation strategies
- Guidelines for responsible AI deployment

---

## Models & Methodology

### Model 1: Logistic Regression
- **Purpose:** Interpretable baseline with clear feature coefficients
- **Configuration:** `max_iter=500`, `class_weight='balanced'`, `random_state=42`
- **Strength:** Explainability—coefficients reveal which features increase or decrease smoking probability

### Model 2: Decision Tree Classifier
- **Purpose:** Capture non-linear patterns and feature interactions
- **Configuration:** `max_depth=5`, `class_weight='balanced'`, `random_state=42`
- **Strength:** Models complex relationships between multiple factors

### Key Methodological Choices

**Lag-1 Transformation:** Uses day i features to predict day i+1 outcome, preventing temporal leakage

**Feature Exclusion:** All spending variables removed to prevent data leakage (cigarette purchases appear in spending data)

**Temporal Split:** 80/20 chronological train/test split that respects time's directionality

**Class Balancing:** Addresses class imbalance where smoking days are minority class

---

## Dataset Features

**Integrated Data Sources:**
- Credit card transactions (amount, category, frequency)
- Store receipts (convenience stores, nightlife, medical purchases)
- Twitter activity (sentiment scores, tweet volume, emotional valence)
- Spotify listening history (musical valence, track mood distribution)
- Wearable health data (sleep duration, step count, heart rate)
- Behavioral flags (alcohol purchases, nicotine patch usage)

**Target Variable:** `cigarette_purchase_day` (binary: 0 = no purchase, 1 = purchase)

**Temporal Scope:** Approximately 365 days of daily behavioral observations

---

## Ethical Framework

This project engages seriously with the ethical implications of behavioral prediction systems.

### Privacy Considerations
The model requires comprehensive surveillance across financial, emotional, cultural, and physical domains. This raises fundamental questions about the boundaries of acceptable data collection and the nature of privacy in algorithmic systems.

### Consent Challenges
Informed consent becomes problematic when individuals cannot fully understand how their data will be used for algorithmic inference. The gap between consenting to data collection and understanding predictive modeling implications is substantial.

### Potential Harms Identified
- Insurance discrimination based on behavioral risk scores
- Employment decisions influenced by addiction predictions
- Targeted manipulation exploiting moments of vulnerability
- Social stigma from leaked or misused predictions
- Psychological burden of constant behavioral monitoring

### Responsible Deployment Requirements

Any real-world deployment would require:
- User retention of full data control and ownership
- Explanatory predictions that reveal reasoning, not just outcomes
- No third-party access without explicit, informed consent
- Regular bias audits and model performance monitoring
- Right to deletion and model opt-out
- Transparent communication of limitations and uncertainties
- Human oversight with ability to override predictions

---

## Note on Synthetic Data

Jane Doe is a fictional persona. All data in this project is synthetically generated for educational and demonstration purposes. No real person's privacy was violated in creating this work.

This approach allows exploration of real methodological and ethical questions without causing actual harm, while demonstrating technical competence in behavioral modeling.

---

## Installation & Local Deployment

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/cigarette-prediction-portfolio.git
cd cigarette-prediction-portfolio

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`

---

## Project Structure

```
cigarette-prediction-portfolio/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── bagg.png                      # Visual asset
├── features_master_dataset.csv   # Synthetic behavioral dataset
└── README.md                     # Project documentation
```

---

## Technologies Used

**Core Technologies:**
- Python 3.9+
- Streamlit (web application framework)

**Data Science Stack:**
- Pandas, NumPy (data manipulation)
- Scikit-learn (machine learning)
- Plotly, Matplotlib, Seaborn (visualization)

**Deployment:**
- Git & GitHub (version control)
- Streamlit Community Cloud (hosting)

---

## Design Philosophy

This portfolio rejects conventional dashboard approaches in favor of narrative-driven data presentation. The goal is to demonstrate not only technical proficiency but also critical thinking about when and how to deploy sophisticated systems.

The project showcases:
- Technical competence in end-to-end ML pipeline development
- Ability to communicate complex ideas to diverse audiences
- Ethical reasoning about algorithmic systems
- Creative approaches to data storytelling

---

## Skills Demonstrated

### Data Analysis
- Exploratory data analysis across multiple data sources
- Statistical analysis and hypothesis testing
- Identifying patterns and correlations in behavioral data
- Data cleaning and quality assessment
- Missing data handling and imputation strategies

### Data Visualization & Communication
- Creating clear, informative visualizations using Plotly, Matplotlib, and Seaborn
- Building interactive dashboards and reports
- Translating complex data insights into understandable narratives
- Designing user-friendly data exploration interfaces
- Presenting data-driven recommendations

### Technical Proficiency
- Python programming for data analysis (Pandas, NumPy)
- Statistical modeling and machine learning fundamentals
- Interactive application development with Streamlit
- Version control with Git and GitHub
- Data pipeline development and automation

### Business & Domain Understanding
- Understanding behavioral patterns and their implications
- Identifying actionable insights from data
- Considering ethical implications of data-driven decisions
- Communicating risks and limitations of analytical approaches
- Connecting technical analysis to real-world contexts

---

## Future Enhancements

Potential extensions not yet implemented:
- Time series forecasting models (ARIMA, LSTM)
- Ensemble methods (Random Forest, XGBoost)
- SHAP values for advanced explainability
- Interactive scenario exploration tools
- Multi-subject analysis expanding beyond single persona

---

## License

This project is available under the MIT License. You may use it as inspiration for your own work, but please develop your own original implementations for professional applications.

---

## Acknowledgments

This project was developed as a demonstration of end-to-end data analysis capabilities with particular emphasis on ethical considerations in behavioral modeling. The ethical framework draws from literature on algorithmic justice, surveillance studies, and responsible AI development.

---

**Developed with commitment to responsible and transparent data science practices.**
