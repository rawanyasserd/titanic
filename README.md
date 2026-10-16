# Titanic: Machine Learning from Disaster

A machine learning project that predicts which passengers survived the Titanic disaster, using the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic).

## Project Steps

1. **Loading the data**: read `train.csv` and `test.csv`
2. **Exploratory data analysis**: missing values, duplicates, and survival compared across Pclass, Sex, SibSp, Parch and Embarked
3. **Feature engineering**
   - Extract a `Title` (Mr, Miss, Mrs, other) from `Name`
   - Encode `Sex` and `Embarked` as numbers
   - Fill missing `Age` (mean) and `Embarked` (most common port), then group `Age` and `Fare` into bins
   - Create `FamilySize` from `SibSp` and `Parch`
   - Drop `Cabin` (too many missing values), `Ticket`, `SibSp`, `Parch`, `Name` and `PassengerId`
4. **Modeling**: 5-fold cross-validation on 10 models
5. **Evaluation**: 70/30 train-test split with accuracy, confusion matrix and classification report

## Models Used

KNN, Decision Tree, Random Forest, Naive Bayes, Logistic Regression, SVM, Gradient Boosting, AdaBoost, XGBoost and a Voting classifier (hard voting).

## Results

Cross-validation (5-fold) accuracy, top models:

| Model | Accuracy (%) |
|---|---|
| Random Forest | 83.39 |
| Decision Tree | 83.17 |
| Voting | 83.17 |
| XGBoost | 83.05 |
| Gradient Boosting | 82.72 |

On the 70/30 split, the Decision Tree gave the best test accuracy (83.21%).

`test.csv` has no `Survived` column, so all evaluation is done on a split of `train.csv`.

## Repository Structure

```
├── TitanicP.ipynb   # main notebook
├── train.csv        # training data (891 passengers)
├── test.csv         # test data (418 passengers, no labels)
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install the libraries:
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn xgboost jupyter
   ```
3. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook
   ```
4. Run all cells. The notebook reads `train.csv` and `test.csv` from its own folder, so keep the three files together.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Jupyter Notebook

## Author

Rawan

## Data Source

Dataset from the Kaggle [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.
