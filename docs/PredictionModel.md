## 7. Prediction Models
 
### What Is It?
 
A prediction model uses patterns learned from historical data to forecast the **outcome for new, unseen examples**. This is the end-goal of the entire data mining pipeline.
 
---
 
### Types of Prediction
 
| Type | Output | Example Method | Example Task |
|------|--------|---------------|--------------|
| Classification | Category label | Decision Tree, Naive Bayes | Spam or not spam? |
| Regression | Continuous number | Linear regression, Lagrange | What will the price be? |
| Probabilistic | Probability distribution | Bayesian Network | What's the chance of disease? |
| Pattern-based | Rule triggered | Association rules | What will the customer buy next? |
 
---
 
### Naive Bayes Classifier (Prediction using Bayes)
 
The Naive Bayes classifier is the simplest probabilistic prediction model. It assumes all features are **independent given the class** (the "naive" assumption).
 
```
P(class | features) ∝ P(class) × ∏ P(featureᵢ | class)
 
Predict the class with the highest value.
```
 
**Example:** Classify a new day as PlayTennis = Yes/No
 
New day: Outlook=Sunny, Temp=Cool, Humidity=High, Wind=Strong
 
**From the training data (14 days, 9 Yes, 5 No):**
 
```
P(Yes) = 9/14,   P(No) = 5/14
 
P(Sunny | Yes) = 2/9,    P(Sunny | No) = 3/5
P(Cool  | Yes) = 3/9,    P(Cool  | No) = 1/5
P(High  | Yes) = 3/9,    P(High  | No) = 4/5
P(Strong| Yes) = 3/9,    P(Strong| No) = 3/5
 
P(Yes | day) ∝ (9/14) × (2/9) × (3/9) × (3/9) × (3/9)
             = 0.643 × 0.222 × 0.333 × 0.333 × 0.333
             = 0.00529
 
P(No | day)  ∝ (5/14) × (3/5) × (1/5) × (4/5) × (3/5)
             = 0.357 × 0.600 × 0.200 × 0.800 × 0.600
             = 0.02057
```
 
**P(No) > P(Yes) → Prediction: Do NOT play tennis.**
 
---
 
### Evaluating a Prediction Model
 
#### Confusion Matrix
 
```
                 Predicted
                 Pos    Neg
Actual  Pos  |  TP  |  FN  |
        Neg  |  FP  |  TN  |
```
 
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)        ← of all predicted positive, how many were right?
Recall    = TP / (TP + FN)        ← of all actual positives, how many did we catch?
F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
```
 
#### Cross-Validation (k-fold)
 
```
1. Split data into k equal folds
2. For each fold i:
     - Train on all folds except fold i
     - Test on fold i
     - Record accuracy
3. Final accuracy = average over all k folds
```
 
This gives a reliable estimate of how the model generalises to unseen data.
 
---
 
### The Full Data Mining Pipeline
 
```
Raw Data
   │
   ▼
[1. Discretization]  ← Convert continuous to categorical
   │
   ▼
[4. Association Rules]  ← Discover patterns in the data
   │
   ▼
[6. Model Discovery]   ← Find the best model structure
   │
   ├──────────────────────────────────┐
   ▼                                  ▼
[2. Decision Tree]            [5. Bayesian Network]
        or                            or
[3. Lagrange fit]             [Naive Bayes Classifier]
   │
   ▼
[7. Prediction Model]  ← Deploy and predict on new data
   │
   ▼
Evaluate: Accuracy, Precision, Recall, AIC/BIC
```
 
---
 
### Key Exam Points
 
- State which type of prediction (classification or regression) before starting.
- For Naive Bayes: multiply all conditional probabilities then compare — do NOT normalise unless specifically asked for probabilities.
- Show the confusion matrix if evaluation is required.
- Cross-validation is the gold standard for evaluation — explain why (avoids overfitting to test set).
---
 
---
 
## 8. How They All Connect
 
```
                        ┌─────────────────────────────────┐
                        │         RAW DATASET              │
                        │  (mixed continuous + categorical) │
                        └──────────────┬──────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │       1. DISCRETIZATION          │
                        │  Bins continuous variables so    │
                        │  other algorithms can process    │
                        └──────────────┬──────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
         ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
         │ 4. ASSOC.    │   │  6. MODEL    │   │  3. LAGRANGE     │
         │  RULES       │   │  DISCOVERY   │   │  INTERPOLATION   │
         │ Find frequent│   │ Find best    │   │ Fit polynomial   │
         │ itemsets +   │   │ model type   │   │ to numeric data  │
         │ rules        │   │ & structure  │   │                  │
         └──────────────┘   └──────┬───────┘   └────────┬─────────┘
                                   │                    │
                       ┌───────────┴────────────┐       │
                       ▼                        ▼       │
              ┌──────────────┐        ┌──────────────┐  │
              │ 2. DECISION  │        │ 5. BAYESIAN  │  │
              │    TREE      │        │   NETWORK    │  │
              │ Splits using │        │ Prob. causal │  │
              │ info. gain   │        │ model + CPTs │  │
              └──────┬───────┘        └──────┬───────┘  │
                     │                       │           │
                     └───────────────────────┴───────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │      7. PREDICTION MODEL         │
                        │  Apply trained model to new data │
                        │  Evaluate: accuracy, AIC, BIC    │
                        └─────────────────────────────────┘
```
 
**The key relationships:**
 
- Discretization is a **prerequisite** for Bayesian networks and association rules.
- Association rules **explore** the data to understand what co-occurs.
- Model discovery **selects** the best structure — it uses independence tests (probability constraints) to determine Bayesian network structure.
- Decision trees, Bayesian networks, and Lagrange interpolation are all **model types** that model discovery can discover.
- All of them feed into a **prediction model** that generalises to new data.
---
 
