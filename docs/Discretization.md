## 1. Discretization
 
### What Is It?
 
Discretization is the process of converting **continuous (numeric) data** into **discrete categories (bins)**. Many data mining algorithms — especially association rules and Bayesian networks — require categorical input. Discretization bridges that gap.
 
**Analogy:** A thermometer gives you 36.8°C (continuous). A doctor bins that into "Normal", "Fever", "High Fever" (discrete). That binning is discretization.
 
---
 
### Why It Matters
 
| Situation | Problem Without Discretization | Solution |
|-----------|-------------------------------|----------|
| Running Apriori on ages | Age 23 ≠ Age 24 → no frequent itemsets | Bin into Young/Adult/Senior |
| Building Bayesian network | CPT would be infinite rows | Discretize to 3–5 categories |
| Decision tree on income | Every value is unique → no split works | Bin into Low/Medium/High |
 
---
 
### Methods
 
#### Method 1: Equal-Width Binning
 
Divide the range `[min, max]` into `k` intervals of equal size.
 
```
Width = (max - min) / k
Bin boundaries: min, min+W, min+2W, ..., max
```
 
**Example:** Ages = `[5, 12, 18, 25, 34, 48, 60, 72]`, k = 3
 
```
min = 5, max = 72
Width = (72 - 5) / 3 = 22.3
 
Bin 1: [5  – 27.3]  → Young
Bin 2: [27.3 – 49.6] → Adult
Bin 3: [49.6 – 72]  → Senior
 
Result: Young, Young, Young, Adult, Adult, Adult, Senior, Senior
```
 
**Weakness:** Uneven distribution — one bin may have many more points than another.
 
---
 
#### Method 2: Equal-Frequency Binning (Quantile)
 
Each bin contains approximately the same number of data points.
 
**Example:** Ages = `[5, 12, 18, 25, 34, 48, 60, 72]`, k = 4
 
```
8 values / 4 bins = 2 values per bin
 
Bin 1: [5,  12]   → Q1
Bin 2: [18, 25]   → Q2
Bin 3: [34, 48]   → Q3
Bin 4: [60, 72]   → Q4
```
 
**Advantage:** Each bin is equally represented. Better for skewed data.
 
---
 
#### Method 3: Entropy-Based (Information-Theoretic)
 
Used inside decision tree algorithms. Splits are placed at the boundary that **maximises information gain** — the split that best separates the classes.
 
```
For a split point t on attribute A:
  Gain(t) = Entropy(S) - [|S_left|/|S| * Entropy(S_left) + |S_right|/|S| * Entropy(S_right)]
 
Choose the t with highest Gain(t)
```
 
This is the "smartest" method because it uses the target class label to guide where boundaries go.
 
---
 
### Diagram: Equal-Width vs Equal-Frequency
 
```
Data: ──●──●────────────●────●●──────●─────────────────●
         5  12          34  4048      72
 
Equal-Width (k=3, W≈22):
  ├─────── Young ────────┤─── Adult ───┤───── Senior ─────┤
  5                     27.3          49.6                72
 
Equal-Frequency (k=4, 2 pts each):
  ├──Q1──┤──────Q2──────┤──────Q3─────┤──────Q4──────────┤
  5     12             25            48                   72
  [5,12]  [18,25]        [34,48]        [60,72]
```
 
---
 
### Key Exam Points
 
- State which method you are using and why.
- For equal-width: always show the width calculation.
- For exam tables: show the bin label assigned to each data point.
- Entropy-based requires knowing class labels — cannot be used unsupervised.
---
