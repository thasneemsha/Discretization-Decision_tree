## 2. Decision Trees
 
### What Is It?
 
A decision tree is a **flowchart-like model** that classifies data by asking a sequence of questions about attributes. Each internal node is a test; each branch is an outcome; each leaf is a class label.
 
**Analogy:** A doctor diagnosing flu: "Do you have a fever? → Yes. Do you have a cough? → Yes. → Likely flu."
 
---
 
### Building a Decision Tree: ID3 Algorithm
 
The ID3 algorithm builds a tree top-down by always picking the **attribute with the highest Information Gain** to split on.
 
#### Step 1: Calculate Entropy of the Dataset
 
Entropy measures **impurity** (how mixed the classes are).
 
```
Entropy(S) = -∑ p_i * log₂(p_i)
 
Where p_i = fraction of examples belonging to class i
```
 
- Entropy = 0 → perfectly pure (all one class)
- Entropy = 1 → perfectly impure (50/50 split for 2 classes)
**Example:** 9 positive, 5 negative examples
 
```
p+ = 9/14,  p- = 5/14
 
Entropy(S) = -(9/14)log₂(9/14) - (5/14)log₂(5/14)
           = -(0.643)(−0.637) - (0.357)(−1.485)
           = 0.410 + 0.530
           = 0.940
```
 
---
 
#### Step 2: Calculate Information Gain for Each Attribute
 
```
Gain(S, A) = Entropy(S) - ∑_v [ |S_v| / |S| * Entropy(S_v) ]
 
Where S_v = subset of S where attribute A has value v
```
 
Pick the attribute with the **highest Gain** as the root split.
 
---
 
#### Full Worked Example
 
**Dataset — PlayTennis:**
 
| Day | Outlook  | Temp | Humidity | Wind   | Play? |
|-----|----------|------|----------|--------|-------|
| D1  | Sunny    | Hot  | High     | Weak   | No    |
| D2  | Sunny    | Hot  | High     | Strong | No    |
| D3  | Overcast | Hot  | High     | Weak   | Yes   |
| D4  | Rain     | Mild | High     | Weak   | Yes   |
| D5  | Rain     | Cool | Normal   | Weak   | Yes   |
| D6  | Rain     | Cool | Normal   | Strong | No    |
| D7  | Overcast | Cool | Normal   | Strong | Yes   |
| D8  | Sunny    | Mild | High     | Weak   | No    |
| D9  | Sunny    | Cool | Normal   | Weak   | Yes   |
| D10 | Rain     | Mild | Normal   | Weak   | Yes   |
| D11 | Sunny    | Mild | Normal   | Strong | Yes   |
| D12 | Overcast | Mild | High     | Strong | Yes   |
| D13 | Overcast | Hot  | Normal   | Weak   | Yes   |
| D14 | Rain     | Mild | High     | Strong | No    |
 
Total: 9 Yes, 5 No
 
**Step 1: Overall Entropy**
 
```
Entropy(S) = -(9/14)log₂(9/14) - (5/14)log₂(5/14) = 0.940
```
 
**Step 2: Gain for Outlook**
 
| Outlook  | Yes | No | Total | Entropy |
|----------|-----|----|-------|---------|
| Sunny    | 2   | 3  | 5     | 0.971   |
| Overcast | 4   | 0  | 4     | 0.000   |
| Rain     | 3   | 2  | 5     | 0.971   |
 
```
Gain(S, Outlook) = 0.940 - [(5/14)(0.971) + (4/14)(0.000) + (5/14)(0.971)]
                = 0.940 - [0.347 + 0 + 0.347]
                = 0.940 - 0.694
                = 0.246
```
 
**Step 3: Gain for Wind**
 
| Wind   | Yes | No | Total | Entropy |
|--------|-----|----|-------|---------|
| Weak   | 6   | 2  | 8     | 0.811   |
| Strong | 3   | 3  | 6     | 1.000   |
 
```
Gain(S, Wind) = 0.940 - [(8/14)(0.811) + (6/14)(1.000)]
              = 0.940 - [0.463 + 0.429]
              = 0.940 - 0.892
              = 0.048
```
 
Since Gain(Outlook) = 0.246 > Gain(Wind) = 0.048 → **Outlook is the root split**.
 
**Resulting Tree Structure:**
 
```
                    [Outlook?]
                   /    |     \
             Sunny  Overcast   Rain
               |       |        |
           [Humid?]   YES    [Wind?]
           /      \          /     \
         High   Normal    Strong   Weak
          |        |        |       |
          NO      YES       NO     YES
```
 
---
 
### Overfitting and Pruning
 
A fully grown tree memorises training data but fails on new data. **Pruning** removes branches that provide little predictive power.
 
- **Pre-pruning:** Stop splitting when Gain < threshold
- **Post-pruning:** Grow full tree, then remove weak subtrees using validation data
---
 
### Key Exam Points
 
- Always compute overall entropy first.
- Show the entropy calculation for each value of each attribute before computing Gain.
- The attribute with the highest Gain becomes the node.
- Repeat recursively on each branch with the remaining attributes.
---
 
