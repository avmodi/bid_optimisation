# Bid Price Optimization

## Task 1: Estimate Expected Win Probability for a Given Bid Price

### **Observations**
- Few bid prices are sparse and do not have enough data for reliable estimation.
- The larger the sample size, the more confident we can be about the win rate estimate.
- Expected win rate is approximately proportional to bid prices, and we model win rate as a function of bid price.

### **Model Formulation**
We define the win probability function as:

F(p) = \alpha f(p) + \beta + \epsilon

Where:
-  F(p) = Estimated win rate
-  f(p) = Function of bid price \( p \) (to be learned)
-  (alpha, beta) = Constants (to be learned)
-  (epsilon) = Noise (irreducible noise due to external factors)

---

### **Solution Approach**

#### **Assumptions**
1. **Stationarity**: The auction dynamics (number of bidders, bidding strategies) remain constant over time.
2. **Independence**: Each auction is independent.
3. **Consistent Bidding**: If a bid is repeated in a similar scenario, the probability of winning remains the same.
4. **Limited Data**: Sparse data for higher bid prices affects estimation accuracy. Bayesian smoothing (e.g., Laplacian) is used to reduce overfitting.

#### **Methodology**
1. **Baseline Model**: Linear Interpolation/Extrapolation (Simple, Less Reliable)
2. **Linear Regression**
3. **Logistic Regression** (Estimates log-odds)
4. **Decision Tree Model**
5. **Isotonic Regression**
6. **Polynomial Regression (Degree = 3)**

---

### **Implementation & Results**
- **Isotonic Regression, Decision Tree, Polynomial Regression, and Bayesian Models** tend to overfit.
- **Linear Regression** has high bias and underfits the data.
- **Logistic Regression** generalizes well if trained on more data.

### **Recommendation**
Given the data limitations, we should consider:
1. **Using Prior Information**: Bayesian methods can be effective if we have good priors, requiring domain expertise.
2. **Getting More Data**: Logistic regression is a reasonable next step as it captures the non-linear relationship between bid price and win rate better than basic interpolation or Bayesian piecewise constant models. With more data, advanced machine learning models can be explored.

---

## Task 2: Maximize Expected Net Revenue

### **Revenue Calculation**
- **Gross Revenue**: \( 0.5 \)
- **Net Revenue Calculation**:

\[
Net\ Revenue = (Gross\ Revenue - Bid\ Price) \times Expected\ Win\ Rate
\]

Where:
-  R(p) = Net revenue for bid price ( p )
-  p = Bid value
-  W(p)  = Expected win rate for given bid price p (learned via ML/statistical models)
- Gross revenue is given as **0.5**.

### **Implementation**
```python
expected_win_rate = {bid_price: smoothed_win_rate for bid_price, smoothed_win_rate in df[['bid_price','smoothed_win_rate']].values}

net_revenue = {bid_price: (0.5 - bid_price) * expected_win_rate[bid_price] for bid_price in df['bid_price'].values}

### **Optimal Bid Price**
- The optimal bid price is **$0.1**, yielding a maximum net revenue of **$0.12** per bid.
- This assumes the win rate is estimated empirically using **Laplacian smoothing**.
- The optimal price may vary if we use different models (e.g., Logistic Regression, Decision Tree).

---

### **Key Takeaways**
- **Revenue increases** with **win rate** but **decreases** with **bid price**.
- The **empirical optimal bid price is $0.1** with a **net revenue of $0.12** per bid.
- If win rate \( W(p) \) is estimated using a different model (e.g., Decision Trees, Logistic Regression), the optimal bid price may change.

---

## **Conclusion**
- **Task 1**: Predicting win rate from bid price can be improved using logistic regression, Bayesian smoothing, or more data.
- **Task 2**: Net revenue is maximized at a bid price of **$0.1**, assuming empirical win rate estimation.

---

## **Files**
- `Verve_1.ipynb`: Implementation of  win probability estimation.
- `Verve_2.ipynb`: Implementation of revenue maximization based on bid price.

---

### **Colab Links**
- [Verve_1.ipynb](https://colab.research.google.com/drive/1u9MveqQPApaBzkanbevwX_I1_9Fin3q5?usp=sharing)
- [Verve_2.ipynb](https://colab.research.google.com/drive/1rpopHH335eGxrY-rS2xuKAxujR4IgMwb?usp=sharing)

