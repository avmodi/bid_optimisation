# bid_optimisation
# README

## Task 1: Estimate Expected Win Probability for a Given Bid Price

### Code:
- **Verve_1.ipynb**

### Observations:
- Some bid prices are sparse and lack sufficient data for reliable estimation.
- The larger the sample size, the more confident we can be about the win rate estimate.
- The expected win rate is approximately directly proportional to bid prices and can be modeled as a function of the bid price.
- More precisely:

  \[ F(p) = \alpha f(p) + \beta + \epsilon \]

  Where:
  - \( F(p) \) = estimated win rate
  - \( f(p) \) = function of \( p \) (to be learned)
  - \( \alpha, \beta \) are constants (to be learned)
  - \( \epsilon \) is noise in the data (irreducible noise due to external factors)

### Solution

#### Assumptions:
- **Stationarity**: We assume that the auction dynamics (number of bidders, their bidding strategies, etc.) remain consistent over time, allowing historical data to be used for predictions.
- **Independence**: Each auction is assumed to be independent of others.
- **Consistent Bidding**: If the same bid price is used again in a similar situation, the probability of winning remains consistent with historical observations.
- **Limited Data**: The dataset is sparse, particularly at higher bid prices, affecting accuracy. Bayesian smoothing (e.g., Laplacian) can help avoid overfitting.

### Methodology:
1. **Baseline Model**: Linear Interpolation/Extrapolation (Simple, Less Reliable)
2. **Linear Regression**
3. **Logistic Regression** (Estimates log odds)
4. **Decision Tree Model**
5. **Isotonic Regression**
6. **Polynomial Regression (Degree = 3)**

### Implementation Results
- **Overfitting**: Isotonic Regression, Decision Tree, Polynomial Regression, and Bayesian Model tend to overfit, as they focus heavily on training data.
- **Underfitting**: Linear Regression has high bias and underfits.
- **Generalization**: Logistic Regression can generalize well with more data points.

### Recommendation:
Given the limited data, we should consider one of the following two strategies:
1. **Use Prior Information**: Bayesian methods can be applied if we have reliable priors, requiring domain expertise.
2. **Collect More Data**: If more data is obtained, Logistic Regression is a reasonable next step. It effectively captures the non-linear relationship between bid price and win rate better than linear interpolation or Bayesian piecewise constant methods. More advanced ML models can be considered as data availability increases.

---

## Task 2: Maximize Expected Net Revenue

### Code:
- **Verve_2.ipynb**

### Formula:
- **Gross Revenue**: \( 0.5 \)
- **Net Revenue**:
  \[ \text{net\_revenue} = \begin{cases}
    0.5 - \text{bid\_value}, & \text{if win} \\
    0, & \text{otherwise}
  \end{cases} \]

### Expected Win Rate:
- Assumed to be equivalent to empirical win rate (smoothed version to avoid overfitting):

  ```python
  expected_win_rate = {bid_price: smoothed_win_rate for bid_price, smoothed_win_rate in df[['bid_price', 'smoothed_win_rate']].values}
  ```

- **Net Revenue Calculation**:

  ```python
  net_revenue = {bid_price: (gross_revenue - bid_price) * expected_win_rate[bid_price] for bid_price in df['bid_price'].values}
  ```

  Output:
  ```
  {0.01: 4.899951000489996e-06,
   0.1: 0.12002799720027997,
   0.2: 0.0600000239999976,
   0.4: 0.030000069999929994,
   0.5: 0.0,
   0.75: -0.07501749825017498,
   1.0: -0.3001998001998002,
   2.0: -1.0544554455445545,
   5.0: -3.681818181818182,
   9.0: -8.5}
  ```

- **Optimal Bid Price**:
  - **0.1$** is optimal, yielding a net revenue of **0.12$** per bid.

### Insights:
- **Revenue Trends**:
  - Revenue increases with win rate.
  - Revenue decreases as bid value increases.

### Formal Definition:
- **Net Revenue Formula**:
  \[ R(p) = W(p) \times (0.5 - p) \]

  Where:
  - \( R(p) \) = Net revenue for bid price \( p \)
  - \( p \) = Bid value
  - \( W(p) \) = Expected win rate for bid price \( p \) (can be estimated using ML models or statistical methods)
  - Gross revenue = \( 0.5 \) (given)

### Conclusion:
- The empirical estimate of **0.12$** assumes \( W(p) \) is estimated using Laplacian smoothing.
- The number could change if \( W(p) \) is estimated using alternative methods, such as Logistic Regression or Decision Trees.

