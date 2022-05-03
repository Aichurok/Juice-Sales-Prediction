# Juice-Sales-Prediction
In order to find the best prediction model for sales, we compare four different models, namely multiple
linear regression (MLR) estimated by OLS, stepwise regression, LASSO, and Ridge regression. 
The this purpose R was used. 

***DATA***

The dataset we use for the purpose of this paper is obtained from one store from Dominick’s Finer Foods
chain in the greater Chicago area. The data covers weekly observations on the sales of orange juice for a
total of 104 weeks in the period from January 1991 to December 1992. Included in the dataset are variables
that represent unit sales, shelf prices and a binary variable, promotion, that indicates the presence of
non-price promotion. Observations on these variables are available for eight Store Keeping Units (SKU)
in the refrigerated subcategory. Given potential competitive effects, weekly category sales, prices and
promotion intensity for the two subcategories of frozen and shelf storable orange juice are included as
well. Promotion intensity denotes the percentage of SKUs in that subcategory with a promotion. The
eight SKUs in the refrigerated subcategory, which are the objects of our research, are differentiated by
size and brand name. These are listed below.
1. 64 ounces Florida Gold
2. 64 ounces Minute Maid
3. 96 ounces Minute Maid
4. 64 ounces Tropicana
5. 96 ounces Tropicana
6. 64 ounces Dominick’s
7. 128 ounces Dominick’s
8. 64 ounces Florida’s Natural

***METHODOLOGY***

Our benchmark model is the MLR model including the full base set of predictors, which we estimate
using ordinary least squares (OLS) estimation.

In contrast to the first method, the second method we use results in a MLR model with only a subset
of our base set of variables as predictors. Stepwise regression (SR), using backward selection, starts with
the full model as in the first method. It then chooses to remove one variable at a time, which improves
some criteria the most until removing a variable is not desirable anymore in terms of the criteria. We
choose to perform this subset selection method according to the Akaike information criterium (AIC) as
AIC is most appropriate for exploratory analysis and model selection to address which model will best
predict the next sample , which is in line with the aim of our work.
As an alternative to the subset selection method, we use two shrinkage methods, which shrink the regression
coefficients toward zero. The shrinkage methods we use as our
third and fourth method of MLR model estimation are Least Absolute Shrinkage and Selection Operator
(LASSO) and Ridge regression. Both methods extend the least squares criterion by putting a penalty
term λ on the size of the parameters.

To find the optimal value of λR and λL, i.e. λR and λL such that the Mean Square Error (MSE) is
minimized, we use the k-fold cross-validation resampling method. Kohavi et al. (1995) found that tenfold
cross-validation is preferred over other resampling methods such as bootstrap and that the optimal
number of folds is ten, even if computation power allows to use more folds. Since our training sample
is relatively small (77 observations), we will use five-fold cross-validation as suggested by James et al.
(2013).

In order to compare prediction accuracy accross models for the same SKU, we employ the modified
Diebold-Mariano (DM) test for small to moderate sample sizes, proposed by Harvey et al. (1997). The
DM test compares prediction accuracy of two models and determines whether the two forecasts are
significantly different or that the difference can be due to chance (Diebold and Mariano, 1995). The null
hypothesis of the DM test is equality of the MSE’s in the test sample. We use a 5% significance level in
all our tests. The alternatives we use in our DM tests are either of one MSE being greater than the other
or MSE’s being significantly different.
