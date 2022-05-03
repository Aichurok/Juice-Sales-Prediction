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
To find the best model for sales prediction, we compare four methods of estimating a multiple linear
regression model (MLR) as shown in Equation (1), where we use the natural logarithm of sales as the
dependent variable. In order to measure out-of-sample prediction accuracy we have to split the data into
a training sample and a test sample. We use the first 78 weeks (75%) as the training sample and the last
26 as the test sample. Because of the inclusion of lagged variables we have to remove the first observation
from the training sample. In Equation (1), Sit denotes the sales of SKU i in week t, where i = 1, ..., 8
and t = 2, ..., 78. Furthermore, xjt for j = 1, ...,K denote the K variables we use in the model in week
t. The number of variables K that are included in the model depends on the method used, but it will
always be a subset of the base set of variables we present in Section 2. Lastly, εit denotes the error term
in the model for SKU i in week t. Besides the sales, lagged sales and all price variables also enter the
model transformed by the natural logarithm. All computations and estimations for the purpose of this
paper are done in R software.
\begin{equation} \label{MLR}
    log(S_{it}) = \beta_0 + \sum_{j=1}^K x_{jt}\beta_j + \varepsilon_{it}
\end{equation}
Our benchmark model is the MLR model including the full base set of predictors, which we estimate
using ordinary least squares (OLS) estimation.
