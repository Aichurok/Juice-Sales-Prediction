library("psych")
library("dplyr")
library("ggplot2")
library("GGally")
library("readxl")
library("tidyverse")
library("magrittr")
library("vctrs")
library("olsrr")
library("glmnetUtils")
library("glmnet")
library("lmtest")
library("normtest")
library("car")
library("caret")
library("MLmetrics")
library("forecast")
library("stats")
library("GMCM")

datasetX13 <- read_excel("datasetX13_results.xlsx")
#ADD RELEVANT LAGGED VARIABLES INTO OUR DATASET
data_with_lagged <- datasetX13 %>%                            
  dplyr::mutate(promo_1_lagged = lag(promo_1, n = 1, default = NA)) %>%
  dplyr::mutate(promo_2_lagged = lag(promo_2, n = 1, default = NA)) %>%
  dplyr::mutate(promo_3_lagged = lag(promo_3, n = 1, default = NA)) %>%
  dplyr::mutate(promo_4_lagged = lag(promo_4, n = 1, default = NA)) %>%
  dplyr::mutate(promo_5_lagged = lag(promo_5, n = 1, default = NA)) %>%
  dplyr::mutate(promo_6_lagged = lag(promo_6, n = 1, default = NA)) %>%
  dplyr::mutate(promo_7_lagged = lag(promo_7, n = 1, default = NA)) %>%
  dplyr::mutate(promo_8_lagged = lag(promo_8, n = 1, default = NA)) %>%
  dplyr::mutate(promo_s_lagged = lag(promo_s, n = 1, default = NA)) %>%
  dplyr::mutate(promo_f_lagged = lag(promo_f, n = 1, default = NA)) %>%
  dplyr::mutate(price_1_lagged = lag(price_1, n = 1, default = NA)) %>%
  dplyr::mutate(price_2_lagged = lag(price_2, n = 1, default = NA)) %>%
  dplyr::mutate(price_3_lagged = lag(price_3, n = 1, default = NA)) %>%
  dplyr::mutate(price_4_lagged = lag(price_4, n = 1, default = NA)) %>%
  dplyr::mutate(price_5_lagged = lag(price_5, n = 1, default = NA)) %>%
  dplyr::mutate(price_6_lagged = lag(price_6, n = 1, default = NA)) %>%
  dplyr::mutate(price_7_lagged = lag(price_7, n = 1, default = NA)) %>%
  dplyr::mutate(price_8_lagged = lag(price_8, n = 1, default = NA)) %>%
  dplyr::mutate(price_s_lagged = lag(price_s, n = 1, default = NA)) %>%
  dplyr::mutate(price_f_lagged = lag(price_f, n = 1, default = NA)) %>%
  dplyr::mutate(sales_1_lagged = lag(sales_1, n = 1, default = NA)) %>%
  dplyr::mutate(sales_2_lagged = lag(sales_2, n = 1, default = NA)) %>%
  dplyr::mutate(sales_3_lagged = lag(sales_3, n = 1, default = NA)) %>%
  dplyr::mutate(sales_4_lagged = lag(sales_4, n = 1, default = NA)) %>%
  dplyr::mutate(sales_5_lagged = lag(sales_5, n = 1, default = NA)) %>%
  dplyr::mutate(sales_6_lagged = lag(sales_6, n = 1, default = NA)) %>%
  dplyr::mutate(sales_7_lagged = lag(sales_7, n = 1, default = NA)) %>%
  dplyr::mutate(sales_8_lagged = lag(sales_8, n = 1, default = NA)) %>%
  dplyr::mutate(sales_s_lagged = lag(sales_s, n = 1, default = NA)) %>%
  dplyr::mutate(sales_f_lagged = lag(sales_f, n = 1, default = NA)) %>%
  as.data.frame()

#-----DATA PREP FOR RIDGE AND LASSO-----------------------------------------------------------------------------------------------
#Recreate data set with logs of all sales and price variables
data_log <- data.frame(week=data_with_lagged$week, logsales_1=log(data_with_lagged$sales_1), logsales_2=log(data_with_lagged$sales_2),
                       logsales_3=log(data_with_lagged$sales_3), logsales_4=log(data_with_lagged$sales_4), logsales_5=log(data_with_lagged$sales_5),
                       logsales_6=log(data_with_lagged$sales_6), logsales_7=log(data_with_lagged$sales_7), logsales_8=log(data_with_lagged$sales_8),
                       logsales_s=log(data_with_lagged$sales_s), logsales_f=log(data_with_lagged$sales_f),
                       logsales_1_lag=log(data_with_lagged$sales_1_lagged), logsales_2_lag=log(data_with_lagged$sales_2_lagged),
                       logsales_3_lag=log(data_with_lagged$sales_3_lagged), logsales_4_lag=log(data_with_lagged$sales_4_lagged), logsales_5_lag=log(data_with_lagged$sales_5_lagged),
                       logsales_6_lag=log(data_with_lagged$sales_6_lagged), logsales_7_lag=log(data_with_lagged$sales_7_lagged), logsales_8_lag=log(data_with_lagged$sales_8_lagged),
                       logsales_s_lag=log(data_with_lagged$sales_s_lagged), logsales_f_lag=log(data_with_lagged$sales_f_lagged),
                       logprice_1=log(data_with_lagged$price_1), logprice_2=log(data_with_lagged$price_2), logprice_3=log(data_with_lagged$price_3),
                       logprice_4=log(data_with_lagged$price_4), logprice_5=log(data_with_lagged$price_5), logprice_6=log(data_with_lagged$price_6),
                       logprice_7=log(data_with_lagged$price_7), logprice_8=log(data_with_lagged$price_8), logprice_s=log(data_with_lagged$price_s),
                       logprice_f=log(data_with_lagged$price_f),
                       logprice_1_lag=log(data_with_lagged$price_1_lagged), logprice_2_lag=log(data_with_lagged$price_2_lagged), logprice_3_lag=log(data_with_lagged$price_3_lagged),
                       logprice_4_lag=log(data_with_lagged$price_4_lagged), logprice_5_lag=log(data_with_lagged$price_5_lagged), logprice_6_lag=log(data_with_lagged$price_6_lagged),
                       logprice_7_lag=log(data_with_lagged$price_7_lagged), logprice_8_lag=log(data_with_lagged$price_8_lagged), logprice_s_lag=log(data_with_lagged$price_s_lagged),
                       logprice_f_lag=log(data_with_lagged$price_f_lagged),
                       promo_1=data_with_lagged$promo_1, promo_2=data_with_lagged$promo_2, promo_3=data_with_lagged$promo_3, promo_4=data_with_lagged$promo_4, 
                       promo_5=data_with_lagged$promo_5, promo_6=data_with_lagged$promo_6, promo_7=data_with_lagged$promo_7, promo_8=data_with_lagged$promo_8,
                       promo_s=data_with_lagged$promo_s, promo_f=data_with_lagged$promo_f,
                       promo_1_lag=data_with_lagged$promo_1_lagged, promo_2_lag=data_with_lagged$promo_2_lagged, promo_3_lag=data_with_lagged$promo_3_lagged,
                       promo_4_lag=data_with_lagged$promo_4_lagged, promo_5_lag=data_with_lagged$promo_5_lagged, promo_6_lag=data_with_lagged$promo_6_lagged,
                       promo_7_lag=data_with_lagged$promo_7_lagged, promo_8_lag=data_with_lagged$promo_8_lagged, promo_s_lag=data_with_lagged$promo_s_lagged,
                       promo_f_lag=data_with_lagged$promo_f_lagged)

#Remove first observation because of missing lagged variables
data_log <- data_log[-1,]

#Split data into training and testing data and remove week column
train_data <- subset(data_log, week <= 78)
test_data <- subset(data_log, week >= 79)
train_data <- train_data[,-1]
test_data <- test_data[,-1]

#Split data into train and test sample and remove column week
#Dataframes for OLS
X_train.df <- train_data[,-c(1:10)]
X_test.df <- test_data[,-c(1:10)]
#Matrices for lasso and ridge
X_train <- data.matrix(X_train.df)
X_test <- data.matrix(X_test.df)

all.Y_train <- train_data[,c(1:8)]
all.Y_test <- test_data[,c(1:8)]

#Create storing matrices for all forecasting measures
OLS_measures <- data.frame(SKU=c(1:8),MAE=numeric(8),MSE=numeric(8))
OLS_G2S_measures <- data.frame(SKU=c(1:8),MAE=numeric(8),MSE=numeric(8),DM_test=numeric(8), DM_p=numeric(8))
Lasso_measures <- data.frame(SKU=c(1:8),Lambda=numeric(8), MAE=numeric(8),MSE=numeric(8), DM_test=numeric(8), DM_p=numeric(8), DM_test_g2s=numeric(8), DM_p_g2s=numeric(8), DM_test_ridge=numeric(8), DM_p_ridge=numeric(8))
Ridge_measures <- data.frame(SKU=c(1:8),Lambda=numeric(8), MAE=numeric(8),MSE=numeric(8), DM_test=numeric(8), DM_p=numeric(8), DM_test_g2s=numeric(8), DM_p_g2s=numeric(8), DM_test_lasso=numeric(8), DM_p_lasso=numeric(8))

#Lambda sequence used in cross validation
lambdas <- seq(0.001, 5, by = 0.001)


#--------------------------------------------------------------------------------------------------------
#for (i in 1:8) {
i<-8
sales_true <-data_with_lagged[79:104,1+i]
Y_train <- all.Y_train[,i]
Y_test <- all.Y_test[,i]

#-----OLS------------------------------------------------------------------------------------------------

model_OLS <- lm(data=X_train.df, Y_train~ logsales_1_lag+logsales_2_lag+ logsales_3_lag+ logsales_4_lag+
                                          logsales_5_lag+ logsales_6_lag+ logsales_7_lag+logsales_8_lag+ 
                                          logsales_s_lag+ logsales_f_lag+
                                          logprice_1+logprice_2+logprice_3+logprice_4+
                                          logprice_5+logprice_6+logprice_7+logprice_8+
                                          logprice_s+ logprice_f+ 
                                          logprice_1_lag+logprice_2_lag+logprice_3_lag+logprice_4_lag+
                                          logprice_5_lag+logprice_6_lag+logprice_7_lag+logprice_8_lag+
                                          logprice_s_lag+ logprice_f_lag+
                                          promo_1+promo_2+promo_3+promo_4+promo_5+
                                          promo_6+promo_7+promo_8+promo_s+promo_f+
                                          promo_1_lag+promo_2_lag+promo_3_lag+promo_4_lag+
                                          promo_5_lag+ promo_6_lag+promo_7_lag+promo_8_lag+
                                          promo_s_lag+promo_f_lag)

se_ols <- summary(model_OLS)$sigma

# Here we forecast using the estimates obtained via OLS
logsales_pred <- predict(model_OLS, newdata = X_test.df)
#Transform logsales -> sales including the correction term
sales_predadj <- exp(logsales_pred + se_ols^2/2)


ferrors_OLS <- sales_true - sales_predadj

OLS_measures$MSE[i] <- MSE(sales_predadj, sales_true)
OLS_measures$MAE[i] <- MAE(sales_predadj, sales_true)

#-----OLS G2S---------------------------------------------------------------------------------------------
OLS_G2S_model <- step(model_OLS, direction="backward", k=log(77))
se_g2s <- summary(OLS_G2S_model)$sigma
g2s_logsales_pred <- predict(OLS_G2S_model, newdata = X_test.df)
g2s_sales_predadj <- exp(g2s_logsales_pred + se_g2s^2/2)

ferrors_g2s <- sales_true - g2s_sales_predadj

OLS_G2S_measures$MSE[i] = MSE(g2s_sales_predadj, sales_true)
OLS_G2S_measures$MAE[i] = MAE(g2s_sales_predadj, sales_true)
OLS_G2S_measures$DM_test[i] <- dm.test(ferrors_OLS, ferrors_g2s, alternative="greater")$statistic
OLS_G2S_measures$DM_p[i] <- dm.test(ferrors_OLS, ferrors_g2s, alternative="greater")$p.value

#-----Lasso--------------------------------------------------------------------------------------------------
set.seed(1) #Set seed for reproducibility 
cv_model_Lasso <- cv.glmnet(X_train, Y_train, alpha = 1, lambda = lambdas, nfolds =10, standardize = TRUE, intercept = FALSE)
#find optimal lambda value that minimizes the MSE
best_lambda_lasso <- cv_model_Lasso$lambda.min
Lasso_measures$Lambda[i] <- best_lambda_lasso
#plot(cv_model_Lasso)

best_model_Lasso <- glmnet(X_train ,Y_train, alpha = 1, lambda = best_lambda_lasso, standardize = TRUE, intercept = FALSE)

lasso_logsales_pred  <- predict(best_model_Lasso, s = best_lambda_lasso, newx = X_test)
#HERE WE DESTANDARDIZE THE FORECASTED LOG(SALES)
#lasso_logsales_pred <- lasso_logsales_pred * Y_stdev[i] + Y_mean[i] 
#HERE WE TRANSFORM THE DESTANDARDIZED LOG(SALES) INTO SALES
sigma_Lasso <- sigma(best_model_Lasso)
lasso_sales_predadj <- exp(lasso_logsales_pred+sigma_Lasso^2/2)

ferrors_Lasso <- sales_true-lasso_sales_predadj

Lasso_measures$MSE[i] <- MSE(lasso_sales_predadj, sales_true)
Lasso_measures$MAE[i] <- MAE(lasso_sales_predadj, sales_true)
Lasso_measures$DM_test[i] <- dm.test(ferrors_OLS, ferrors_Lasso, alternative="greater")$statistic
Lasso_measures$DM_p[i] <- dm.test(ferrors_OLS, ferrors_Lasso, alternative="greater")$p.value
Lasso_measures$DM_test_g2s[i] <- dm.test(ferrors_g2s, ferrors_Lasso, alternative="greater")$statistic
Lasso_measures$DM_p_g2s[i] <- dm.test(ferrors_g2s, ferrors_Lasso, alternative="greater")$p.value

#-----RIDGE------------------------------------------------------------------------------------------------- 
set.seed(1)#Set seed for reproducibility 
cv_model_Ridge <- cv.glmnet(X_train, Y_train, alpha = 0, lambda = lambdas, nfolds =10, standardize = TRUE, intercept = FALSE)
#find optimal lambda value that minimizes test MSE
best_lambda_Ridge <- cv_model_Ridge$lambda.min
Ridge_measures$Lambda[i] <- best_lambda_Ridge
#plot(cv_model_Lasso)
best_model_Ridge <- glmnet(X_train ,Y_train, alpha = 0, lambda = best_lambda_Ridge, standardize = TRUE, intercept = FALSE)

ridge_logsales_pred  <- predict(best_model_Ridge, s = best_lambda_Ridge, newx = X_test)
#HERE WE DESTANDARDIZE THE FORECASTED LOG(SALES)
#ridge_logsales_pred <- ridge_logsales_pred * Y_stdev[i] + Y_mean[i]  
#HERE WE TRANSFORM THE DESTANDARDIZED LOG(SALES) INTO SALES
sigma_Ridge <- sigma(best_model_Ridge)
ridge_sales_predadj <- exp(ridge_logsales_pred+sigma_Ridge^2/2)

ferrors_Ridge <- sales_true - ridge_sales_predadj

Ridge_measures$MSE[i] <- MSE(ridge_sales_predadj, sales_true)
Ridge_measures$MAE[i] <- MAE(ridge_sales_predadj, sales_true)
Ridge_measures$DM_test[i] <- dm.test(ferrors_OLS, ferrors_Ridge, alternative="greater")$statistic
Ridge_measures$DM_p[i] <- dm.test(ferrors_OLS, ferrors_Ridge, alternative="greater")$p.value 
Ridge_measures$DM_test_g2s[i] <- dm.test(ferrors_g2s, ferrors_Ridge, alternative="greater")$statistic
Ridge_measures$DM_p_g2s[i] <- dm.test(ferrors_g2s, ferrors_Ridge, alternative="greater")$p.value
Ridge_measures$DM_test_lasso[i] <- dm.test(ferrors_Lasso, ferrors_Ridge, alternative="greater")$statistic
Ridge_measures$DM_p_lasso[i] <- dm.test(ferrors_Lasso, ferrors_Ridge, alternative="greater")$p.value 
#}

OLS_measures <- round(OLS_measures,3)
OLS_G2S_measures <- round(OLS_G2S_measures,3)
Ridge_measures <- round(Ridge_measures, 3)
Lasso_measures <- round(Lasso_measures, 3)

OLS.table <- coefficients(model_OLS)
OLS.g2s.table <- coefficients(OLS_G2S_model)
view(OLS.table)
view(OLS.g2s.table)
coefficients(best_model_Lasso)
coefficients(best_model_Ridge)
