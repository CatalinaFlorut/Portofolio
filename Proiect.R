library(ggplot2)
library(rpart)
library(rpart.plot)
library(rsample)
library(tidyr)
library(dplyr)
library(caret)
library(readr)
library(magrittr)
library(ipred)
library(partykit)
library(randomForest)
library(ranger)
library(ggplot2)
library(modelr)

#citire + afisare date
products<- read_csv("D:/an3/Florut_Catalina_Marchis_Alexandra/summer-products.csv")
summary(products)

ggplot(data=products,aes(x=retail_price, y=units_sold)) + geom_point()+geom_smooth()
ggplot(data=products,aes(x=rating, y=units_sold)) + geom_point()+geom_smooth()
ggplot(data=products,aes(x=merchant_rating, y=units_sold)) + geom_point()+geom_smooth()
ggplot(data=products,aes(x=rating_count, y=units_sold)) + geom_point()+geom_smooth()
ggplot(data=products,aes(x=rating, y=units_sold)) + geom_point()+geom_smooth()



#prelucrare date
products<- select_if(products, is.numeric)
products<- products  %>% mutate(uses_ad_boosts= factor(uses_ad_boosts))
products<- products  %>% mutate(has_urgency_banner= factor(has_urgency_banner))
products<- products  %>% mutate(uses_ad_boosts= factor(uses_ad_boosts))
products<- products  %>% mutate(merchant_has_profile_picture= factor(merchant_has_profile_picture))
products<- products  %>% mutate(shipping_is_express= factor(shipping_is_express))
products<- products  %>% mutate(badge_local_product= factor(badge_local_product))
products<- products  %>% mutate(badge_fast_shipping= factor(badge_fast_shipping))
products<- products  %>% mutate(badge_product_quality= factor(badge_product_quality))
summary(products$units_sold)


#selection predictors
set.seed(131)
products_na<-na.omit(products)
products.rf <- randomForest(units_sold ~ ., products_na, importance=TRUE)
varImpPlot(products.rf) 

#functie de regresie liniara simpla
lm_sales_rating = lm(units_sold ~ rating, data = products)
#review the results
summary(lm_sales_rating)


lm_sales_merchant_rating = lm(units_sold ~ merchant_rating, data = products)

summary(lm_sales_merchant_rating)


lm_sales_rating__three_count = lm(units_sold ~ rating_three_count, data = products)

summary(lm_sales_rating__three_count)


lm_sales_rating_count = lm(units_sold ~ rating_count, data = products)

summary(lm_sales_rating_count)

#linia de regresie rating_count
grid <- products %>%
  data_grid(rating_count = seq_range(rating_count, 100)) %>%
  add_predictions(lm_sales_rating_count, "units_sold")

ggplot(products, aes(rating_count, units_sold)) +
  geom_point() +
  geom_line(data=grid, color="red", size=2)

#intervale de incredere
confint(lm_sales_rating_count)


#regresie liniara multipla

#regresie multipla cu cei mai relevanti 6 predictori
lm_sales_all = lm(data=products, units_sold ~ rating_count + rating_five_count + rating_three_count + rating_four_count  +  rating_two_count + rating_one_count)
summary(lm_sales_all)



#regresie multipla cu cei mai relevanti predictori, din care am eliminat rating_count_one pt ca e irelevant
lm_sales_all_relevant = lm(data=products, units_sold ~ rating_count + rating_five_count + rating_three_count + rating_four_count  +  rating_two_count)
summary(lm_sales_all_relevant)
confint(lm_sales_all_relevant)
tb_sales <- tibble( rating_count = 200, 
                    rating_five_count = 50, 
                    rating_three_count = 40, 
                    rating_four_count = 10,
                    rating_two_count = 50)
predict(lm_sales_all_relevant, newdata = tb_sales, interval = "confidence")
predict(lm_sales_all_relevant, newdata = tb_sales, interval = "prediction")



#split date - date test si date antrenament
set.seed(123)
products_split <-initial_split(products)
products_train <-training(products_split)
products_test <-testing(products_split)


#aplicare formula arbore de decizie
m1 <− rpart(
  formula = units_sold ∼rating_count+rating_five_count+rating_four_count+rating_three_count+rating_two_count+rating_one_count+product_variation_inventory+merchant_rating_count+price+rating+badge_product_quality+badges_count+countries_shipped_to+badge_local_product+retail_price+shipping_option_price,
  data = products_train,
  method = "anova"
)
m1
rpart.plot(m1)
summary(m1)

#optimizare parametrii cu hyper_grid pt fiecare combinatie de maxdepth si minsplit
hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
)
hyper_grid
models<-list()
for (i in 1:nrow(hyper_grid)) {
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  models[[i]] <- rpart(
    formula = units_sold ∼ . ,
    data = products_train,
    method = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  ) }

get_cp<- function(x){
  min<-which.min(x$cptable[,"xerror"])
  cp<- x$cptable[min,"CP"]
}

get_min_error<- function(x){
  min<-which.min(x$cptable[,"xerror"])
  xerror<- x$cptable[min,"xerror"]
}
hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)

#creare arbore de decizie optim
optimal_tree <− rpart(
  formula = units_sold ∼ rating_count+rating_five_count+rating_four_count+rating_three_count+rating_two_count+rating_one_count+product_variation_inventory+merchant_rating_count+price+rating+badge_product_quality+badges_count+countries_shipped_to+badge_local_product+retail_price+shipping_option_price,
  data = products_train,
  method = "anova",
  control = list(minsplit = 8, maxdepth = 10, cp = 0.01)
)
optimal_tree
rpart.plot(optimal_tree)

#realizare predictie
pred <- predict(optimal_tree, newdata = products_test)
summary(pred)
pred
RMSE(pred = pred, obs = products_test$units_sold)

#impartire set de date fara valori nule pentru randomForest
set.seed(123)
products_split_na <-initial_split(products_na)
products_train_na <-training(products_split_na)
products_test_na <-testing(products_split_na)

#aplicare formula randomForest
m1_rf<- randomForest(
  formula =units_sold ∼ rating_count+rating_five_count+rating_four_count+rating_three_count+rating_two_count+rating_one_count+product_variation_inventory+merchant_rating_count+price+rating+badge_product_quality+badges_count+countries_shipped_to+badge_local_product+retail_price+shipping_option_price,
  data= products_train_na
)
m1_rf


#bagging cu ajutorul librariei ipred
bagged_m1 <− bagging(
  formula = units_sold ∼ rating_count+rating_five_count+rating_four_count+rating_three_count+rating_two_count+rating_one_count+product_variation_inventory+merchant_rating_count+price+rating+badge_product_quality+badges_count+countries_shipped_to+badge_local_product+retail_price+shipping_option_price,
  data = products_train,
  coob = TRUE
)

# by default se folosesc 25 bags
bagged_m1


#optimizarea procedurii de bagging
ntree<- 10:50
rmse<-vector(mode = "numeric",length = length(ntree)) 

for (i in seq_along(ntree)) {
  set.seed(123)
  model <− bagging(
    formula =units_sold ∼rating_count+rating_five_count+rating_four_count+rating_three_count+rating_two_count+rating_one_count+product_variation_inventory+merchant_rating_count+price+rating+badge_product_quality+badges_count+countries_shipped_to+badge_local_product+retail_price+shipping_option_price,
    data = products_train,
    coob = TRUE,
    nbagg = ntree[i] )
  rmse[i] = model$err }

#grafic pentru a vizualiza rmse
plot(ntree, rmse, type ="l", lwd=2)
abline(v=25, col = "red", lty="dashed")

#bagging optimizat
bagged_m2 <− bagging(
  formula = units_sold ∼ rating_count+rating_five_count+rating_four_count+rating_three_count+rating_two_count+rating_one_count+product_variation_inventory+merchant_rating_count+price+rating+badge_product_quality+badges_count+countries_shipped_to+badge_local_product+retail_price+shipping_option_price,
  data = products_train_na,
  nbag = 21,
  coob = TRUE
)
# by default se folosesc 25 bags
bagged_m2


#bagging cu metoda Cross Validation
fitControl <− trainControl(
  method = "cv",
  number = 10
)

bagged_cv <− train(
  units_sold ∼ rating_count+rating_five_count+rating_four_count+rating_three_count+rating_two_count+rating_one_count+product_variation_inventory+merchant_rating_count+price+rating+badge_product_quality+badges_count+countries_shipped_to+badge_local_product+retail_price+shipping_option_price,
  data = products_train_na,
  method = "treebag",
  trControl = fitControl,
  importance = TRUE
)
bagged_cv
