#Megan Stiles
#MES5AC

#Question 3

##1.	Using the “Income” dataset (a dataset supplied by R),  you will generate basic association rules.  
#(a)Compute the top ten rules with the highest lift.  

library(arules)
library(datasets)
data("Income")

myrules<- apriori(data = Income, parameter = list(support = 0.1, confidence = 0.1, minlen = 5, maxlen = 20))
myrules<- apriori(data = Income, parameter = list(support = 0.2, confidence = 0.15, minlen = 1, maxlen = 20))
inspect(sort(myrules, by = 'lift')[1:10])

    
#lhs                           rhs                         support confidence     lift
#[1]  {marital status=married,                                                             
#language in home=english} => {dual incomes=yes}        0.2190227  0.6167076 2.434260
#[2]  {dual incomes=yes,                                                                   
#language in home=english} => {marital status=married}  0.2190227  0.9365672 2.428294
#[3]  {dual incomes=yes}         => {marital status=married}  0.2370564  0.9357061 2.426061
#[4]  {marital status=married}   => {dual incomes=yes}        0.2370564  0.6146305 2.426061
#[5]  {number in household=1,                                                              
#householder status=rent}  => {type of home=apartment}  0.2246946  0.6582872 2.388592
#[6]  {number of children=0,                                                               
#householder status=rent}  => {type of home=apartment}  0.2052065  0.6559740 2.380199
#[7]  {number in household=1,                                                              
#householder status=rent,                                                            
#language in home=english} => {type of home=apartment}  0.2089878  0.6549681 2.376549
#[8]  {number of children=0,                                                               
#type of home=apartment}   => {householder status=rent} 0.2052065  0.9598639 2.290085
#[9]  {age=35+,                                                                            
#type of home=house,                                                                 
#language in home=english} => {householder status=own}  0.2293485  0.8538170 2.271999
#[10] {income=$0-$40,000,                                                                  
#householder status=rent}  => {type of home=apartment}  0.2017161  0.6239316 2.263933

#(b) Using the explanation and instructions on page 9 and 10 of the CRAN project apriori.pdf 
#(https://cran.r-project.org/web/packages/arules/arules.pdf ),  
#find rules only pertaining to people of hispanic ethnicity, over at least 35 years of age with an income of at least $40,000.  
#What is the rule with the highest confidence that you obtain?  What is its lift?


new_rules <- apriori(data = Income, parameter = list(supp= 0.01, conf = 0.5, minlen = 3, maxlen = 40),
              appearance = list(both = c('age=35+', 'ethnic classification=hispanic', 'income=$40,000+'), default = 'both'))
new_rules <- apriori(data = Income, parameter = list(support= 0.01, confidence = 0.5, minlen = 3, maxlen = 40))
new_rules.subset<- subset(new_rules, subset = lhs %ain% c("age=35+", 'income=$40,000+', 'ethnic classification=hispanic')) #, 'ethnic classification=hispanic'))
inspect((new_rules.subset))

#  lhs                                 rhs                         support confidence     lift
#[1] {income=$40,000+,                                                                          
#age=35+,                                                                                  
#ethnic classification=hispanic} => {householder status=own} 0.01032577  0.7977528 2.122813
#[2] {income=$40,000+,                                                                          
#age=35+,                                                                                  
#ethnic classification=hispanic} => {marital status=married} 0.01003490  0.7752809 2.010117
#[3] {income=$40,000+,                                                                          
#age=35+,                                                                                  
#ethnic classification=hispanic} => {years in bay area=10+}  0.01119837  0.8651685 1.338034

#highest confidence rule = {income=$40,000+,age=35+,ethnic classification=hispanic} => {years in bay area=10+}
#lift = 1.338034


#Question 4
#Create a decision tree for the mushroom data from https://archive.ics.uci.edu/ml/datasets/Mushroom.  
#Create and evaluate the model using 10-fold cross-validation.  

install.packages("C50")
library(C50)
library(caret)
library(mlbench)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
mushrooms <- read.table(file = url, header = FALSE, sep = ",")
colnames(mushrooms) <- c("Class","cap-shape", "cap-surface", "cap-color", "bruises", 
                        "odor","gill-atachment","gill-spacing","gill-size","gill-color",
                        "stalk-shape","stalk-root","stalk-surface-above-ring", 
                        "stalk-surface-below-ring", "stalk-color-above-ring","stalk-color-below-ring",
                        "veil-type","veil-color","ring-number","ring-type","spore-print-color",
                        "population","habitat")
#Make response variable a Factor
mushrooms$Class<- factor(mushrooms$Class)

#Drop variable with the same value

mushrooms<- mushrooms[, -17]

#Split data into training and testing

training.indices = sample(1:nrow(mushrooms), as.integer(nrow(mushrooms) * 0.75))
training.set = mushrooms[training.indices,]
testing.set = mushrooms[-training.indices,]


mushrooms_model <- C5.0(x = training.set[,-1], y= training.set$Class)
mushrooms_model
summary(mushrooms_model)

#Make Predictions
mushrooms_pred <- predict(mushrooms_model, testing.set)

library(gmodels)
table<-CrossTable(testing.set$Class, mushrooms_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
#Confusion Matrix

matrix<- confusionMatrix(mushrooms_pred, testing.set$Class)
matrix$overall['Accuracy']

#10-fold CV

#Create Folds
folds<- createFolds(mushrooms$Class, k=10, list = TRUE, returnTrain = FALSE)

#initialzie empty vector to store accuracy
raw_accuracy<- vector()
i=0
for (i in 1:10) {
  #Create testing indicies based on folds
  test.indices<- folds[[i]]
  
  #Create training and testing sets
  train = mushrooms[-test.indices,]
  test = mushrooms[test.indices,]
  
  #Train Model
  model = C5.0( x = train[,-1], y = train$Class)
  
  #Make predictions based on model for testing set
  predictions = predict(model, test)
  
  #create confusion Matrix
  matrix<- confusionMatrix(predictions, test$Class)
  
  #Extract Accuracy from confusion Matrix
  accuracy = matrix$overall['Accuracy']
  
  #Store accuracy for each run in vector
  raw_accuracy[i]= accuracy
}

Total_Accuracy = mean(raw_accuracy)
Total_Accuracy #0.9995