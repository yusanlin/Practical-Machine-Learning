# read in training sets
training <- read.csv("pml-training.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)

keeps <- c(7:ncol(training))
training <- training[keeps]

# first handle missing values
training[training == ""] <- NA

# encode dummy variable
# encode "new_window"
# library(dummies)
# training$new_window <- dummy(training$new_window)
# testing$new_window <- dummy(testing$new_window)

# only keep the columns with numeric values
numeric <- sapply(training, is.numeric)
training <- cbind(training[numeric], training$classe)
names(training)[length(names(training))] <- "classe"
training$classe <- as.factor(training$classe) # so that it can be predicted

# only keep the data with no missing values
training <- training[complete.cases(training), ]

n_features = ncol(training) - 1 # because the very last column is the y

# PCA
library(caret)
preProc <- preProcess(training[, -(n_features + 1)], method="pca", pcaComp=2)
trainPC <- predict(preProc,training[, -(n_features + 1)])
modelFit <- train(training$classe ~ ., method="svmLinear", data=trainPC)