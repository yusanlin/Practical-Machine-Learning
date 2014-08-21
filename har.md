Human Activity Recognition Project
========================================================

We first read in the training data:

```r
training <- read.csv("pml-training.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)
```

We take a look at the names (features) of this dataset (for the reason that the number of names/features is large, we don't show the results):

```r
names(training)
```

And found that the first 5 names are: `user_name`,  `raw_timestamp_part_1`, `raw_timestamp_part_2`, and `cvtd_timestamp`. All are not relevant for predicting the activity class, i.e. how well the activity is performed. Also, we also take out the 6th name, , because it is categorical. Therefore, we take all of them out by doing the following:


```r
keeps <- c(7:ncol(training))
training <- training[keeps]
```