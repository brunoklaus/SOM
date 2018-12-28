setwd("/home/klaus/eclipse-workspace/RedesNeurais_at2/isolet")
wine <- read.csv("isolet1+2+3+4.data",header=F,sep=",")
library(randomForest)
library(magrittr)
colnames(wine)[ncol(wine)] <- "label"


wine[,colnames(wine) != ("label")] %<>% apply(2,scale)

train=sample(1:nrow(wine),round(0.66*nrow(wine)))
#wine.rf=randomForest(label ~ . , data = wine , subset = train)
#plot(wine.rf)

library(MASS)
pca <- prcomp(wine[,-ncol(wine)],subset = train,rank.=3)

library(pca3d)
plot(pca3d(pca,group=factor(wine$label)))