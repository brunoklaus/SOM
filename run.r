setwd("/home/klaus/eclipse-workspace/RedesNeurais_at1")
args = commandArgs(trailingOnly=TRUE)

getFileStr <- function(typ="acc",dataset,lr,arch,mom) {

str <- paste0("/home/klaus/eclipse-workspace/RedesNeurais_at1/results/",
              dataset,"/",typ,"/arch=",arch,"_lr=",
              ifelse(lr==1,"1.0",lr),
              "_momentum=",
              ifelse(mom,yes = "True",no = "False"),
              "_lamb=0_dropout=0_dataset=",dataset,"_cv=",
              ifelse(dataset=="iris","1","5"),".csv",
              collapse="",sep=""
              )
  return(str)
}
MOD <- 3
this_mod <- as.character(args[1])
print(this_mod)
options(scipen=999)

getStr <- function (dataset,lr,arch,mom) {
  str <-paste0("python3 main.py --dataset='",dataset,"' --lr=",lr,
               ifelse(mom,yes = " --momentum",no = "")," --arch='",arch,"'",collapse = "",
               sep="")
  return(str)
  
}

arch_r = c("h2048r","h64r","h16r","h4r;h4r","h32r;h32r","h64r;h64r",
           "h128r;h64r",
           "h2048r;h2048r","h16r;h16r;h16r","h16r;h16r;h16r;h16r",
           "h1024r;h1024r;h1024r","h1024r;h1024r;h1024r;h1024r")
arch_s = gsub(pattern = "r",replacement = "s",x = arch_r)

arch = c(arch_r,arch_s)
mom = c(T,F)
ds <- c("iris","isolet","wine")
lr = c(0.0001,0.001,0.01,0.1,1.0)

library(tidyr)
df <- tidyr::crossing(mom,lr,arch,ds)
colnames(df) <- c("mom","lr","arch","ds")

go <-function(){

for (i in 1:nrow(df)){
  if (i %% MOD != this_mod){
    print(paste0("Jumping",i))
    next
  }
  x <- df[i,]
  print(i)
  print(df[i,])
  comm <- getStr(dataset=x["ds"],
                 lr = x["lr"],
                 arch = x["arch"],
                 mom = x["mom"])
  print(comm)
  #print(comm)
  (system(command = comm,ignore.stdout = F, ignore.stderr = F));
}
}
go()
return(T)
###############################################
df <- cbind(df,sapply(1:nrow(df),
                      function(i){
                        x <- df[i,]
                        s <- getFileStr( typ="acc",
                                      dataset=x["ds"],
                                     lr = x["lr"],
                                     arch = x["arch"],
                                     mom = x["mom"])
                        return(as.character(s))
                      }
                      ))
colnames(df)[ncol(df)] <- "File_acc"
df$File_acc <- as.character(df$File_acc)

df$File_exists <- sapply(1:nrow(df),
       function(i){
          return(file.exists(df[i,"File_acc"]))                    
       }
  )
df <- df[df$File_exists == T,]

last <- (matrix(data = unlist(sapply(df$File_acc,
               function(f){
                 cf <- read.csv(f,sep=",",header = T,row.names = 1)
                 return(cf[nrow(cf),])
                 
               })),nrow=nrow(df),byrow = TRUE))

df <- cbind(df,last)
colnames(df)[(ncol(df)-3):ncol(df)] <- c("last_train_avg_loss",
                                         "last_train_error_rate",
                                         "last_test_avg_loss",
                                         "last_test_error_rate"
                                         )
df$last_test_minus_train <- df$last_test_error_rate - df$last_train_error_rate

df$uses_s <- sapply(df$arch,function(x)return(x%in%arch_s))

#########################################
#IRIS - last_test_error x mom
df_iris <- df[df$ds=="iris",]
#Histogram of testing accuracies
library(ggplot2)
g <- ggplot(df_iris) + 
  ggplot2::geom_histogram(aes(x=last_test_error_rate,fill=mom),
                          position = "stack") + 
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Iris - Histogram of testing error at epoch 200")+
  xlab("error") + ylab("freq")   + labs(fill='Momentum') 
plot(g)
#########################################
#ISOLET - last_test_error x mom
df_isolet <- df[df$ds=="isolet",]
#Histogram of testing accuracies
library(ggplot2)
g <- ggplot(df_isolet) + 
  ggplot2::geom_histogram(aes(x=last_test_error_rate,fill=mom),
                          position = "stack") + 
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Isolet - Histogram of testing error at epoch 350")+
  xlab("error") + ylab("freq")   + labs(fill='Momentum') 
plot(g)
#########################################
#WINE - last_test_error x mom
df_wine <- df[df$ds=="wine",]
#Histogram of testing accuracies
library(ggplot2)
g <- ggplot(df_wine) + 
  ggplot2::geom_histogram(aes(x=last_test_error_rate,fill=mom),
                          position = "stack") + 
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Wine - Histogram of testing error at epoch 1000")+
  xlab("error") + ylab("freq")   + labs(fill='Momentum') 
plot(g)

#########################################
#IRIS - last_test_error x lr
df_iris <- df[df$ds=="iris",]
df_iris$lr <- as.factor(df_iris$lr)
#Histogram of testing accuracies
library(ggplot2)
g <- ggplot(df_iris) + 
  ggplot2::geom_histogram(aes(x=last_test_error_rate,fill=lr),
                          position = "stack") + 
   theme(panel.background = element_rect(fill = 'gray')) +
  scale_fill_brewer(type = "div", palette = "Spectral") +
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Iris - Histogram of testing error at epoch 200")+
  xlab("error") + ylab("freq")   + labs(fill='Learning rate') 
plot(g)
#########################################
#ISOLET - last_test_error x lr
df_isolet <- df[df$ds=="isolet",]
df_isolet$lr <- as.factor(df_isolet$lr)
#Histogram of testing accuracies
library(ggplot2)
g <- ggplot(df_isolet) + 
  ggplot2::geom_histogram(aes(x=last_test_error_rate,fill=lr),
                          position = "stack") + 
  theme(panel.background = element_rect(fill = 'gray')) +
  scale_fill_brewer(type = "div", palette = "Spectral") +
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Isolet - Histogram of testing error at epoch 350")+
  xlab("error") + ylab("freq")   + labs(fill='Learning rate') 
plot(g)

#########################################
#ISOLET - last_test_error x lr
df_wine <- df[df$ds=="wine",]
df_wine$lr <- as.factor(df_wine$lr)
#Histogram of testing accuracies
library(ggplot2)
g <- ggplot(df_wine) + 
  ggplot2::geom_histogram(aes(x=last_test_error_rate,fill=lr),
                          position = "stack") + 
  theme(panel.background = element_rect(fill = 'gray')) +
  scale_fill_brewer(type = "div", palette = "Spectral") +
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Wine - Histogram of testing error at epoch 1000")+
  xlab("error") + ylab("freq")   + labs(fill='Learning rate') 
plot(g)
#########################################
#ISOLET - last_test_error - last_train_error 
df_isolet <- df[df$ds=="isolet",]
df_isolet$lr <- as.factor(df_isolet$lr)
#Histogram of training accuracies
library(ggplot2)
g <- ggplot(df_isolet) + 
  ggplot2::geom_histogram(aes(x=last_test_minus_train,fill=lr),
                          position = "stack") + 
  theme(panel.background = element_rect(fill = 'gray')) +
  scale_fill_brewer(type = "div", palette = "Spectral") +
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Isolet - Test error MINUS training error")+
  xlab("diff") + ylab("freq")   + labs(fill='learning rate') 
plot(g)
################################################
#IRIS - last_test_error - last_train_error 
df_iris <- df[df$ds=="iris",]
df_iris$lr <- as.factor(df_iris$lr)
#Histogram of training accuracies
library(ggplot2)
g <- ggplot(df_iris) + 
  ggplot2::geom_histogram(aes(x=last_test_minus_train,fill=lr),
                          position = "stack") + 
  theme(panel.background = element_rect(fill = 'gray')) +
  scale_fill_brewer(type = "div", palette = "Spectral") +
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Iris - Test error MINUS training error")+
  xlab("diff") + ylab("freq")   + labs(fill='learning rate') 
plot(g)
############################################################
#WINE - last_test_error - last_train_error 
df_wine <- df[df$ds=="wine",]
df_wine$lr <- as.factor(df_wine$lr)
#Histogram of training accuracies
library(ggplot2)
g <- ggplot(df_wine) + 
  ggplot2::geom_histogram(aes(x=last_test_minus_train,fill=lr),
                          position = "stack") + 
  theme(panel.background = element_rect(fill = 'gray')) +
  scale_fill_brewer(type = "div", palette = "Spectral") +
  #geom_text(aes(x=names(freq),y=(freq),label = freq), vjust = -0.5,size=16) +
  ggtitle("Wine - Test error MINUS training error")+
  xlab("diff") + ylab("freq")   + labs(fill='learning rate') 
plot(g)
#######################################################################
df_s <- df[df$uses_s==T,]
differences <- vector(mode = "list",length = nrow(df_s))
for(i in 1:nrow(df_s)){
  xx <- df_s[i,1:4]
  xx[3] <- gsub(pattern = "s","r",xx[3])
  y <- NULL
  for(j in 1:nrow(df)) {
    if(all((df[j,])[1:4] == xx)){
      y <- j
      break
    }
  }
  differences[i] <- df_s[i,"last_test_error_rate"] - df[y,"last_test_error_rate"] 
  
  print(i)
  #print(rbind(df[y,],  df_s[i,]))
}
differences <- unlist(differences)
differences_iris <- differences[df_s$ds == "iris"]
differences_isolet <- differences[df_s$ds == "isolet"]
differences_wine <- differences[df_s$ds == "wine"]

arch_df <- data.frame(arch=sort(unique(df$arch)))
iris_best_train_which <- sapply(arch_df$arch,
          function(arch) {
            temp <- df_iris[df_iris$arch==arch,]
            res <- temp$lr[which(
              temp$last_train_error_rate == 
                min(temp$last_train_error_rate))]
            return(paste0(res,collapse=";",sep=""))
          })
isolet_best_train_which <- 
  sapply(arch_df$arch,
          function(arch) {
            temp <- df_isolet[df_isolet$arch==arch,]
            res <- temp$lr[which(
              temp$last_train_error_rate == 
                min(temp$last_train_error_rate))]
            return(paste0(res,collapse=";",sep=""))
          })
wine_best_train_which <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_wine[df_wine$arch==arch,]
           res <- temp$lr[which(
             temp$last_train_error_rate == 
               min(temp$last_train_error_rate))]
           return(paste0(res,collapse=";",sep=""))
         })
iris_best_test_which <-
  sapply(arch_df$arch,
            function(arch) {
              temp <- df_iris[df_iris$arch==arch,]
              res <- temp$lr[which(
                temp$last_test_error_rate == 
                  min(temp$last_test_error_rate))]
              return(paste0(res,collapse=";",sep=""))
            })
isolet_best_test_which <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_isolet[df_isolet$arch==arch,]
           res <- temp$lr[which(
             temp$last_test_error_rate == 
               min(temp$last_test_error_rate))]
           return(paste0(res,collapse=";",sep=""))
         })
wine_best_test_which <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_wine[df_wine$arch==arch,]
           res <- temp$lr[which(
             temp$last_test_error_rate == 
               min(temp$last_test_error_rate))]
           return(paste0(res,collapse=";",sep=""))
         })
iris_best_train <- 
  sapply(arch_df$arch,
          function(arch) {
            temp <- df_iris[df_iris$arch==arch,]
            res <- min(temp$last_train_error_rate)
            return(res)
          })
iris_best_test <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_iris[df_iris$arch==arch,]
           res <- min(temp$last_test_error_rate)
           return(res)
         })
isolet_best_train <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_isolet[df_isolet$arch==arch,]
           res <- min(temp$last_train_error_rate)
           return(res)
         })
isolet_best_test <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_isolet[df_isolet$arch==arch,]
           res <- min(temp$last_test_error_rate)
           return(res)
         })
isolet_best_train <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_isolet[df_isolet$arch==arch,]
           res <- min(temp$last_train_error_rate)
           return(res)
         })
wine_best_test <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_wine[df_wine$arch==arch,]
           res <- min(temp$last_test_error_rate)
           return(res)
         })
wine_best_train <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_wine[df_wine$arch==arch,]
           res <- min(temp$last_train_error_rate)
           return(res)
         })
iris_avg_train <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_iris[df_iris$arch==arch,]
           res <- mean(temp$last_train_error_rate)
           return(res)
         })
isolet_avg_train <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_isolet[df_isolet$arch==arch,]
           res <- mean(temp$last_train_error_rate)
           return(res)
         })
wine_avg_train <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_wine[df_wine$arch==arch,]
           res <- mean(temp$last_train_error_rate)
           return(res)
         })
iris_avg_test <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_iris[df_iris$arch==arch,]
           res <- mean(temp$last_test_error_rate)
           return(res)
         })
isolet_avg_test <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_isolet[df_isolet$arch==arch,]
           res <- mean(temp$last_test_error_rate)
           return(res)
         })
wine_avg_test <- 
  sapply(arch_df$arch,
         function(arch) {
           temp <- df_wine[df_wine$arch==arch,]
           res <- mean(temp$last_test_error_rate)
           return(res)
         })
archinfo <- list(iris_best_train_which,
              iris_best_train,
              iris_avg_train,
              iris_best_test_which,
              iris_best_test,
              iris_avg_test,
              isolet_best_train_which,
              isolet_best_train,
              isolet_avg_train,
              isolet_best_test_which,
              isolet_best_test,
              isolet_avg_test,
              wine_best_train_which,
              wine_best_train,
              wine_avg_train,
              wine_best_test_which,
              wine_best_test,
              wine_avg_test
)


for (x in archinfo) {
  arch_df <- cbind.data.frame(arch_df, x)
}
colnames(arch_df) <-
  c("arch",
    "iris_best_train_which",
    "iris_best_train",
    "iris_avg_train",
    "iris_best_test_which",
    "iris_best_test",
    "iris_avg_test",
    "isolet_best_train_which",
    "isolet_best_train",
    "isolet_avg_train",
    "isolet_best_test_which",
    "isolet_best_test",
    "isolet_avg_test",
    "wine_best_train_which",
    "wine_best_train",
    "wine_avg_train",
    "wine_best_test_which",
    "wine_best_test",
    "wine_avg_test"
    )