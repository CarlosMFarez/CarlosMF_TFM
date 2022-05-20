
#detach(package:plyr)
#library(dummies)
#library(caret)
#library(dplyr)



cruzadagbm<-
 function(data=data,vardep="vardep",
  listconti="listconti",listclass="listclass",
  grupos=4,sinicio=1234,repe=5,
  n.minobsinnode=20,shrinkage=0.1,n.trees=100,interaction.depth=2)
  
 { 

  # Preparación del archivo
  
  # b)pasar las categóricas a dummies
  
  if  (listclass!=c(""))
  {
   databis<-data[,c(vardep,listconti,listclass)]
   databis<- dummy.data.frame(databis, listclass, sep = ".")
  }  else   {
   databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  formu<-formula(paste(vardep,"~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",
   number=grupos,repeats=repe,
   savePredictions = "all") 
  
  # Aplico caret y construyo modelo

  # n.minobsinnode=20,shrinkage=0.1,n.trees=100,interaction.depth=2
  
   gbmgrid <-expand.grid(n.minobsinnode=n.minobsinnode,
    shrinkage=shrinkage,n.trees=n.trees,
    interaction.depth=interaction.depth)
  
  gbm<- train(formu,data=databis,
   method="gbm",trControl=control,
   tuneGrid=gbmgrid,distribution="gaussian",verbose=FALSE)
  
  print(gbm$results)
  
  preditest<-gbm$pred


    preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL

  preditest$error<-(preditest$pred-preditest$obs)^2
    
medias<-preditest %>%
  group_by(Rep) %>%
  summarize(error=mean(error))  
  
return(medias)
  
}


# source("cruzada arbol continua.R")
# source("cruzadas avnnet y lin.R")
# source("cruzada rf continua.R")
# source("cruzada gbm continua.R")
# load("compressbien.Rda")
# 
# data<-compressbien
# 
# medias1<-cruzadaavnnet(data=data,
# vardep="cstrength",listconti=c("age","water","cement","blast"),
# listclass=c(""),grupos=4,sinicio=1234,repe=5,
# size=c(15),decay=c(0.01),repeticiones=5,itera=100)
# 
# medias1$modelo="avnnet"
# 
# medias2<-cruzadalin(data=data,
# vardep="cstrength",listconti=c("age","water","cement","blast"),
# listclass=c(""),grupos=4,sinicio=1234,repe=5)
# 
# medias2$modelo="lineal"
# 
# medias3<-cruzadaarbol(data=data,
# vardep="cstrength",listconti=c("age","water","cement","blast"),
# listclass=c(""),
#  grupos=4,sinicio=1234,repe=5,cp=0,minbucket=5)
# 
# medias3$modelo="arbol"
# 
# medias4<-cruzadarf(data=data,
#  vardep="cstrength",listconti=c("age","water","cement","blast"),
# listclass=c(""),
#   grupos=4,sinicio=1234,repe=5,
#   nodesize=20,replace=TRUE,ntree=200,mtry=4)
# 
# medias4$modelo="bagging"
# 
# 
# medias5<-cruzadarf(data=data,
#  vardep="cstrength",listconti=c("age","water","cement","blast"),
# listclass=c(""),
#   grupos=4,sinicio=1234,repe=5,
#   nodesize=10,replace=TRUE,ntree=600,mtry=3)
# 
# medias5$modelo="rf"
# 
# 
# medias6<-cruzadagbm(data=data,
#  vardep="cstrength",listconti=c("age","water","cement","blast"),
# listclass=c(""),
#   grupos=4,sinicio=1234,repe=5,
# n.minobsinnode=20,shrinkage=0.10,n.trees=500,interaction.depth=2)
# 
# medias6$modelo="gbm"
# 
# union1<-rbind(medias1,medias2,medias3,medias4,medias5,medias6)
# 
# par(cex.axis=0.5)
# boxplot(data=union1,error~modelo)
# 
# 
# union1<-rbind(medias1,medias3,medias4,medias5,medias6)
# 
# par(cex.axis=0.5)
# boxplot(data=union1,error~modelo)
