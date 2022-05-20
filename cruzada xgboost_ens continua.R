library(plyr)
detach(package:plyr)
library(dummies)
library(MASS)
library(reshape)
library(caret)
library(dplyr)

cruzadaxgbm_ens<-
  function(data=data,vardep="vardep",
           listconti="listconti",listclass="listclass",
           grupos=4,sinicio=1234,repe=5,
           min_child_weight=20,eta=0.1,nrounds=100,max_depth=2,
           gamma=0,colsample_bytree=1,subsample=1,alpha=0,lambda=0,lambda_bias=0)
  {
    
    if (any(listclass==c(""))==FALSE)
    {
      databis<-data[,c(vardep,listconti,listclass)]
      databis<- dummy.data.frame(databis, listclass, sep = ".")
    } else {
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
    xgbmgrid <-expand.grid( min_child_weight=min_child_weight,
                            eta=eta,nrounds=nrounds,max_depth=max_depth,
                            gamma=gamma,colsample_bytree=colsample_bytree,subsample=subsample)
    
    xgbm<- train(formu,data=databis,
                 method="xgbTree",trControl=control,
                 tuneGrid=xgbmgrid,verbose=FALSE,
                 alpha=alpha,lambda=lambda,lambda_bias=lambda_bias)
    
    print(xgbm$results)
    
    preditest<-xgbm$pred[,c("pred","obs","Resample")]
    
    
    preditest$prueba<-strsplit(preditest$Resample,"[.]")
    preditest$Fold <- sapply(preditest$prueba, "[", 1)
    preditest$Rep <- sapply(preditest$prueba, "[", 2)
    preditest$prueba<-NULL
    preditest$error<-(preditest$pred-preditest$obs)^2
    
    tabla<-table(preditest$Rep)
    listarep<-c(names(tabla))
    medias<-data.frame()
    for (repi in listarep) {
      paso1<-preditest[which(preditest$Rep==repi),]
      error=mean(paso1$error)
      medias<-rbind(medias,error)
    }
    names(medias)<-"error"
    
    return(list(medias,preditest))
    
  }
