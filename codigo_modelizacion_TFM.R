# **********************************
# Librerías
# **********************************
library(e1071)
library(caret)
library(MASS)
library(dummies)
library(naniar)
library(nnet)
library(NeuralNetTools)
library(ggplot2)
library(plotly)
library(dplyr)
library(data.table)
library(reshape)
library(pROC)
library(reshape2)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(gbm)
library(xgboost)
library(caretEnsemble)
library(parallel)
library(doParallel)
library(visualpred)
library(sas7bdat)
library(foreign)
library(klaR)
library(Boruta)
library(MXM)
library(readr)
library(klaR)
library(Boruta)
library(MXM)
library(knitr)
library(performance)
library(corrplot)

cluster <- makeCluster(detectCores() - 2)
registerDoParallel(cluster)

# **********************************
# Importación de los datos
# **********************************

#Funciones a utilizar
source("cruzadas avnnet y lin.R")
source("cruzada arbol continua.R")
source("cruzada rf continua.R")
source("cruzada gbm continua.R")
source("cruzada xgboost continua.R")
source("cruzada SVM continua lineal.R")
source("cruzada SVM continua polinomial.R")
source("cruzada SVM continua RBF.R")

# Validación cruzada simple
control<-trainControl(method = "cv",number=10,savePredictions = "all")

datos <- read_csv("Datos_dep_TFM.csv")
as.data.frame(datos)
datos<- dplyr::rename(datos, farmacia = LOG_IMP_REP_farmacia, Tot_Enf = IMP_REP_TotEnf, Tot_Med = IMP_REP_TotMed, Tot_Lab = IMP_REP_TotLab)


listclass=c("extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac")
listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
            "Tot_Lab", "Tot_Med")
vardep<-"farmacia"

datos_dep<-datos[,c(listconti,listclass,vardep)]

control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# **********************************
# Regresión
# **********************************


lineal1<-cruzadalin(data=datos_dep,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Lab", "Tot_Med","Tot_Enf","I_Complejidad"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20)

lineal1$modelo="Reg_SV"

lineal2<-cruzadalin(data=datos_dep,
                    vardep="farmacia",listconti=
                      c("EDAD_pac", "Tot_Med","Tot_Lab","Tot_Enf"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20)

lineal2$modelo="Reg_MCO"

lineal3<-cruzadalin(data=datos_dep,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Lab", "Tot_Med","Tot_Enf","I_Complejidad","P_Asistencial","sexo_pac"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20)

lineal3$modelo="Reg_lineal"

lineal4<-cruzadalin(data=datos_dep,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Lab", "Tot_Med","Tot_Enf","I_Complejidad","P_Asistencial","sexo_pac","Edad_med","RMD"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20)

lineal4$modelo="Reg_GB"

lineal5<-cruzadalin(data=datos_dep,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Lab", "Tot_Med","Tot_Enf","I_Complejidad","P_Asistencial","sexo_pac","Edad_med","RMD","Sexo_med","ESPECIALIDAD"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20)

lineal5$modelo="Reg_boruta"

lineal6<-cruzadalin(data=datos_dep,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Lab", "Tot_Med","Tot_Enf","I_Complejidad","P_Asistencial","sexo_pac","Edad_med","RMD","Sexo_med"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20)

lineal6$modelo="Reg_AIC"

lineal7<-cruzadalin(data=datos_dep,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Med","Tot_Enf"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20)

lineal7$modelo="Reg_BIC"

lineal8<-cruzadalin(data=datos_dep,
                    vardep="farmacia",listconti=
                      c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                        "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                    listclass=c(""),grupos=8,sinicio=1234,repe=20)

lineal8$modelo="all_var"

#Boxplot
union1<-rbind(lineal1,lineal2,lineal3,lineal4,lineal5,lineal6,lineal7,lineal8)

par(cex.axis=0.8)
boxplot(data=union1,col="blue",error~modelo)
union1$RMSE<-sqrt(union1$error)
par(cex.axis=1.2)
boxplot(data=union1,col="blue",RMSE~modelo)

#Hipotesis paramétricas
reg_final<-lm(data=datos_dep,farmacia~EDAD_pac+Tot_Med+Tot_Enf+sexo_pac+I_Complejidad+Tot_Lab+P_Asistencial)
check_model(reg_final)

#nterpretación de los coeficientes
summary(reg_final)

# **********************************
# Árboles de regresión
# **********************************

#Optimización del minbucket
set.seed(1234)
arbolgrid <- expand.grid(cp=c(0))
for (minbu in seq(from=50, to=180, by=5))
{
  arbolcaret<- train(farmacia~.,
                     data=datos_dep,method="rpart",minbucket=minbu,
                     trControl=control,tuneGrid=arbolgrid)
  print(minbu)
  print(arbolcaret)
}

arbol1<-cruzadaarbol(data=datos_dep,
                     vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                   "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                     listclass=c(""),grupos=10,sinicio=1234,repe=20,cp=0,minbucket=50)
arbol1$modelo="arbol_50"

arbol2<-cruzadaarbol(data=datos_dep,
                     vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                   "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                     listclass=c(""),grupos=10,sinicio=1234,repe=20,cp=0,minbucket=85)
arbol2$modelo="arbol_85"

arbol3<-cruzadaarbol(data=datos_dep,
                     vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                   "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                     listclass=c(""),grupos=10,sinicio=1234,repe=20,cp=0,minbucket=95)
arbol3$modelo="arbol_95"

arbol4<-cruzadaarbol(data=datos_dep,
                     vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                   "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                     listclass=c(""),grupos=10,sinicio=1234,repe=20,cp=0,minbucket=105)
arbol4$modelo="arbol_105"

arbol5<-cruzadaarbol(data=datos_dep,
                     vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                   "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                     listclass=c(""),grupos=10,sinicio=1234,repe=20,cp=0,minbucket=115)
arbol5$modelo="arbol_115"

arbol6<-cruzadaarbol(data=datos_dep,
                     vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                   "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                     listclass=c(""),grupos=10,sinicio=1234,repe=20,cp=0,minbucket=145)
arbol6$modelo="arbol_145"

arbol7<-cruzadaarbol(data=datos_dep,
                     vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                   "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                     listclass=c(""),grupos=10,sinicio=1234,repe=20,cp=0,minbucket=150)
arbol7$modelo="arbol_150"

#Boxplot de todos los árboles
union2<-rbind(arbol1,arbol2,arbol3,arbol4,arbol5,arbol6,arbol7)
par(cex.axis=0.5)
boxplot(data=union2,error~modelo)

union2$RMSE<-sqrt(union2$error)

par(cex.axis=1.2)
boxplot(data=union2,col="blue",RMSE~modelo)

#Especificamos el árbol óptimo
arbol_opt <- rpart(farmacia~ .,
                   data = datos_dep,minbucket =85,method = "anova",maxsurrogate=0,cp=0)
summary(arbol_opt)

#Importancia de las variables
asRules(arbol_opt)
arbol_opt$variable.importance
par(cex=0.7)
barplot(arbol_opt$variable.importance,col="red")
#Representación gráfica del árbol óptimo
rpart.plot(arbol_opt,extra=1, nn=TRUE)

#Representación gráfica simplificada del árbol

arbol_simp <- rpart(farmacia~ .,data = datos_dep,minbucket =450,cp=0)
rpart.plot(arbol_simp,extra=1, nn=TRUE)

# **********************************
# Red neuronal
# **********************************
#Distribución de las principales variables del modelo
plot_ly(datos_dep, x = ~Tot_Lab, y = ~EDAD_pac, z = ~farmacia, type='mesh3d')
plot_ly(datos_dep, x = ~Tot_Med, y = ~EDAD_pac, z = ~farmacia, type='mesh3d')
plot_ly(datos_dep, x = ~Tot_Enf, y = ~EDAD_pac, z = ~farmacia, type='mesh3d')
plot_ly(datos_dep, x = ~I_Complejidad, y = ~EDAD_pac, z = ~farmacia, type='mesh3d')
plot_ly(datos_dep, x = ~P_Asistencial, y = ~EDAD_pac, z = ~farmacia, type='mesh3d')

#Especificación del modelo óptimo (tuneo de la red)

vardep="farmacia"
sel_var<-c("EDAD_pac","Tot_Lab", "Tot_Med","Tot_Enf","I_Complejidad","P_Asistencial","sexo_pac")

data2<-datos_dep[,c(sel_var,vardep)]

#Estandarización
means <-apply(data2[,sel_var],2,mean,na.rm=TRUE)
sds<-sapply(data2[,sel_var],sd,na.rm=TRUE)

datos2<-scale(data2[,sel_var], center = means, scale = sds)

datos_est<-data.frame(cbind(datos2,data2[,c(vardep)]))


control<-trainControl(method = "cv",
                      number=10,savePredictions = "all") 

set.seed(1234)
nnetgrid <-  expand.grid(size=c(5,10,15,20,26),decay=c(0.01,0.1,0.001,0.0001),bag=F)

completo<-data.frame()
listaiter<-c(10,50,100,250,500,1000,2000,3000)

for (iter in listaiter)
{
  rednnet<- train(farmacia~.,
                  data=datos_est,
                  method="avNNet",linout = TRUE,maxit=iter,
                  trControl=control,tuneGrid=nnetgrid,trace=F)
  # Añado la columna del parametro de iteraciones
  rednnet$results$itera<-iter
  # Voy incorporando los resultados a completo
  completo<-rbind(completo,rednnet$results)
  
  
}

completo<-completo[order(completo$RMSE),]

ggplot(completo, aes(x=factor(itera), y=RMSE, 
                     color=factor(decay),pch=factor(size))) +
  geom_point(position=position_dodge(width=0.5),size=3)


set.seed(4321)
nnetgrid_opt <- expand.grid(size=c(5,10,15),
                            decay=c(0.01,0.1,0.001,0.0001),bag=F)

rednnet_opt<- train(farmacia~EDAD_pac+Tot_Med+Tot_Enf+sexo_pac+I_Complejidad+Tot_Lab+P_Asistencial,
                    data=datos_est,
                    method="avNNet",linout = TRUE,maxit=250,
                    trControl=control,tuneGrid=nnetgrid_opt)

rednnet_opt


red1<-cruzadaavnnet(data=datos_est,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20,itera=250,
                    size=c(15),decay=c(0.001))

red1$modelo="Red1"

red2<-cruzadaavnnet(data=datos_est,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20,itera=250,
                    size=c(10),decay=c(0.001))

red2$modelo="Red2"

red3<-cruzadaavnnet(data=datos_est,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20,itera=250,
                    size=c(10),decay=c(0.1))

red3$modelo="Red3"

red4<-cruzadaavnnet(data=datos_est,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20,itera=250,
                    size=c(15),decay=c(0.01))

red4$modelo="Red4"


red5<-cruzadaavnnet(data=datos_est,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20,itera=250,
                    size=c(10),decay=c(0.0001))

red5$modelo="Red5"

red6<-cruzadaavnnet(data=datos_est,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20,itera=250,
                    size=c(5),decay=c(0.1))

red6$modelo="Red6"

red7<-cruzadaavnnet(data=datos_est,
                    vardep="farmacia",listconti=
                      c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                    listclass=c(""),grupos=10,sinicio=1234,repe=20,itera=500,
                    size=c(15),decay=c(0.1))

red7$modelo="Red7"


#Boxplot
union3<-rbind(red1,red2,red3,red4,red5,red6,red7)
par(cex.axis=0.8)
boxplot(data=union3,col="blue",error~modelo)

union3$RMSE<-sqrt(union3$error)

par(cex.axis=1.2)
boxplot(data=union3,col="blue",RMSE~modelo)


set.seed(1234)
red_optima<-nnet(data=datos_est, farmacia~EDAD_pac+Tot_Med+Tot_Enf+sexo_pac+I_Complejidad+Tot_Lab+P_Asistencial,linout = TRUE,size=10,maxit=250,decay=0.0001)
summary(red_optima)

# **********************************
# BAGGING
# **********************************

#Búsqueda del número de árboles óptimos a realizar
set.seed(1234)
rfbis<-randomForest(farmacia~.,
                    data=datos_dep,
                    mtry=13,ntree=1000,nodesize=85,replace=TRUE)

rfbis$mse<-sqrt(rfbis$mse)
rfbis
plot(rfbis$mse)

#Cambio del tamaño de nodo para identificar diferencias respecto al árbol óptimo.
set.seed(1234)
rfbis2<-randomForest(farmacia~.,
                     data=datos_dep,
                     mtry=13,ntree=1000,nodesize=60,replace=TRUE)

rfbis2$mse<-sqrt(rfbis2$mse)
rfbis2
plot(rfbis2$mse) 

set.seed(1234)
rfbis3<-randomForest(farmacia~.,
                     data=datos_dep,
                     mtry=13,ntree=1000,nodesize=120,replace=TRUE)
rfbis3
rfbis3$mse<-sqrt(rfbis3$mse)
plot(rfbis3$mse) 

bagging1<-cruzadarf(data=datos_dep,
                    vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                  "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                    listclass=c(""),
                    grupos=10,sinicio=1234,repe=20,
                    nodesize=85,replace=TRUE,sampsize=1000,ntree=130,mtry=13)
bagging1$modelo="bagging1000"

bagging2<-cruzadarf(data=datos_dep,
                    vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                  "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                    listclass=c(""),
                    grupos=10,sinicio=1234,repe=20,
                    nodesize=85,replace=TRUE,sampsize=1500,ntree=130,mtry=13)
bagging2$modelo="bagging1500"


bagging3<-cruzadarf(data=datos_dep,
                    vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                  "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                    listclass=c(""),
                    grupos=10,sinicio=1234,repe=20,
                    nodesize=85,replace=TRUE,sampsize=2500,ntree=130,mtry=13)
bagging3$modelo="bagging2500"

bagging4<-cruzadarf(data=datos_dep,
                    vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                  "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                    listclass=c(""),
                    grupos=10,sinicio=1234,repe=20,
                    nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=13)
bagging4$modelo="bagging3500"

bagging5<-cruzadarf(data=datos_dep,
                    vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                  "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                    listclass=c(""),
                    grupos=10,sinicio=1234,repe=20,
                    nodesize=85,replace=TRUE,sampsize=4500,ntree=130,mtry=13)
bagging5$modelo="bagging4500"

bagging6<-cruzadarf(data=datos_dep,
                    vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                  "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                    listclass=c(""),
                    grupos=10,sinicio=1234,repe=20,
                    nodesize=85,replace=TRUE,sampsize=5500,ntree=130,mtry=13)
bagging6$modelo="bagging5500"

bagging7<-cruzadarf(data=datos_dep,
                    vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                  "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                    listclass=c(""),
                    grupos=10,sinicio=1234,repe=20,
                    nodesize=85,replace=TRUE,ntree=130,mtry=13)
bagging7$modelo="baggingBASE"

union4<-rbind(bagging1,bagging2,bagging3,bagging4,bagging5,bagging6,bagging7)
par(cex.axis=0.5)
boxplot(data=union4,error~modelo)

union4$RMSE<-sqrt(union4$error)

par(cex.axis=1.2)
boxplot(data=union4,col="blue",RMSE~modelo)

# **********************************
# Random Forest
# **********************************

#Búsqueda del mtry optimo
set.seed(1234)
rfgrid<-expand.grid(mtry=c(2,3,4,5,6,7,8,9,10,11,12,13))

rf<- train(farmacia~.,
           data=datos_dep,
           method="rf",trControl=control,tuneGrid=rfgrid,
           ntree=130,sampsize=3500,nodesize=85,replace=TRUE,
           importance=TRUE)

rf


set.seed(1234)
rfbisrf<-randomForest(farmacia~.,
                      data=datos_dep,
                      mtry=9,ntree=1000,nodesize=85,sampsize=3500,replace=TRUE)

#Obtenemos el RMSE
rfbisrf$mse<-sqrt(rfbisrf$mse)
rfbisrf
plot(rfbisrf$mse)

randforest1<-cruzadarf(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=6)
randforest1$modelo="rf_m6"

randforest2<-cruzadarf(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=7)
randforest2$modelo="rf_m7"


randforest3<-cruzadarf(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=8)
randforest3$modelo="rf_m8"

randforest4<-cruzadarf(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=9)
randforest4$modelo="rf_m9"

randforest5<-cruzadarf(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=10)
randforest5$modelo="rf_m10"

randforest6<-cruzadarf(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=11)
randforest6$modelo="rf_m11"

randforest7<-cruzadarf(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=12)
randforest7$modelo="rf_m12"

randforest8<-cruzadarf(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=13)
randforest8$modelo="rf_m13"


union5<-rbind(randforest1,randforest2,randforest3,randforest4,randforest5,randforest6,randforest7,randforest8)
par(cex.axis=0.5)
boxplot(data=union5,error~modelo)

union5$RMSE<-sqrt(union5$error)

par(cex.axis=1.2)
boxplot(data=union5,col="blue",RMSE~modelo)

#Importancia de las variables

rfgrid_opt<-expand.grid(mtry=c(7))

rf_optimo<- train(farmacia~.,
                  data=datos_dep,
                  method="rf",trControl=control,tuneGrid=rfgrid_opt,
                  ntree=130,sampsize=3500,nodesize=85,replace=TRUE,
                  importance=TRUE)

rf_optimo

#Importancia de las variables

final<-rf_optimo$finalModel
tabla<-as.data.frame(importance(final))
tabla<-tabla[order(-tabla$IncNodePurity),]
tabla
par(cex=0.4)
barplot(tabla$IncNodePurity,names.arg=rownames(tabla), col = "red")


# **********************************
# GBM
# **********************************

#Búsqueda de los parámetros óptimos
set.seed(1234)
gbmgrid<-expand.grid(shrinkage=c(0.2,0.1,0.05,0.01,0.001,0.0001),
                     n.minobsinnode=c(50,85,100,135,170,200),
                     n.trees=c(100,250,500,1000),
                     interaction.depth=c(2))

gbm<- train(farmacia~.,data=datos_dep,
            method="gbm",trControl=control,tuneGrid=gbmgrid,
            distribution="gaussian", bag.fraction=1,verbose=FALSE)
gbm

plot(gbm)

#The final values used for the model were n.trees = 500, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 50.
set.seed(1234)
#Evaluacíon por early-stopping
gbmgrid_trees<-expand.grid(shrinkage=c(0.1),
                           n.minobsinnode=c(50),
                           n.trees=c(100,150,200,250,300,350,400,450,500,550,600,650,800,1000,1500,2000),
                           interaction.depth=c(2))

gbm_trees<- train(farmacia~.,data=datos_dep,
                  method="gbm",trControl=control,tuneGrid=gbmgrid_trees,
                  distribution="gaussian", bag.fraction=1,verbose=FALSE)

plot(gbm_trees)
gbm_trees

#Validación cruzada repetida y boxplot (RMSE<1.480)

GBM1<-cruzadagbm(data=datos_dep,
                 vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                               "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                 listclass=c(""),
                 grupos=10,sinicio=1234,repe=20,
                 n.minobsinnode=50,shrinkage=0.05,n.trees=1000,interaction.depth=2)
GBM1$modelo="gbm1"


GBM2<-cruzadagbm(data=datos_dep,
                 vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                               "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                 listclass=c(""),
                 grupos=10,sinicio=1234,repe=20,
                 n.minobsinnode=85,shrinkage=0.1,n.trees=500,interaction.depth=2)
GBM2$modelo="gbm2"


GBM3<-cruzadagbm(data=datos_dep,
                 vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                               "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                 listclass=c(""),
                 grupos=10,sinicio=1234,repe=20,
                 n.minobsinnode=85,shrinkage=0.05,n.trees=1000,interaction.depth=2)
GBM3$modelo="gbm3"


GBM4<-cruzadagbm(data=datos_dep,
                 vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                               "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                 listclass=c(""),
                 grupos=10,sinicio=1234,repe=20,
                 n.minobsinnode=50,shrinkage=0.1,n.trees=550,interaction.depth=2)
GBM4$modelo="gbm4"


GBM5<-cruzadagbm(data=datos_dep,
                 vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                               "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                 listclass=c(""),
                 grupos=10,sinicio=1234,repe=20,
                 n.minobsinnode=50,shrinkage=0.1,n.trees=600,interaction.depth=2)
GBM5$modelo="gbm5"


GBM6<-cruzadagbm(data=datos_dep,
                 vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                               "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                 listclass=c(""),
                 grupos=10,sinicio=1234,repe=20,
                 n.minobsinnode=50,shrinkage=0.1,n.trees=800,interaction.depth=2)
GBM6$modelo="gbm6"


union6<-rbind(GBM1,GBM2,GBM3,GBM4,GBM5,GBM6)
par(cex.axis=0.5)
boxplot(data=union6,error~modelo)

union6$RMSE<-sqrt(union6$error)

par(cex.axis=1.2)
boxplot(data=union6,col="blue",RMSE~modelo)

#Importancia de las variables

gbmgrid_opt<-expand.grid(shrinkage=c(0.1),
                         n.minobsinnode=c(50),
                         n.trees=c(1000),
                         interaction.depth=c(2))

gbm_opt<- train(farmacia~.,data=datos_dep,
                method="gbm",trControl=control,tuneGrid=gbmgrid_opt,
                distribution="gaussian", bag.fraction=1,verbose=FALSE)

#Importancia de las variables del modelo ganador
tabla_gb<-summary(gbm_opt)
par(cex=0.4)
barplot(tabla_gb$rel.inf,names.arg=row.names(tabla_gb), col = "red")

# **********************************
# Xgboost
# **********************************

set.seed(1234)
xgbmgrid<-expand.grid(
  min_child_weight=c(50,85,100,135,170,200),
  eta=c(0.1,0.05,0.03,0.01,0.001),
  nrounds=c(100,250,500,1000),
  max_depth=6,gamma=0,colsample_bytree=1,subsample=1)

xgbm<- train(farmacia~.,data=datos_dep,
             method="xgbTree",trControl=control,
             tuneGrid=xgbmgrid,verbose=FALSE)
xgbm

plot(xgbm)

#early-stopping
xgbmgrid_es<-expand.grid(eta=c(0.01),
                         min_child_weight=c(50),
                         nrounds=c(100,250,500,750,1000,2000,2500),
                         max_depth=6,gamma=0,colsample_bytree=1,subsample=1)

set.seed(1234)

xgbm_es<- train(farmacia~.,data=datos_dep,
                method="xgbTree",trControl=control,
                tuneGrid=xgbmgrid_es,verbose=FALSE)
xgbm_es
plot(xgbm_es)

xgb1<-cruzadaxgbm(data=datos_dep,
                  vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                  listclass=c(""),
                  grupos=10,sinicio=1234,repe=20,
                  min_child_weight=50,eta=0.01,nrounds=500,max_depth=6,
                  gamma=0,colsample_bytree=1,subsample=1)
xgb1$modelo="xgb1"

xgb2<-cruzadaxgbm(data=datos_dep,
                  vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                  listclass=c(""),
                  grupos=10,sinicio=1234,repe=20,
                  min_child_weight=85,eta=0.01,nrounds=1000,max_depth=6,
                  gamma=0,colsample_bytree=1,subsample=1)
xgb2$modelo="xgb2"

xgb3<-cruzadaxgbm(data=datos_dep,
                  vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                  listclass=c(""),
                  grupos=10,sinicio=1234,repe=20,
                  min_child_weight=85,eta=0.05,nrounds=250,max_depth=6,
                  gamma=0,colsample_bytree=1,subsample=1)
xgb3$modelo="xgb3"

xgb4<-cruzadaxgbm(data=datos_dep,
                  vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                  listclass=c(""),
                  grupos=10,sinicio=1234,repe=20,
                  min_child_weight=50,eta=0.05,nrounds=100,max_depth=6,
                  gamma=0,colsample_bytree=1,subsample=1)
xgb4$modelo="xgb4"

xgb5<-cruzadaxgbm(data=datos_dep,
                  vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                  listclass=c(""),
                  grupos=10,sinicio=1234,repe=20,
                  min_child_weight=35,eta=0.01,nrounds=500,max_depth=6,
                  gamma=0,colsample_bytree=1,subsample=1)
xgb5$modelo="xgb5"

xgb6<-cruzadaxgbm(data=datos_dep,
                  vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                  listclass=c(""),
                  grupos=10,sinicio=1234,repe=20,
                  min_child_weight=50,eta=0.01,nrounds=750,max_depth=6,
                  gamma=0,colsample_bytree=1,subsample=1)
xgb6$modelo="xgb6"

union7<-rbind(xgb1,xgb2,xgb3,xgb4,xgb5,xgb6)

union7$RMSE<-sqrt(union7$error)

par(cex.axis=1)
boxplot(data=union7,col="blue",RMSE~modelo)

#Evaluamos la importancia de las variables del modelo óptimo
xgbmgrid_opt<-expand.grid(eta=c(0.01),
                          min_child_weight=c(50),
                          nrounds=c(500),
                          max_depth=6,gamma=0,colsample_bytree=1,subsample=1)

set.seed(1234)

xgbm_opt<- train(farmacia~.,data=datos_dep,
                 method="xgbTree",trControl=control,
                 tuneGrid=xgbmgrid_opt,verbose=FALSE)


# IMPORTANCIA DE VARIABLES
varImp(xgbm_opt)
plot(varImp(xgbm_opt))

# **********************************
# SVM
# **********************************
#SVM Linear
#Se utiliza el conjunto depurado estandarizado

set.seed(1234)
SVMgrid_lin<-expand.grid(C=c(0.01,0.1,0.25,0.5,1,3,5,8,10,20,50))

SVM<- train(data=datos_est,farmacia~.,
            method="svmLinear",trControl=control,
            tuneGrid=SVMgrid_lin,verbose=FALSE)

SVM

resultados <- SVM[["results"]]
soluti <- SVM[["pred"]]

plot(SVM$results$C,SVM$results$RMSE)

#SVM Polinomial

set.seed(1234)
SVMgrid_pol<-expand.grid(C=c(0.1,1,3,5,10),
                         degree=c(2,3),scale=c(0.1,0.5,1,2,5))


SVM2<- train(data=datos_est,farmacia~.,
             method="svmPoly",trControl=control,
             tuneGrid=SVMgrid_pol,verbose=FALSE)

SVM2

SVM2$results

dat1<-as.data.frame(SVM2$results)
library(ggplot2)

# PLOT DE DOS VARIABLES CATEGÓRICAS, UNA CONTINUA
ggplot(dat1, aes(x=factor(C), y=RMSE, 
                 color=factor(degree),pch=factor(scale))) +
  geom_point(position=position_dodge(width=0.5),size=3)


#SVM RBF
SVMgrid_RBF<-expand.grid(C=c(0.01,0.2,0.5,1,3,5,10,20),
                         sigma=c(0.01,0.1,0.2,0.5,1,2,5,10,20))

SVM3<- train(data=datos_est,farmacia~.,
             method="svmRadial",trControl=control,
             tuneGrid=SVMgrid,verbose=FALSE)

SVM3

dat2<-as.data.frame(SVM3$results)

ggplot(dat2, aes(x=factor(C), y=RMSE, 
                 color=factor(sigma)))+ 
  geom_point(position=position_dodge(width=0.5),size=3)

#Validación cruzada y boxplot

SVM_lin1<-cruzadaSVM(data=datos_est,
                     vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                     listclass=c(""),
                     grupos=10,sinicio=1234,repe=20,C=1)

SVM_lin1$modelo="SVM_1"

SVM_lin2<-cruzadaSVM(data=datos_est,
                     vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                     listclass=c(""),
                     grupos=10,sinicio=1234,repe=20,C=5)

SVM_lin2$modelo="SVM_2"


SVM_lin3<-cruzadaSVM(data=datos_est,
                     vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                     listclass=c(""),
                     grupos=10,sinicio=1234,repe=5,C=8)

SVM_lin3$modelo="SVM_3"

SVM_pol1<-cruzadaSVMpoly(data=datos_est,
                         vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                         listclass=c(""),
                         grupos=10,sinicio=1234,repe=20,C=0.1,degree=3,scale=0.1)

SVM_pol1$modelo="SVM_4"

SVM_pol2<-cruzadaSVMpoly(data=datos_est,
                         vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                         listclass=c(""),
                         grupos=10,sinicio=1234,repe=20,C=0.1,degree=3,scale=1)

SVM_pol2$modelo="SVM_5"

SVM_pol3<-cruzadaSVMpoly(data=datos_est,
                         vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                         listclass=c(""),
                         grupos=10,sinicio=1234,repe=20,C=1,degree=3,scale=0.1)

SVM_pol3$modelo="SVM_6"

SVM_rbf1<-cruzadaSVMRBF(data=datos_est,
                        vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                        listclass=c(""),
                        grupos=10,sinicio=1234,repe=20,C=0.5,sigma=0.1)

SVM_rbf1$modelo="SVM_7"

SVM_rbf2<-cruzadaSVMRBF(data=datos_est,
                        vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                        listclass=c(""),
                        grupos=10,sinicio=1234,repe=20,C=0.2,sigma=0.1)

SVM_rbf2$modelo="SVM_8"

SVM_rbf3<-cruzadaSVMRBF(data=datos_est,
                        vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                        listclass=c(""),
                        grupos=10,sinicio=1234,repe=20,C=1,sigma=0.1)

SVM_rbf3$modelo="SVM_9"

union8<-rbind(SVM_lin1,SVM_lin2,SVM_lin3,SVM_pol1,SVM_pol2,SVM_pol3,SVM_rbf1,SVM_rbf2,SVM_rbf3)

par(cex.axis=0.5)
boxplot(data=union8,error~modelo)

union8$RMSE<-sqrt(union8$error)

par(cex.axis=1)
boxplot(data=union8,col="blue",RMSE~modelo)

# **********************************
# Modelos de ensamblado
# **********************************

source("cruzadas avnnet_ens")
source("cruzada lin_ens.R")
source("cruzada rf_ens continua.R")
source("cruzada gbm_ens continua.R")
source("cruzada xgboost_ens continua.R")
source("cruzada SVM_ens continua RBF.R")

#Regresion lineal
medias1<-cruzadalin_ens(data=datos_dep,
                        vardep="farmacia",listconti=
                          c("EDAD_pac","Tot_Lab", "Tot_Med","Tot_Enf","I_Complejidad","P_Asistencial","sexo_pac"),
                        listclass=c(""),grupos=10,sinicio=1234,repe=20)

medias1bis<-as.data.frame(medias1[1])
medias1bis$modelo<-"regresion"
predi1<-as.data.frame(medias1[2])
predi1$reg<-predi1$pred


#Árboles de regresión

medias2<-cruzadaarbol_ens(data=datos_dep,
                          vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                        "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                          listclass=c(""),grupos=10,sinicio=1234,repe=20,cp=0,minbucket=85)

medias2bis<-as.data.frame(medias2[1])
medias2bis$modelo<-"arbol"
predi2<-as.data.frame(medias2[2])
predi2$tree<-predi2$pred

#Redes neuronales
medias3<-cruzadaavnnet_ens(data=datos_est,
                           vardep="farmacia",listconti=
                             c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                           listclass=c(""),grupos=10,sinicio=1234,repe=20,itera=250,
                           size=c(10),decay=c(0.0001))

medias3bis<-as.data.frame(medias3[1])
medias3bis$modelo<-"avnnet"
predi3<-as.data.frame(medias3[2])
predi3$avnnet<-predi3$pred


#RANDOM FOREST
medias4<-cruzadarf_ens(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=7)


medias4bis<-as.data.frame(medias4[1])
medias4bis$modelo<-"rf"
predi4<-as.data.frame(medias4[2])
predi4$rf<-predi4$pred


#Gradient Boosting

medias5<-cruzadagbm_ens(data=datos_dep,
                        vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                      "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                        listclass=c(""),
                        grupos=10,sinicio=1234,repe=20,
                        n.minobsinnode=50,shrinkage=0.05,n.trees=1000,interaction.depth=2)

medias5bis<-as.data.frame(medias5[1])
medias5bis$modelo<-"gbm"
predi5<-as.data.frame(medias5[2])
predi5$gbm<-predi5$pred


# Extreme Gradient Boosting
medias6<-cruzadaxgbm_ens(data=datos_dep,
                         vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                       "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                         listclass=c(""),
                         grupos=10,sinicio=1234,repe=20,
                         min_child_weight=50,eta=0.01,nrounds=500,max_depth=6,
                         gamma=0,colsample_bytree=1,subsample=1)


medias6bis<-as.data.frame(medias6[1])
medias6bis$modelo<-"xgbm"
predi6<-as.data.frame(medias6[2])
predi6$xgbm<-predi6$pred


#SVM
medias7<-cruzadaSVMRBF_ens(data=datos_est,
                           vardep="farmacia",listconti=c("EDAD_pac","Tot_Med", "Tot_Enf","Tot_Lab","sexo_pac","I_Complejidad","P_Asistencial"),
                           listclass=c(""),
                           grupos=10,sinicio=1234,repe=20,C=0.5,sigma=0.1)

medias7bis<-as.data.frame(medias7[1])
medias7bis$modelo<-"svmRadial"
predi7<-as.data.frame(medias7[2])
predi7$svmRadial<-predi7$pred

#Bagging
medias8<-cruzadarf_ens(data=datos_dep,
                       vardep="farmacia",listconti=c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "registro","Tot_Enf","P_Asistencial",
                                                     "Tot_Lab", "Tot_Med","extranjero", "Sexo_med", "ESPECIALIDAD","sexo_pac"),
                       listclass=c(""),
                       grupos=10,sinicio=1234,repe=20,
                       nodesize=85,replace=TRUE,sampsize=3500,ntree=130,mtry=13)

medias8bis<-as.data.frame(medias8[1])
medias8bis$modelo<-"Bagging"
predi8<-as.data.frame(medias8[2])
predi8$bag<-predi8$pred

predi8$
  #Unimos las predicciones
  unipredi<-cbind(predi1,predi3,predi4,predi5,predi6,predi7,predi8)

# Eliminar columnas duplicadas
unipredi<- unipredi[, !duplicated(colnames(unipredi))]
dput(names(unipredi))

#Posibles combinaciones ensamblado
unipredi$predi1<-(unipredi$xgbm+unipredi$rf)/2
unipredi$predi2<-(unipredi$avnnet+unipredi$svmRadial)/2
unipredi$predi3<-(unipredi$xgbm+unipredi$gbm+unipredi$rf)/3
unipredi$predi4<-(unipredi$avnnet+unipredi$rf+unipredi$gbm)/3
unipredi$predi5<-(unipredi$avnnet+unipredi$gbm+unipredi$svmRadial)/3
unipredi$predi6<-(unipredi$gbm+unipredi$svmRadial+unipredi$rf)/3
unipredi$predi7<-(unipredi$xgbm+unipredi$svmRadial+unipredi$rf)/3
unipredi$predi8<-(unipredi$gbm+unipredi$svmRadial+unipredi$rf+unipredi$avnnet)/4
unipredi$predi9<-(unipredi$gbm+unipredi$svmRadial+unipredi$rf+unipredi$reg)/4
unipredi$predi10<-(unipredi$gbm+unipredi$xgbm+unipredi$rf+unipredi$avnnet)/4
unipredi$predi11<-(unipredi$xgbm+unipredi$avnnet+unipredi$rf+unipredi$gbm+unipredi$svmRadial)/5
unipredi$predi12<-(unipredi$xgbm+unipredi$reg+unipredi$rf+unipredi$gbm+unipredi$svmRadial)/5
unipredi$predi13<-(unipredi$xgbm+unipredi$reg+unipredi$rf+unipredi$gbm+unipredi$svmRadial+unipredi$avnnet)/6


# Recorto los modelos de la lista de variables
listado<-c("predi1","predi2","predi3","predi4","predi5","predi6","predi7","predi8","predi9","predi10","predi11","predi12","predi13")

repeticiones<-nlevels(factor(unipredi$Rep))
unipredi$Rep<-as.factor(unipredi$Rep)
unipredi$Rep<-as.numeric(unipredi$Rep)

# Calculo el MSE para cada repeticion de validacion cruzada

medias0<-data.frame(c())
for (prediccion in listado)
{
  paso <-unipredi[,c("obs",prediccion,"Rep")]
  paso$error<-(paso[,c(prediccion)]-paso$obs)^2
  paso<-paso %>%group_by(Rep)%>%summarize(error=mean(error))
  paso$modelo<-prediccion
  medias0<-rbind(medias0,paso)
}

# Boxplot

medias0$RMSE<-sqrt(medias0$error)
par(cex.axis=1)
boxplot(data=medias0,col="blue",RMSE~modelo)


# **********************************
# Comparación de modelos
# **********************************
#Unimos las predicciones

listado<-c("predi11")

medias0<-data.frame(c())
for (prediccion in listado)
{
  paso <-unipredi[,c("obs",prediccion,"Rep")]
  paso$error<-(paso[,c(prediccion)]-paso$obs)^2
  paso<-paso %>%group_by(Rep)%>%summarize(error=mean(error))
  paso$modelo<-prediccion
  medias0<-rbind(medias0,paso)
}

medias_final<-rbind(medias0,lineal3,red5,bagging4,randforest2,GBM1,xgb1,arbol2,SVM_rbf1)

#Equiparamos las dimensiones de los resultados
medias0<-medias0%>% select (-Rep)
lineal3<-lineal3%>% select (-Rep)
red5<-red5%>% select (-Rep)
bagging4<-bagging4%>% select (-Rep)
randforest2<-randforest2%>% select (-Rep)
GBM1<-GBM1%>% select (-Rep)
xgb1<-xgb1%>% select (-Rep)
arbol2<-arbol2%>% select (-Rep)

#Boxplot
medias_final$RMSE<-sqrt(medias_final$error)
par(cex.axis=1)
boxplot(data=medias_final,col="blue",RMSE~modelo)

#Ampliaos los 4 mejores modelos
medias_ampliado<-rbind(medias0,red5,randforest2,xgb1)
medias_ampliado$RMSE<-sqrt(medias_ampliado$error)
par(cex.axis=1)
boxplot(data=medias_ampliado,col="blue",RMSE~modelo)