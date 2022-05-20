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
# Importaciób y exploración de los datos
# **********************************

#Matriz de correlaciones 
datos_cont <- read_csv("datos_cont.csv")
mat_corr <- round(datos_cont %>% cor(), 3)
mat_corr
corrplot(mat_corr, method = "number", bg = "grey50", number.cex = 0.5)


source("cruzadas avnnet y lin.R")
source("cruzada arbol continua.R")
source("cruzada rf continua.R")
source("cruzada gbm continua.R")
source("cruzada xgboost continua.R")
source("cruzada SVM continua lineal.R")
source("cruzada SVM continua polinomial.R")
source("cruzada SVM continua RBF.R")

# Validación cruzada una sola vez
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
# Selección de variables
# **********************************


# para BIC ponemos k=log(numoer de observaciones), en el stepAIC aproximadamente k=9.

full<-lm(farmacia~.,data=datos_dep)
null<-lm(farmacia~1,data=datos_dep)

selec1<-stepAIC(null,scope=list(upper=full),direction="both",trace=FALSE,k=9)
dput(names(selec1$coefficients))

#c("(Intercept)", "EDAD_pac", "Tot_Med", "Tot_Enf", 
#  "I_Complejidad")

#AIC

full<-glm(farmacia~.,data=datos_dep,family = gaussian(link = "identity"))
null<-glm(farmacia~1,data=datos_dep,family = gaussian(link = "identity"))

selec1<-stepAIC(null,scope=list(upper=full),
                direction="both",family = gaussian(link = "identity"),trace=FALSE)

vec<-(names(selec1[[1]]))

length(vec)

# 10variables-1

dput(vec)

#c("(Intercept)", "EDAD_pac", "Tot_Med", "Tot_Enf", 
#  "I_Complejidad", "Tot_Lab", "sexo_pac", "P_Asistencial", 
#  "Sexo_med", "RMD")

# BORUTA
# 

out.boruta <- Boruta(farmacia~., data = datos_dep)

sal<-data.frame(out.boruta$finalDecision)

sal2<-sal[which(sal$out.boruta.finalDecision=="Confirmed"),,drop=FALSE]
dput(row.names(sal2))

length(dput(row.names(sal2)))

#c("EDAD_pac", "Edad_med", "I_Complejidad", "RMD", "Tot_Enf", 
#  "P_Asistencial", "Tot_Lab", "Tot_Med", "Sexo_med", "ESPECIALIDAD", 
#  "sexo_pac")