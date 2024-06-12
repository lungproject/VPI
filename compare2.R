rm(list=ls(all=TRUE))
library(Hmisc); library(grid); library(lattice);library(Formula); library(ggplot2)
library(rms)
library(pROC)
library(rmda)
library(data.table)
library(survival)
library(survminer)
setwd("E:/Code/PleuralInvasion/R")
library(DTComPair)
pttrain<-read.csv(file="training.csv",header = TRUE);
pttrain[which(pttrain$operationtype==3),]$operationtype=2

pttrain[which(pttrain$operationtype==1),]$operationtype="Lob"
pttrain[which(pttrain$operationtype==2),]$operationtype="Sub"

pttrain$Nodule.type[which(pttrain$Nodule.type<=2)]=0
pttrain$Nodule.type[which(pttrain$Nodule.type>=3)]=1

pttrain[which(pttrain$Nodule.type==1),]$Nodule.type="solid"
pttrain[which(pttrain$Nodule.type==0),]$Nodule.type="non-solid"

ptall<-rbind(pttrain,ptval,pttest)

model2 <- decision_curve(label~DLS, data = pttrain ,family = binomial(link = 'logit'),thresholds = seq(0, 1, by = .01),study.design = 'cohort',confidence.intervals = FALSE) #number of bootstraps should be higher

plot_decision_curve( model2,
                     curve.names = c( "VPI-Net"),
                     cost.benefit.axis = FALSE, standardize = TRUE,legend.position = "bottomleft",
                     xlab ='Threshold probability',ylab ='Standardized Net Benefit') #adjust the legend position

ind1 = which(ptall$RFST>=24|(ptall$RFST<24&ptall$RFS==1))
ptall1 = ptall[ind1,]

fit<- survfit(Surv(ptall1$RFST, ptall1$RFS) ~ ptall1$combinegroup, data = ptall1)
ggsurvplot(fit, pval = TRUE,risk.table = TRUE,legend.labs = c("VPIscoreL", "VPIscoreH"),
           xlab="Time since surgery", ylab="Relapse-free survival")

fit<- survfit(Surv(ptall1$TTRT, ptall1$TTR) ~ ptall1$combinegroup, data = ptall1)
ggsurvplot(fit, pval = TRUE,risk.table = TRUE,legend.labs = c("VPIscoreL", "VPIscoreH"),
           xlab="Time since surgery", ylab="Time to Progression")
fit<- survfit(Surv(ptall1$OST, ptall1$OS) ~ ptall1$combinegroup, data = ptall1)
ggsurvplot(fit, pval = TRUE,risk.table = TRUE,legend.labs = c("VPIscoreL", "VPIscoreH"),
           xlab="Time since surgery", ylab="Overall survival")


f<- coxph(Surv(ptall$OST, ptall$OS)~ptall$combinegroup,data=ptall)
summary(f)
f<- coxph(Surv(ptall$RFST, ptall$RFS)~ptall$combinegroup,data=ptall)
summary(f)
f<- coxph(Surv(ptall$TTRT, ptall$TTR)~ptall$combinegroup,data=ptall)
summary(f)



ypttrain<-pttrain$label


tab1 <- tab.paired(ypttrain,pttrain$combinegroup,pttrain$readerAgroup)
# print.acc.1test(temp)
sesp.mcnemar(tab1)
pv.rpv(tab1)
roc1 <- roc(pttrain$label,pttrain$DLS)
roc2 <- roc(pttrain$label,pttrain$combine)

roc.test(roc1, roc2,method="delong")
