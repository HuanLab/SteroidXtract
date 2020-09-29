# This script is to perform data augmentation (spectra cutoff & Gaussian noise addition) and mass binning on MS2 spectra.
# Shipei Xing, Sep 29, 2020
# @Copyright: The University of British Columbia

##############################
directory <- 'E:/SteroidXtract_2020/steroid_lib'  # set the working directory
db.name <- 'LC-MSMS_steroid_library.msp'          # database name (MSP format)

##############################
# parameter setting
bin <- 0.1                    # bin width, 0.1 m/z by default
start_mz <- 50                # starting m/z value
end_mz <- 500                 # ending m/z value
mz.tol <- 0.01                # m/z tolerance, 0.01 m/z by default
topNo <- 20                   # the maxumum number of fragments reserved, default 20

##############################
# load libraries
library(metaMS)

##############################
# read database
db <- read.msp(db.name)

# data augmentation and mass binning
X.df <- data.frame(matrix(ncol = (end_mz-start_mz)/bin + 1)) 
p <- 1
for(i in 1:length(db)){
  ms2 <- data.frame(db[[i]]$pspectrum)
  premass <- db[[i]]$PrecursorMZ
  ms2 <- ms2[ms2[,2] >= 1,]
  ms2 <- ms2[ms2[,1] <= (premass + mz.tol),]
  ms2 <- ms2[ms2[,1] >= start_mz,] 
  ms2 <- ms2[ms2[,1] < end_mz,]
  
  if(nrow(ms2) > topNo){ ms2 <- ms2[ms2[,2] >= sort(ms2[,2],decreasing = TRUE)[topNo],]}
  # intensity normalization
  ms2[,2] <- ms2[,2]*100/max(ms2[,2])
  ms2[,2] <- sqrt(ms2[,2])
  if(nrow(ms2)==0) next
  
  ##########################
  # data augmentation: spectra cutoff
  for(x in 0:5){
    No <- ceiling(nrow(ms2)*(1-0.1*x))  # different levels of noises, 0% -> 50%
    ms2.x <- ms2[ ms2[,2] >= sort(ms2[,2],decreasing = TRUE)[No],]
    if(No == nrow(ms2)){  ms2.x <- ms2  }
    
    for(j in 1:nrow(ms2.x)){
      binNo <- floor((ms2.x[j,1]-start_mz)/bin) + 1
      X.df[p,binNo] <- max(ms2.x[j,2],X.df[p,binNo])# in case two fragments are in the same bin, use the max int
    }
    p <- p + 1
  }
}

############################
# data augmentation: Gaussian noise addition
X <- matrix(0,ncol = (end_mz-start_mz)/bin, nrow = nrow(X.df))
for(i in 1:nrow(X)){X[i,] <- abs(rnorm(ncol(X), mean = 0.5, sd = 0.1))}
X.df.copy <- X.df
X.df.copy[,1:(ncol(X.df.copy)-1)] <- X.df.copy[,1:(ncol(X.df.copy)-1)] + X
X.df <- cbind(X.df, X.df.copy)

############################
# labels for MS2 spectra in X.df
X.df[,ncol(X.df)] <- 'steroid'

# output X.df
write.csv(X.df, file = 'data_matrix.csv',row.names = F)

