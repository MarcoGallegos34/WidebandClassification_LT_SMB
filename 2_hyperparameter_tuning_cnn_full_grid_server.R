#!/usr/bin/env Rscript
library("optparse")
option_list = list(
  # make_option(c("-o", "--order"), type="integer", default=0, 
  #             help="Variable for ordering constraints [default= %default]", metavar="integer"),
  # make_option(c("-e", "--emission"), type="integer", default=0, 
  #             help="Emission stream about to explored [default= %default]", metavar="integer"),
  # make_option(c("-t", "--ntemp"), type="integer", default=10, 
  #             help="Number of temperatures [default= %default]", metavar="integer"),
  make_option(c("-d", "--debugging"), type="character", default="n",
              help="Debugging mode (only for local runs) [default= %default]", metavar="character"),
  make_option(c("-w", "--class_weights"), type="character", default="y",
              help="Include class weights [default= %default]", metavar="character"),
  make_option(c("-c", "--callbacks"), type="character", default="y",
              help="Include callabcks [default= %default]", metavar="character"),
  make_option(c("-m", "--model"), type="integer", default="1", 
              help="model ID [default= %default]", metavar="integer"),
  make_option(c("-b", "--batch"), type="integer", default="1", 
              help="100k batch ID for rerunning missing models [default= %default]", metavar="integer"),
  # make_option(c("-n", "--nburnin"), type="integer", default="1000000", 
  #             help="number of iterations taken as burnin [default= %default]", metavar="integer"),
  make_option(c("-r", "--rerun"), type="character", default="n",
              help="Rerun missing models [default= %default]", metavar="character")
  
)
### Everything is set to incorporate step size tunning for sourceCpp if needed in the future 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

model_id = opt$model
include_cw = opt$class_weights == "y"
include_callbacks = opt$callbacks == "y"
debug_mode = opt$debugging == "y"
rerun_model = opt$rerun == "y"

Sys.setenv(UV_OFFLINE=1)
## Changes model_id if rerun is set to true ##

if(rerun_model){
  if(opt$batch == 1){
    
    missing_models = readRDS("1d_structured_networks/cnn/missing_models_1_100000.rds")  
    
  } else if(opt$batch == 2){
    
    missing_models = readRDS("1d_structured_networks/cnn/missing_models_100001_200000.rds")  
    
  } else if(opt$batch == 3){
    
    missing_models = readRDS("1d_structured_networks/cnn/missing_models_200001_300000.rds")  
    
  } else if(opt$batch == 4){
    
    missing_models = readRDS("1d_structured_networks/cnn/missing_models_300001_total.rds")  
    
  }
  new_model_id = missing_models[model_id]
  model_id = new_model_id
}

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(keras3) # change here

##
if(debug_mode){
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE) # change here
}


## Setting working directory ##
# setwd("../../")

## loading training and validation data ##
x_validate = readRDS("Data/x_validate.rds")
x_train = readRDS("Data/x_train.rds")
dummy_y_val = readRDS("Data/dummy_y_val.rds")
dummy_y_train = readRDS("Data/dummy_y_train.rds")

cnn_class_weights = NULL
if(include_cw){
  ## Do this for 1:10 then 11:20 because it takes too long on its own
  cw<-2.15867
  cnn_class_weights = list("0"=1,"1"=cw)
}

cnn_callbacks = NULL
if(include_callbacks){
  # for implementing early stopping
  cnn_callbacks <- list(
    callback_early_stopping(
      monitor = "val_loss",
      min_delta = 1e-2,
      patience = 50#,
      # restore_best_weights = TRUE
    )
  )
}

# for using legacy optimizers which work better with newer Macs
optimizers <- keras3::keras$optimizers # change here

# ## ---- Grid Search ----
# # create grid of parameter space we want to search
# filters1 <- c(8, 16, 32, 64)
# filters2 <- c(8, 16, 32, 64)
# filters3 <- c(8, 16, 32, 64)
# filters4 <- c(8, 16, 32, 64)
# filters5 <- c(8, 16, 32, 64)
# kernel_size <- c(3, 5, 7)
# batch_size <- c(100, 500, 1000, 1500)
# droprate1=c(0,0.1) 
# droprate2=c(0,0.1) 
# droprate3=c(0,0.1) 
# droprate4=c(0,0.1) 
# droprate5=c(0,0.1) 
# 
# # expand the grid so that every possible combination of the above parameters is present. 
# # creating every possible combination to test
# grid.search.full<-expand.grid(filters1 = filters1, filters2 = filters2, filters3 = filters3, filters4 = filters4, filters5 = filters5,kernel_size = kernel_size,
#                               batch_size = batch_size, droprate=droprate1,droprate2=droprate2,
#                               droprate3=droprate3, droprate4=droprate4, droprate5=droprate5)

grid.search.full = readRDS("1d_structured_networks/cnn/grid.search.full.rds")

## ---- Take 20 models to search through (4 hidden layers) ----
# set.seed(15)
# x<-sample(1:nrow(grid.search.full),20,replace=F)
# x<-1:nrow(grid.search.full)
# grid.search.subset<-grid.search.full[x,]

# Create vectors to store validation loss and best epoch in
val_loss<-rep(NA,1)
best_epoch_loss<-rep(NA,1)
val_auc<-rep(NA,1)

print(sprintf("Processing Model #%d", model_id))

set_random_seed(15)

t1 = Sys.time()
cnn = keras_model_sequential(input_shape=c(249,1)) # change here
#model
cnn %>% 
  layer_conv_1d(
    filters = grid.search.full$filters1[model_id],
    kernel_size = grid.search.full$kernel_size[model_id], 
    activation = 'relu',
    padding = 'same',
    strides = 1) %>%
  layer_dropout(rate = grid.search.full$droprate[model_id])%>%
  layer_batch_normalization()%>%
  layer_conv_1d(filters = grid.search.full$filters2[model_id],
                kernel_size = grid.search.full$kernel_size[model_id],  
                activation = 'relu',
                padding = 'same',
                strides = 1) %>%
  layer_dropout(rate = grid.search.full$droprate2[model_id]) %>%
  layer_batch_normalization() %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = grid.search.full$filters3[model_id],
                kernel_size = grid.search.full$kernel_size[model_id],  
                activation = 'relu',
                padding = 'same',
                strides = 1) %>%
  layer_dropout(rate = grid.search.full$droprate3[model_id]) %>%
  layer_batch_normalization() %>%
  layer_conv_1d(filters = grid.search.full$filters4[model_id],
                kernel_size = grid.search.full$kernel_size[model_id], 
                activation = 'relu',
                padding = 'same',
                strides = 1) %>%
  layer_dropout(rate = grid.search.full$droprate4[model_id]) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_batch_normalization() %>%
  layer_conv_1d(filters = grid.search.full$filters5[model_id],
                kernel_size = grid.search.full$kernel_size[model_id], 
                activation = 'relu',
                padding = 'same',
                strides = 1) %>%
  layer_dropout(rate = grid.search.full$droprate5[model_id]) %>%
  layer_batch_normalization() %>%
  layer_flatten() %>%
  layer_dense(units = 2, activation="sigmoid")

cnn %>% compile(
  optimizer = optimizer_adam(1e-4), # change here
  loss = loss_binary_crossentropy(),
  metrics = c("accuracy", tf$keras$metrics$AUC())
)


# Fit model
cnn_history <- cnn %>% fit(
  x_train, dummy_y_train,
  batch_size = grid.search.full$batch_size[model_id],
  epochs = 200,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = cnn_class_weights,
  callbacks = cnn_callbacks
)

t2 = Sys.time()
t2 - t1

# print("--------")


val_loss[1]<-min(cnn_history$metrics$val_loss)
best_epoch_loss[1]<-which(cnn_history$metrics$val_loss==min(cnn_history$metrics$val_loss))
val_auc[1] <- cnn_history$metrics$val_auc[best_epoch_loss[1]]

final_output = c(val_loss,best_epoch_loss,val_auc,model_id)

fit_cw = ""
if(include_cw){
  fit_cw = "_cw"
}

fit_callbacks = ""
if(include_callbacks){
  fit_callbacks = "_callback"
}  

nbatch = (model_id-1)%/%100000 + 1
saveRDS(final_output,file = paste0("1d_structured_networks/cnn/training_metrics",fit_cw,fit_callbacks,"_b",nbatch,"/training_output_",model_id,".rds"))

print(paste0("include class_weights: ",include_cw))
print(paste0("include callback: ", include_callbacks))

print(paste0("training metrics for cnn with parameter configuration ",model_id," ready!"))


