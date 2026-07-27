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
              help="model ID [default= %default]", metavar="integer")#,
  # make_option(c("-n", "--nburnin"), type="integer", default="1000000", 
  #             help="number of iterations taken as burnin [default= %default]", metavar="integer"),
  # make_option(c("-s", "--stimulli"), type="character", default="n", 
  #             help="Include sound stimulli in transition probabilities [default= %default]", metavar="character")
  
)
### Everything is set to incorporate step size tunning for sourceCpp if needed in the future 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

model_id = opt$model
include_cw = opt$class_weights == "y"
include_callbacks = opt$callbacks == "y"
debug_mode = opt$debugging == "y"
# Hyperparameter tuning of thee and four hidden layer DNN

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

if(include_cw){
  ## Do this for 1:10 then 11:20 because it takes too long on its own
  cw<-summary(as.factor(dummy_y_train[,1]))[2]/summary(as.factor(dummy_y_train[,1]))[1]
}

if(include_callbacks){
  # for implementing early stopping
  callbacks <- list(
    callback_early_stopping(
      monitor = "val_loss",
      min_delta = 1e-2,
      patience = 15,
      restore_best_weights = TRUE
    )
  )
}

# for using legacy optimizers which work better with newer Macs
optimizers <- keras3::keras$optimizers # change here

## ---- Create grid of all the hyperparameters for 4 hidden layers ----
regrate<-c(1e-6,1e-5,1e-4)
dropout<-c(0,0.1,0.15,0.2)
neuron1<-c(128,96,64)
neuron2<-c(64,48,32)
neuron3<-c(32,24,16)
neuron4<-c(16,8,4)

grid.search.full<-expand.grid(regrate=regrate,dropout=dropout,neuron1=neuron1,neuron2=neuron2,neuron3=neuron3,neuron4=neuron4)

## ---- Take 20 models to search through (4 hidden layers) ----
# set.seed(15)
# x<-sample(1:nrow(grid.search.full),20,replace=F)
# x<-1:nrow(grid.search.full)
# grid.search.subset<-grid.search.full[x,]

## ---- Cross validation of 4 hidden layer models ----
# # Create vectors to store validation loss and best epoch in
# val_loss_4layer<-rep(NA,nrow(grid.search.subset))
# best_epoch_4layer<-rep(NA,nrow(grid.search.subset))
# val_auc_4layer<-rep(NA,nrow(grid.search.subset))

# Create vectors to store validation loss and best epoch in
val_loss_4layer<-rep(NA,1)
best_epoch_4layer<-rep(NA,1)
val_auc_4layer<-rep(NA,1)

print(sprintf("Processing Model #%d", model_id))
set_random_seed(15)
model1 <- keras_model_sequential(input_shape = c(249)) # change here
model1 %>%
  layer_dense(units = grid.search.full$neuron1[model_id], activity_regularizer = regularizer_l2(l=grid.search.full$regrate[model_id])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(grid.search.full$dropout[model_id])%>%
  layer_dense(units = grid.search.full$neuron2[model_id],activity_regularizer = regularizer_l2(l=grid.search.full$regrate[model_id])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(grid.search.full$dropout[model_id])%>%
  layer_dense(units = grid.search.full$neuron3[model_id],activity_regularizer = regularizer_l2(l=grid.search.full$regrate[model_id])) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(grid.search.full$dropout[model_id])%>%
  layer_dense(units = grid.search.full$neuron4[model_id],activity_regularizer = regularizer_l2(l=grid.search.full$regrate[model_id])) %>%
  layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = "sigmoid")

model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer =  optimizer_adam(3e-4), # change here
  metrics = c("accuracy", tf$keras$metrics$AUC()))

history <- model1 %>% fit(
  x_train, dummy_y_train,
  batch_size = 500,
  epochs = 125,
  validation_data = list(x_validate,dummy_y_val)#,
  # class_weight = list("0"=1,"1"=cw),
  # callbacks = callbacks
)

val_loss_4layer[1]<-min(history$metrics$val_loss)
best_epoch_4layer[1]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
val_auc_4layer[1]<-history$metrics$val_auc[best_epoch_4layer[1]]

final_output = c(val_loss_4layer,best_epoch_4layer,val_auc_4layer,model_id)

saveRDS(final_output,file = paste0("1d_structured_networks/dnn/training_metrics/training_output_",model_id,".rds"))

print(paste0("training metrics for model with parameter configuration ",model_id," ready!"))

# ## Do this for 1:10 then 11:20 because it takes too long on its own
# t1 <- Sys.time()
# for(i in 100:300){
#   print(sprintf("Processing Model #%d", i))
#     set_random_seed(15)
#     model1 <- keras_model_sequential(input_shape = c(249)) # change here
#     model1 %>%
#       layer_dense(units = grid.search.full$neuron1[i], activity_regularizer = regularizer_l2(l=grid.search.full$regrate[i])) %>%
#       layer_activation_leaky_relu()%>%
#       layer_dropout(grid.search.full$dropout[i])%>%
#       layer_dense(units = grid.search.full$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.full$regrate[i])) %>%
#       layer_activation_leaky_relu()%>%
#       layer_dropout(grid.search.full$dropout[i])%>%
#       layer_dense(units = grid.search.full$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.full$regrate[i])) %>%
#       layer_activation_leaky_relu()%>%
#       layer_dropout(grid.search.full$dropout[i])%>%
#       layer_dense(units = grid.search.full$neuron4[i],activity_regularizer = regularizer_l2(l=grid.search.full$regrate[i])) %>%
#       layer_activation_leaky_relu()%>%
#       layer_dense(units = 2, activation = "sigmoid")
# 
#     model1 %>% compile(
#       loss = 'binary_crossentropy',
#       optimizer =  optimizer_adam(3e-4), # change here
#       metrics = c("accuracy", tf$keras$metrics$AUC()))
# 
#     history <- model1 %>% fit(
#       x_train, dummy_y_train,
#       batch_size = 500,
#       epochs = 125,
#       validation_data = list(x_validate,dummy_y_val)#,
#       # class_weight = list("0"=1,"1"=cw),
#       # callbacks = callbacks
#       )
# 
#     val_loss_4layer[i]<-min(history$metrics$val_loss)
#     best_epoch_4layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
#     val_auc_4layer[i]<-history$metrics$val_auc[best_epoch_4layer[i]]
# 
#   }
# t2 <- Sys.time()
# t2 - t1

# t1 <- Sys.time()
# for(i in 301:450){
#   print(sprintf("Processing Model #%d", i))
#   set_random_seed(15)
#   model1 <- keras_model_sequential(input_shape = c(249)) # change here
#   model1 %>%
#     layer_dense(units = grid.search.subset$neuron1[i], activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron4[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dense(units = 2, activation = "sigmoid")
# 
#   model1 %>% compile(
#     loss = 'binary_crossentropy',
#     optimizer =  optimizer_adam(3e-4), # change here
#     metrics = c("accuracy", tf$keras$metrics$AUC()))
# 
#   history <- model1 %>% fit(
#     x_train, dummy_y_train,
#     batch_size = 500,
#     epochs = 125,
#     validation_data = list(x_validate,dummy_y_val)#,
#     # class_weight = list("0"=1,"1"=cw),
#     # callbacks = callbacks
#     )
# 
#   val_loss_4layer[i]<-min(history$metrics$val_loss)
#   best_epoch_4layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
#   val_auc_4layer[i]<-history$metrics$val_auc[best_epoch_4layer[i]]
# 
# }
# t2 <- Sys.time()
# t2 - t1
# 
# t1 <- Sys.time()
# for(i in 451:600){
#   print(sprintf("Processing Model #%d", i))
#   set_random_seed(15)
#   model1 <- keras_model_sequential(input_shape = c(249)) # change here
#   model1 %>%
#     layer_dense(units = grid.search.subset$neuron1[i], activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron4[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dense(units = 2, activation = "sigmoid")
# 
#   model1 %>% compile(
#     loss = 'binary_crossentropy',
#     optimizer =  optimizer_adam(3e-4), # change here
#     metrics = c("accuracy", tf$keras$metrics$AUC()))
# 
#   history <- model1 %>% fit(
#     x_train, dummy_y_train,
#     batch_size = 500,
#     epochs = 125,
#     validation_data = list(x_validate,dummy_y_val)#,
#     # class_weight = list("0"=1,"1"=cw),
#     # callbacks = callbacks
#     )
# 
#   val_loss_4layer[i]<-min(history$metrics$val_loss)
#   best_epoch_4layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
#   val_auc_4layer[i]<-history$metrics$val_auc[best_epoch_4layer[i]]
# 
# }
# t2 <- Sys.time()
# t2 - t1
# 
# t1 <- Sys.time()
# for(i in 601:873){
#   print(sprintf("Processing Model #%d", i))
#   set_random_seed(15)
#   model1 <- keras_model_sequential(input_shape = c(249)) # change here
#   model1 %>%
#     layer_dense(units = grid.search.subset$neuron1[i], activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron4[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dense(units = 2, activation = "sigmoid")
# 
#   model1 %>% compile(
#     loss = 'binary_crossentropy',
#     optimizer =  optimizer_adam(3e-4), # change here
#     metrics = c("accuracy", tf$keras$metrics$AUC()))
# 
#   history <- model1 %>% fit(
#     x_train, dummy_y_train,
#     batch_size = 500,
#     epochs = 125,
#     validation_data = list(x_validate,dummy_y_val)#,
#     # class_weight = list("0"=1,"1"=cw),
#     # callbacks = callbacks
#     )
# 
#   val_loss_4layer[i]<-min(history$metrics$val_loss)
#   best_epoch_4layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
#   val_auc_4layer[i]<-history$metrics$val_auc[best_epoch_4layer[i]]
# 
# }
# t2 <- Sys.time()
# t2 - t1
# 
# t1 <- Sys.time()
# for(i in 874:nrow(grid.search.subset)){
#   print(sprintf("Processing Model #%d", i))
#   set_random_seed(15)
#   model1 <- keras_model_sequential(input_shape = c(249)) # change here
#   model1 %>%
#     layer_dense(units = grid.search.subset$neuron1[i], activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dropout(grid.search.subset$dropout[i])%>%
#     layer_dense(units = grid.search.subset$neuron4[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
#     layer_activation_leaky_relu()%>%
#     layer_dense(units = 2, activation = "sigmoid")
# 
#   model1 %>% compile(
#     loss = 'binary_crossentropy',
#     optimizer =  optimizer_adam(3e-4), # change here
#     metrics = c("accuracy", tf$keras$metrics$AUC()))
# 
#   history <- model1 %>% fit(
#     x_train, dummy_y_train,
#     batch_size = 500,
#     epochs = 125,
#     validation_data = list(x_validate,dummy_y_val)#,
#     # class_weight = list("0"=1,"1"=cw),
#     # callbacks = callbacks
#     )
# 
#   val_loss_4layer[i]<-min(history$metrics$val_loss)
#   best_epoch_4layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
#   val_auc_4layer[i]<-history$metrics$val_auc[best_epoch_4layer[i]]
# 
# }
# t2 <- Sys.time()
# t2 - t1

# saveRDS(val_loss_4layer,file = "1D Structured Networks/Deep Neural Network/val_loss_4layer_allCombs.rds")
# saveRDS(best_epoch_4layer,file = "1D Structured Networks/Deep Neural Network/best_epoch_4layer_allCombs.rds")
# saveRDS(val_auc_4layer,file = "1D Structured Networks/Deep Neural Network/val_auc_4layer_allCombs.rds")

