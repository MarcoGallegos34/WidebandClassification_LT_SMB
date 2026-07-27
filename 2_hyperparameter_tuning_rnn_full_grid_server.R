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
  # make_option(c("-b", "--batch"), type="integer", default="1", 
  #             help="100k batch ID for rerunning missing models [default= %default]", metavar="integer"),
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

# Can this be deactivated if running in Vianey's server ?
Sys.setenv(UV_OFFLINE=1)

## Changes model_id if rerun is set to true ##
if(rerun_model){
  
  missing_models = readRDS("1d_structured_networks/rnn/missing_models_1_total.rds")  
  
  new_model_id = missing_models[model_id]
  model_id = new_model_id
}

# Hyperparameter tuning (RNN)

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

## loading training and validation data ##
x_data_test = readRDS("Data/rnn/x_data_test.rds")
x_data_train = readRDS("Data/rnn/x_data_train.rds")
x_data_val = readRDS("Data/rnn/x_data_val.rds")
y_data_test = readRDS("Data/rnn/y_data_test.rds")
y_data_train = readRDS("Data/rnn/y_data_train.rds")
y_data_val = readRDS("Data/rnn/y_data_val.rds")
y_test = readRDS("Data/rnn/y_test.rds")
y_train = readRDS("Data/rnn/y_train.rds")
y_val = readRDS("Data/rnn/y_val.rds")

dummy_y_test = readRDS("Data/rnn/dummy_y_test.rds")
dummy_y_train = readRDS("Data/rnn/dummy_y_train.rds")
dummy_y_val = readRDS("Data/rnn/dummy_y_val.rds")


## ---- Set Up ----
rnn_class_weights = NULL
if(include_cw){
  ## Do this for 1:10 then 11:20 because it takes too long on its own
  # cw<-2.15867
  cw<-summary(as.factor(dummy_y_train[,1]))[2]/summary(as.factor(dummy_y_train[,1]))[1]
  rnn_class_weights = list("0"=1,"1"=cw)
}

# for restoring best weights
rnn_callbacks <- list(
  callback_early_stopping(
    # monitor = "val_loss",
    # min_delta = 1e-2,
    # patience = 25,
    restore_best_weights = TRUE
  )
)

rnn_callbacks = NULL
if(include_callbacks){
  # for implementing early stopping
  rnn_callbacks <- list(
    callback_early_stopping(
      monitor = "val_loss",
      min_delta = 1e-2,
      patience = 25,
      restore_best_weights = TRUE
    )
  )
}

# for using legacy optimizers which work better with newer Macs
optimizers <- keras3::keras$optimizers


# ## ---- Create grid of all the hyperparameters ----
# 
# ## create grid of parameter space we want to search
# regrate<-c(1e-6,1e-5,1e-4)
# lstmunits<-c(256,128,64)
# neuron1<-c(256,128,64,32,16)
# batchsize<-c(100, 500, 1000, 1500)
# 
# # expand the grid so that every possible combination of the above parameters is present. 
# # creating every possible combination to test
# grid.search.full<-expand.grid(regrate=regrate,lstmunits=lstmunits,neuron1=neuron1, batchsize = batchsize)

grid.search.full = readRDS("1d_structured_networks/rnn/grid.search.full.rds")

# # randomly select 20 of these models to fit. 
# set.seed(15)
# 
# x<-sample(1:nrow(grid.search.full),20,replace=F)
# grid.search.subset<-grid.search.full[x,]

# objects to store evaluation metrics
val_loss<-rep(NA,1)
best_epoch<-rep(NA,1)
val_auc<-rep(NA,1)

print(sprintf("Processing Model #%d", model_id))

t1 = Sys.time()

set_random_seed(15)
rnn = keras_model_sequential(input_shape=c(5,249)) # initialize model

# our input layer
rnn %>%
  layer_lstm(units = grid.search.full$lstmunits[model_id]) %>%
  layer_activation_leaky_relu()%>%
  layer_batch_normalization()%>%
  layer_dense(units = grid.search.full$neuron1[model_id],activity_regularizer = regularizer_l2(l=grid.search.full$regrate[model_id])) %>%
  layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = 'sigmoid')

rnn %>% compile(
  optimizer = optimizer_adam(1e-4),
  loss = loss_binary_crossentropy(),
  metrics = c("accuracy", tf$keras$metrics$AUC())
)

rnn_history <- rnn %>% fit(
  x_data_train, dummy_y_train,
  batch_size = grid.search.full$batchsize[model_id], 
  epochs = 200,
  validation_data = list(x_data_val,dummy_y_val),
  class_weight = rnn_class_weights,
  callbacks = rnn_callbacks)


t2 = Sys.time()
t2 - t1

val_loss[1]<-min(rnn_history$metrics$val_loss)
best_epoch[1]<-which(rnn_history$metrics$val_loss==min(rnn_history$metrics$val_loss))[1]
val_auc[1]<-max(rnn_history$metrics$val_auc)

final_output = c(val_loss,best_epoch,val_auc,model_id)

fit_cw = ""
if(include_cw){
  fit_cw = "_cw"
}

fit_callbacks = ""
if(include_callbacks){
  fit_callbacks = "_callback"
}  

nbatch = (model_id-1)%/%100000 + 1
saveRDS(final_output,file = paste0("1d_structured_networks/rnn/training_metrics",fit_cw,fit_callbacks,"_b",nbatch,"/training_output_",model_id,".rds"))

print(paste0("include class_weights: ",include_cw))
print(paste0("include class_weights: ", include_callbacks))

print(paste0("training metrics for rnn with parameter configuration ",model_id," ready!"))

