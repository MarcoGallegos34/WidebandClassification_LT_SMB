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

  missing_models = readRDS("1d_structured_networks/resnet/missing_models_1_total.rds")  
    
  new_model_id = missing_models[model_id]
  model_id = new_model_id
}

# Hyperparameter Tuning for ResNet

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
# library(caret)
library(keras3) # change here


##
if(debug_mode){
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE) # change here
}


## ---- Set up ----
# functions for adding layers conditionally
conv_activation_layer <- function(input_layer, filters, kernel_size, leaky_relu) {
  if (!leaky_relu) {
    # conv layer with ReLU activation
    output_layer <- input_layer %>% 
      layer_conv_1d(filters = filters, kernel_size = kernel_size, 
                    activation = 'relu', padding = 'same', strides = 1)
  } else {
    # conv layer followed by Leaky ReLU
    output_layer <- input_layer %>% 
      layer_conv_1d(filters = filters, kernel_size = kernel_size, 
                    padding = 'same', strides = 1) %>%
      layer_activation_leaky_relu()
  }
  return(output_layer)
}

add_batch_normalization <- function(input_layer, batch_normalization) {
  if (batch_normalization) {
    output_layer <- input_layer %>% layer_batch_normalization()
  } else {
    output_layer <- input_layer
  }
  return(output_layer)
}

## loading training and validation data ##
x_validate = readRDS("Data/x_validate.rds")
x_train = readRDS("Data/x_train.rds")
dummy_y_val = readRDS("Data/dummy_y_val.rds")
dummy_y_train = readRDS("Data/dummy_y_train.rds")

resnet_class_weights = NULL
if(include_cw){
  ## Do this for 1:10 then 11:20 because it takes too long on its own
  cw<-2.15867
  resnet_class_weights = list("0"=1,"1"=cw)
}

# for restoring best weights
resnet_callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-2,
    patience = 30,
    restore_best_weights = TRUE
  )
)


resnet_callbacks = NULL
if(include_callbacks){
  # for implementing early stopping
  resnet_callbacks <- list(
    callback_early_stopping(
      monitor = "val_loss",
      min_delta = 1e-2,
      patience = 30,
      restore_best_weights = TRUE
    )
  )
}

# for using legacy optimizers which work better with newer Macs
optimizers <- keras3::keras$optimizers # change here

# ## ---- Grid Search ----
# # create grid of parameter space we want to search
# filters <- c(16, 32, 64)
# kernel_size <- c(3, 5, 7)
# leaky_relu <- c(T, F)
# batch_normalization <- c(T, F)
# batch_size <- c(500, 800, 1000, 1200, 1500)
# 
# # expand the grid so that every possible combination of the above parameters is present. 
# # creating every possible combination to test
# grid.search.full<-expand.grid(filters = filters, kernel_size = kernel_size, 
#                               leaky_relu = leaky_relu, 
#                               batch_normalization = batch_normalization,
#                               batch_size = batch_size)

grid.search.full = readRDS("1d_structured_networks/resnet/grid.search.full.rds")

# set.seed(15)
# n_subset <- 20
# x<-sample(1:nrow(grid.search.full), n_subset,replace=F)
# grid.search.subset<-grid.search.full[x,]

# Create vectors to store validation loss and best epoch in
val_loss<-rep(NA,1)
best_epoch_loss<-rep(NA,1)
val_auc<-rep(NA,1)

print(sprintf("Processing Model #%d", model_id))

t1 = Sys.time()

input_shape <- c(249,1)
set_random_seed(15)
inputs <- layer_input(shape = input_shape)

block_1_output <- inputs %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id], 
                        kernel_size = grid.search.full$kernel_size[model_id], 
                        leaky_relu = grid.search.full$leaky_relu[model_id]) # self-defined function with optional leaky relu (or relu)

# Adjust block_2_output to include the first convolutional layer for block_2
block_2_prep <- block_1_output %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id], 
                        kernel_size = grid.search.full$kernel_size[model_id], 
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  add_batch_normalization(batch_normalization = grid.search.full$batch_normalization[model_id])

block_2_output <- block_2_prep %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id], 
                        kernel_size = grid.search.full$kernel_size[model_id], 
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  add_batch_normalization(batch_normalization = grid.search.full$batch_normalization[model_id]) %>%
  layer_add(block_1_output)

# Introduce a skip from block_1_output to block_3_output
block_3_output <- block_2_output %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id],
                        kernel_size = grid.search.full$kernel_size[model_id],
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id],
                        kernel_size = grid.search.full$kernel_size[model_id],
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  add_batch_normalization(batch_normalization = grid.search.full$batch_normalization[model_id]) %>%
  layer_add(block_2_output) %>%
  layer_add(block_1_output) # Adding block_1_output as a skip to block_3_output

# Continue from block_3_output to block_4_output
block_4_output <- block_3_output %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id],
                        kernel_size = grid.search.full$kernel_size[model_id],
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id],
                        kernel_size = grid.search.full$kernel_size[model_id],
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  add_batch_normalization(batch_normalization = grid.search.full$batch_normalization[model_id]) %>%
  layer_add(block_3_output)

# Introduce a skip from block_3_output to block_5_output
block_5_output <- block_4_output %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id],
                        kernel_size = grid.search.full$kernel_size[model_id],
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id],
                        kernel_size = grid.search.full$kernel_size[model_id],
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  add_batch_normalization(batch_normalization = grid.search.full$batch_normalization[model_id]) %>%
  layer_add(block_4_output) %>%
  layer_add(block_3_output) # Adding block_3_output as a skip to block_5_output

outputs <- block_5_output %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id], 
                        kernel_size = grid.search.full$kernel_size[model_id], 
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  conv_activation_layer(filters = grid.search.full$filters[model_id], 
                        kernel_size = grid.search.full$kernel_size[model_id], 
                        leaky_relu = grid.search.full$leaky_relu[model_id]) %>%
  add_batch_normalization(batch_normalization = grid.search.full$batch_normalization[model_id]) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(2, activation="sigmoid")

model <- keras_model(inputs, outputs)

model %>% compile(
  optimizer = optimizer_adam(1e-4), # change here
  loss = loss_categorical_crossentropy,
  metrics = c("accuracy", tf$keras$metrics$AUC())
)

# Fit model (just resnet)
resnet_history <- model %>% fit(
  x_train, dummy_y_train,
  batch_size = grid.search.full$batch_size[model_id],
  epochs = 100,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = resnet_class_weights,
  callbacks = resnet_callbacks
)


t2 = Sys.time()
t2 - t1

val_loss[1]<-min(resnet_history$metrics$val_loss)
best_epoch_loss[1]<-which(resnet_history$metrics$val_loss==min(resnet_history$metrics$val_loss))
val_auc[1]<-resnet_history$metrics$val_auc[best_epoch_loss[1]]

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
saveRDS(final_output,file = paste0("1d_structured_networks/resnet/training_metrics",fit_cw,fit_callbacks,"_b",nbatch,"/training_output_",model_id,".rds"))

print(paste0("include class_weights: ",include_cw))
print(paste0("include class_weights: ", include_callbacks))

print(paste0("training metrics for resnet with parameter configuration ",model_id," ready!"))

