# Training the model and examining performance on test set (RNN)
#!/usr/bin/env Rscript
library("optparse")
option_list = list(
  make_option(c("-d", "--debugging"), type="character", default="n",
              help="Debugging mode (only for local runs) [default= %default]", metavar="character"),
  make_option(c("-w", "--class_weights"), type="character", default="y",
              help="Include class weights [default= %default]", metavar="character"),
  make_option(c("-c", "--callbacks"), type="character", default="y",
              help="Include callabcks [default= %default]", metavar="character"),
  make_option(c("-m", "--model"), type="integer", default="1", 
              help="model ID [default= %default]", metavar="integer")#,
  # make_option(c("-b", "--batch"), type="integer", default="1", 
  #             help="100k batch ID for rerunning missing models [default= %default]", metavar="integer"),
  # make_option(c("-n", "--nburnin"), type="integer", default="1000000", 
  #             help="number of iterations taken as burnin [default= %default]", metavar="integer"),
  # make_option(c("-r", "--rerun"), type="character", default="n",
  #             help="Rerun missing models [default= %default]", metavar="character")
  
)
### Everything is set to incorporate step size tunning for sourceCpp if needed in the future 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

model_id = opt$model
include_cw = opt$class_weights == "y"
include_callbacks = opt$callbacks == "y"
debug_mode = opt$debugging == "y"
# rerun_model = opt$rerun == "y"

# Can this be deactivated if running in Vianey's server ?
Sys.setenv(UV_OFFLINE=1)

# candidate_models = readRDS("1d_structured_networks/rnn/missing_models_1_total.rds")  

# Hyperparameter Tuning for ResNet

## ---- Libraries ----
# library(rsample)
library(keras3)

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)
library(shapviz)
library(kernelshap)



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

x_test = readRDS("Data/x_test.rds")
dummy_y_test = readRDS("Data/dummy_y_test.rds")

test = readRDS("Data/test.rds")


resnet_class_weights = NULL
if(include_cw){
  ## Do this for 1:10 then 11:20 because it takes too long on its own
  cw<-2.15867
  resnet_class_weights = list("0"=1,"1"=cw)
}

# for restoring best weights
resnet_callbacks <- list(
  callback_early_stopping(
    # monitor = "val_loss",
    # min_delta = 1e-2,
    # patience = 30,
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
optimizers <- keras3::keras$optimizers

fit_cw = ""
if(include_cw){
  fit_cw = "_cw"
}

fit_callbacks = ""
if(include_callbacks){
  fit_callbacks = "_callback"
}  

# Validation metric fitted models
final_data = readRDS(paste0("1d_structured_networks/resnet/val_metrics",fit_cw,fit_callbacks,".rds"))

## ---- Grid Search ----
# create grid of parameter space we want to search
# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full = readRDS(file = "1d_structured_networks/resnet/grid.search.full.rds")
grid.search.full = as_tibble(grid.search.full) %>% mutate(model_id = row_number())

candidate_tunings = grid.search.full %>% 
  right_join(final_data) %>% 
  arrange(val_loss) %>% 
  filter(row_number() < 21)

mod_metrics = c()
mod_train_local_metrics = c()

## ---- Run Search ----

print(sprintf("Processing Model #%d", model_id))

best_param = candidate_tunings %>% filter(row_number() == opt$model)

t1 = Sys.time()

set_random_seed(15)

input_shape <- c(249,1)
set_random_seed(15)
inputs <- layer_input(shape = input_shape)

block_1_output <- inputs %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu)

# Adjust block_2_output to include the first convolutional layer for block_2
block_2_prep <- block_1_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization)

block_2_output <- block_2_prep %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_1_output)

# Introduce a skip from block_1_output to block_3_output
block_3_output <- block_2_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_2_output) %>%
  layer_add(block_1_output) # Adding block_1_output as a skip to block_3_output

# Continue from block_3_output to block_4_output
block_4_output <- block_3_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_3_output)

# Introduce a skip from block_3_output to block_5_output
block_5_output <- block_4_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_4_output) %>%
  layer_add(block_3_output) # Adding block_3_output as a skip to block_5_output

outputs <- block_5_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(2, activation="sigmoid")

model <- keras_model(inputs, outputs)

model %>% compile(
  optimizer = optimizer_adam(1e-4), # change here
  loss = loss_categorical_crossentropy,
  metrics = c("accuracy", tf$keras$metrics$AUC())
)

# Fit model
resnet_history <- model %>% fit(
  x_train, dummy_y_train,
  batch_size = best_param$batch_size,
  epochs = 100,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = resnet_class_weights,
  callbacks = resnet_callbacks,
  verbose = 2
)

# storing training metrics to double check fitting consistency between
# local and remote NN fitting during the exploration phase
mod_train_local_metrics[1] = min(resnet_history$metrics$val_loss)
mod_train_local_metrics[2] = which(resnet_history$metrics$val_loss==min(resnet_history$metrics$val_loss))
mod_train_local_metrics[3] = resnet_history$metrics$val_auc[mod_train_local_metrics[2]]
mod_train_local_metrics[4] = best_param$model_id


eval <- evaluate(model, (x_test), dummy_y_test)
preds<-predict(model, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
cm <- confusionMatrix(species.predictions,as.factor(test$species))
auc_value <- eval[grepl("auc", names(eval))]
# metrics <- c(eval[['loss']], cm$byClass['Sensitivity'], cm$byClass['Specificity'], auc_value)

mod_metrics = c(cm$byClass[c(1,2,11)], auc_value[[1]],best_param$model_id)
cm
# print(metrics)

# Names for training metrics during local implementation
# colnames(mod_train_local_metrics) = c("val_loss_local","best_epoch_local","val_auc_local","model_id")


# colnames(mod_metrics) = c("test_sensit","test_specif","test_bal_acc","test_auc","model_id")

# Exploring important predictors & model output
pred_fun <- function(mod, X) predict(mod, data.matrix(X), batch_size = 1e4, verbose = FALSE)
ks <- kernelshap(model, x_train[1000:2000,c(1:249)], bg_X = x_train[1:200,c(1:249)], pred_fun = pred_fun)

# Return to sequential plan when done
# plan(sequential)

shaps <- shapviz(ks)
sv_importance(shaps, show_numbers = TRUE)
sv_importance(shaps, kind = "beeswarm")
imp<-sv_importance(shaps, kind = "no")

frequencies<-rownames(imp)
frequencies<-as.numeric(gsub('F','',frequencies))
imp<-cbind.data.frame(imp,frequencies)

saveRDS(imp, file = paste0("1d_structured_networks/resnet/test/test_metrics",fit_cw,fit_callbacks,"/importance_shaps_",model_id,".rds"))

t2 = Sys.time()
t2 - t1


saveRDS(mod_metrics,file = paste0("1d_structured_networks/resnet/test/test_metrics",fit_cw,fit_callbacks,"/importance_shaps_metrics_output_",model_id,".rds"))
saveRDS(mod_train_local_metrics,file = paste0("1d_structured_networks/resnet/test/test_metrics",fit_cw,fit_callbacks,"/importance_shaps_dc_training_output_",model_id,".rds"))

print(paste0("include class_weights: ",include_cw))
print(paste0("include callback: ", include_callbacks))

print(paste0("test metrics for resnet with parameter configuration ",opt$model," ready!"))

