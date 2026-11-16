# Training the model and examining performance on test set (CNN)
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

# Hyperparameter tuning (RNN)

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(keras3)
library(caret)

##
if(debug_mode){
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE) # change here
}

## ---- Set Up ----
## loading training and validation data ##
x_validate = readRDS("Data/x_validate.rds")
x_train = readRDS("Data/x_train.rds")
dummy_y_val = readRDS("Data/dummy_y_val.rds")
dummy_y_train = readRDS("Data/dummy_y_train.rds")

x_test = readRDS("Data/x_test.rds")
dummy_y_test = readRDS("Data/dummy_y_test.rds")

test = readRDS("Data/test.rds")

# Class weight calculation to account for imbalanced data
cnn_class_weights = NULL
if(include_cw){
  ## Do this for 1:10 then 11:20 because it takes too long on its own
  # cw<-2.15867
  cw<-summary(as.factor(dummy_y_train[,1]))[2]/summary(as.factor(dummy_y_train[,1]))[1]
  cnn_class_weights = list("0"=1,"1"=cw)
}

# for restoring best weights
cnn_callbacks <- list(
  callback_early_stopping(
    # monitor = "val_loss",
    # min_delta = 1e-2,
    # patience = 25,
    restore_best_weights = TRUE
  )
)

cnn_callbacks = NULL
if(include_callbacks){
  # for implementing early stopping
  callbacks <- list(
    callback_early_stopping(
      monitor = "val_loss",
      min_delta = 1e-2,
      patience = 45,
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
final_data = readRDS(paste0("1d_structured_networks/dnn/val_metrics",fit_cw,fit_callbacks,".rds"))

## ---- Grid Search ----
# create grid of parameter space we want to search
# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full = readRDS(file = "1d_structured_networks/dnn/grid.search.full.rds")
grid.search.full = as_tibble(grid.search.full) %>% mutate(model_id = row_number())

candidate_tunings = grid.search.full %>% 
  right_join(final_data) %>% 
  arrange(val_loss) %>% 
  filter(row_number() < 21) #%>% 
  # rename(dropout1 = droprate,
  #        dropout2 = droprate2,
  #        dropout3 = droprate3,
  #        dropout4 = droprate4,
  #        dropout5 = droprate5
  #        )

mod_metrics = c()
mod_train_local_metrics = c()

## ---- Run Search ----

print(sprintf("Processing Model #%d", model_id))

best_param = candidate_tunings %>% filter(row_number() == opt$model)

t1 = Sys.time()

set_random_seed(15)

# continue here
cnn = keras_model_sequential(input_shape=c(249,1))

#model
cnn %>% 
  layer_conv_1d(
    filters = best_param$filters1,
    kernel_size = best_param$kernel_size, 
    activation = 'relu',
    padding = 'same',
    strides = 1) %>%
  layer_dropout(rate = best_param$dropout1)%>%
  layer_batch_normalization()%>%
  layer_conv_1d(filters = best_param$filters2,
                kernel_size = best_param$kernel_size,  
                activation = 'relu',
                padding = 'same',
                strides = 1) %>%
  layer_dropout(rate = best_param$dropout2) %>%
  layer_batch_normalization() %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = best_param$filters3,
                kernel_size = best_param$kernel_size,  
                activation = 'relu',
                padding = 'same',
                strides = 1) %>%
  layer_dropout(rate = best_param$dropout3) %>%
  layer_batch_normalization() %>%
  layer_conv_1d(filters = best_param$filters4,
                kernel_size = best_param$kernel_size, 
                activation = 'relu',
                padding = 'same',
                strides = 1) %>%
  layer_dropout(rate = best_param$dropout4) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_batch_normalization() %>%
  layer_conv_1d(filters = best_param$filters5,
                kernel_size = best_param$kernel_size, 
                activation = 'relu',
                padding = 'same',
                strides = 1) %>%
  layer_dropout(rate = best_param$dropout5) %>%
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
  batch_size = best_param$batch_size,
  epochs = 200,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = cnn_class_weights,
  callbacks = cnn_callbacks
)


# storing training metrics to double check fitting consistency between
# local and remote NN fitting during the exploration phase
mod_train_local_metrics[1] = min(cnn_history$metrics$val_loss)
mod_train_local_metrics[2] = which(cnn_history$metrics$val_loss==min(cnn_history$metrics$val_loss))
mod_train_local_metrics[3] = max(cnn_history$metrics$val_auc)
mod_train_local_metrics[4] = best_param$model_id


eval <- evaluate(cnn, (x_test), dummy_y_test)
preds<-predict(cnn, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
cm <- confusionMatrix(species.predictions,as.factor(test$species))
auc_value <- eval[grepl("auc", names(eval))]
# metrics <- c(eval[['loss']], cm$byClass['Sensitivity'], cm$byClass['Specificity'], auc_value)

mod_metrics = c(cm$byClass[c(1,2,11)], auc_value[[1]])
cm
# print(metrics)
  
# Names for training metrics during local implementation
# colnames(mod_train_local_metrics) = c("val_loss_local","best_epoch_local","val_auc_local","model_id")


# colnames(mod_metrics) = c("test_sensit","test_specif","test_bal_acc","test_auc","model_id")

## ---- ROC Curve Generation ----
# 1. Create a dataframe with true labels and predicted probabilities for the target class
# Based on your which.max logic, preds[, 1] is "LT" and preds[, 2] is "SMB". 
# We will use the probabilities for "SMB" as our event of interest.
roc_data <- data.frame(
  truth = as.factor(test$species),
  prob_LT = preds[, 1], 
  prob_SMB = preds[, 2] 
)

saveRDS(roc_data, file = paste0("1d_structured_networks/cnn/test/test_metrics",fit_cw,fit_callbacks,"/roc_data_",model_id,".rds"))

t2 = Sys.time()
t2 - t1


saveRDS(mod_metrics,file = paste0("1d_structured_networks/cnn/test/test_metrics",fit_cw,fit_callbacks,"/metrics_output_",model_id,".rds"))
saveRDS(mod_train_local_metrics,file = paste0("1d_structured_networks/cnn/test/test_metrics",fit_cw,fit_callbacks,"/dc_training_output_",model_id,".rds"))

print(paste0("include class_weights: ",include_cw))
print(paste0("include callback: ", include_callbacks))

print(paste0("test metrics for cnn with parameter configuration ",opt$model," ready!"))
