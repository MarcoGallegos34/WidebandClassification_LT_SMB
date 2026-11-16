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

# Hyperparameter tuning (RNN)

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(keras3)
library(caret)
library(shapviz)
library(kernelshap)
library(future)

##
if(debug_mode){
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE) # change here
}


## ---- Set up ----
## loading training and validation data ##

x_data_val = readRDS("Data/rnn/x_data_val.rds")
x_data_train = readRDS("Data/rnn/x_data_train.rds")
dummy_y_val = readRDS("Data/rnn/dummy_y_val.rds")
dummy_y_train = readRDS("Data/rnn/dummy_y_train.rds")

x_data_test = readRDS("Data/rnn/x_data_test.rds")
y_data_test = readRDS("Data/rnn/y_data_test.rds")

dummy_y_test = readRDS("Data/rnn/dummy_y_test.rds")

# Class weight calculation to account for imbalanced data
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

fit_cw = ""
if(include_cw){
  fit_cw = "_cw"
}

fit_callbacks = ""
if(include_callbacks){
  fit_callbacks = "_callback"
}  

# Validation metric fitted models
final_data = readRDS(paste0("1d_structured_networks/rnn/val_metrics",fit_cw,fit_callbacks,".rds"))

## ---- Grid Search ----
# create grid of parameter space we want to search
# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full = readRDS(file = "1d_structured_networks/rnn/grid.search.full.rds")
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

set_random_seed(15)
rnn = keras_model_sequential() # initialize model
# our input layer
rnn %>%
  layer_lstm(input_shape=c(5,249),units = best_param$lstmunits) %>%
  layer_activation_leaky_relu()%>%
  layer_batch_normalization()%>%
  layer_dense(units = best_param$neuron1,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  layer_activation_leaky_relu()%>%
  # layer_dropout(rate = best_param$droprate)%>%
  # layer_dense(units = best_param$neuron2,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  # layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = 'sigmoid')

rnn %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adam(3e-4),
  metrics = c('accuracy', tf$keras$metrics$AUC()))

rnn_history <- rnn %>% fit(
  x_data_train, dummy_y_train,
  batch_size = best_param$batchsize, 
  epochs = 200,
  validation_data = list(x_data_val,dummy_y_val),
  class_weight = rnn_class_weights,
  callbacks = rnn_callbacks,
  verbose = 2)
    
# storing training metrics to double check fitting consistency between
# local and remote NN fitting during the exploration phase
mod_train_local_metrics[1] = min(rnn_history$metrics$val_loss)
mod_train_local_metrics[2] = which(rnn_history$metrics$val_loss==min(rnn_history$metrics$val_loss))[1]
mod_train_local_metrics[3] = max(rnn_history$metrics$val_auc)
mod_train_local_metrics[4] = best_param$model_id


## evaluating the best model on test data ##
## this was carried by me, not Jess or the students
eval <- evaluate(rnn, (x_data_test), dummy_y_test)
preds<-predict(rnn, x=x_data_test)


species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
cm <- confusionMatrix(species.predictions,as.factor(y_data_test)) # this is the variable that fits in length
auc_value <- eval[grepl("auc", names(eval))]
# metrics <- c(eval[['loss']], cm$byClass['Sensitivity'], cm$byClass['Specificity'], auc_value)

mod_metrics = c(cm$byClass[c(1,2,11)], auc_value[[1]],best_param$model_id)
cm
# print(metrics)

# Names for training metrics during local implementation
# colnames(mod_train_local_metrics) = c("val_loss_local","best_epoch_local","val_auc_local","model_id")

  
# colnames(mod_metrics) = c("test_sensit","test_specif","test_bal_acc","test_auc","model_id")

# ---- SHAP Setup for 3D RNN Input ----

n_timesteps <- dim(x_data_train)[2] # 5
n_features  <- dim(x_data_train)[3] # 249

# 1. Subset 3D slices
x_explain_3d <- x_data_train[1000:1050, , ]
bg_x_3d      <- x_data_train[1:10, , ]

# 2. Flatten 3D arrays [N, 5, 249] into 2D matrices [N, 1245]
x_explain_2d <- matrix(x_explain_3d, nrow = dim(x_explain_3d)[1], ncol = n_timesteps * n_features)
bg_x_2d      <- matrix(bg_x_3d, nrow = dim(bg_x_3d)[1], ncol = n_timesteps * n_features)

# Assign feature names (e.g., F1_t1 ... F249_t5)
col_names <- unlist(lapply(1:n_features, function(f) paste0("F", f, "_t", 1:n_timesteps)))
colnames(x_explain_2d) <- col_names
colnames(bg_x_2d)      <- col_names

# 3. Prediction function: converts 2D back to 3D tensor inside TensorFlow
pred_fun <- function(mod, X) {
  X_mat <- as.matrix(X)
  # Reconstruct 3D array [batch_size, 5, 249]
  X_3d <- array(X_mat, dim = c(nrow(X_mat), n_timesteps, n_features))
  
  # Direct TensorFlow tensor call (bypasses high-level predict wrapper overhead)
  tf_X <- tf$convert_to_tensor(X_3d, dtype = "float32")
  preds <- mod(tf_X, training = FALSE)
  return(as.matrix(preds))
}


ks <- kernelshap(
  rnn, 
  x_explain_2d, 
  bg_X = bg_x_2d, 
  pred_fun = pred_fun, 
  parallel = FALSE,
  hybrid_degree = 0
)

shaps <- shapviz(ks)
sv_importance(shaps, show_numbers = TRUE)
sv_importance(shaps, kind = "beeswarm")
imp<-sv_importance(shaps, kind = "no")

# saving the shap values prior to obtaining the E(|SHAP|) (needed to compute the frequencies)
saveRDS(shaps, file = paste0("1d_structured_networks/rnn/test/test_metrics",fit_cw,fit_callbacks,"/object_importance_shaps_",model_id,".rds"))
saveRDS(shaps$S, file = paste0("1d_structured_networks/rnn/test/test_metrics",fit_cw,fit_callbacks,"/raw_importance_shaps_",model_id,".rds"))

frequencies<-rownames(imp)
frequencies<-as.numeric(gsub('F','',frequencies))
imp<-cbind.data.frame(imp,frequencies)

saveRDS(imp, file = paste0("1d_structured_networks/rnn/test/test_metrics",fit_cw,fit_callbacks,"/importance_shaps_",model_id,".rds"))


# saveRDS(mod_metrics,file = paste0("1d_structured_networks/rnn/test/test_metrics",fit_cw,fit_callbacks,"/metrics_output_",model_id,".rds"))
# saveRDS(mod_train_local_metrics,file = paste0("1d_structured_networks/rnn/test/test_metrics",fit_cw,fit_callbacks,"/dc_training_output_",model_id,".rds"))

print(paste0("include class_weights: ",include_cw))
print(paste0("include callback: ", include_callbacks))

print(paste0("test metrics for rnn with parameter configuration ",opt$model," ready!"))

