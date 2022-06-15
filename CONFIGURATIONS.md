# Configuration details
Config files contain the model hyperparameters. Here we describe every parameter along with its default value.

Our config is organized as follows:

* `gpus` (List[int, ...]) - ids of gpus to run training on, default=`[0, 1]` 
  

* `data`:  
  * `root_path` (str) - root directory for MegaDepth data
  * `train_list_path` (str) - path to the file with train split, ex.: megadepth_train_2.0.txt
  * `val_list_path` (str) - path to the file with validation split, ex.: megadepth_valid_2.0.txt
  * `test_list_path` (str) - path to the file with test split. Note: OpenGlue's testing framework consists of calling the functions in/running inference.py, on 2 images at a time, and is not set up with scripts to do this on a data sample in an automated way. This config option appears to be non functional in this way, and out of the box from their repo, all instances of it are populated with the same path as the val_list_path, so, I'd reccommend setting it to the same value as the val_list_path until code is written to utilize it for testing and analysis in a more automated manner.
  * `batch_size_per_gpu` (int) -  number of image pairs in a batch per gpu, default=`4` Note: lowering this value seems to help with reducing memory expenditure, which seems to have an impact in terms of whether or not the run can complete without crashing the machine on certian devices. I have set it to 2 in the config files saved in this repository
  * `dataloader_workers_per_gpu` (int) - number of workers per one gpu, default=`4` Note: lowering this value seems to help with reducing memory expenditure, which seems to have an impact in terms of whether or not the run can complete without crashing the machine on certian devices.
  * `target_size` (List[int, int]) - shape of input image after resizing, default=`[ 960, 720 ]` Note: Make sure that this size is greater than or equal to all of the images you intend to use the model on in the inference step. Inference will fail if the images inputted have dimensions greater than the target size the model was trained on. If your images are too large for the model, you will need to crop the images to a smaller size or retrain the model with a larger target size.
  * `val_max_pairs_per_scene` (int) - max number of image pairs retrieved from the same scene, default=`50` Note: reducing this value helps to reduce training time.
    
  * `features_dir` --optional-- [ONLY if features were CACHED] (str) - path to the saved cached features directory.
  * `max_keypoints` --optional-- [ONLY if features were CACHED] (int) - max number of keypoints to detect, default=`1024` Note: reducing this value helps to reduce both runtime and memory expenditure.

* `logging`:
    * `root_path` (str) - directory, where experiment's logs will be saved. IMPORTANT: Logs include experiments containing model trained model weights as well as generated configuration files, both necessary for inference. The experiment path will be root_path/experiment_name, wheere experiment_name is generated from the option 'name', the settings with which the model was trained, and the timestamp of when training began.
    * `name` (str) - experiment name
    * `val_frequency` (int) - number of iterations for frequency of computing validation, default=`10000`
    * `train_logs_steps` (int) - number of iterations for frquency of logging results on train, default=`50`

  
* `train`: 
    * `epochs` (int) - The number of passes through the training set that the model preforms, lowering this greately speeds up training
    * `steps_per_epoch` - The number of batches to use in an epoch. Lowering this value greately speeds up training time at the cost of accuracy/effectiveness of the model
    * `grad_clip` (float) - clip gradients' global norm to <= threshold, default=`10.0` 
    * `precision` (int) - mixed precision configuration, combines the use of both 32 and 16 bit floating points, default=`32`
    * `gt_positive_threshold` (float) - threshold value for ignoring match, default=`3.`
    * `gt_negative_threshold` (float) - threshold value for an unmatched match, default=`5.`
    * `margin` (float) - margin for the criterion, default=`null`
    * `nll_weight` (float) - weight for the proportion of NLL loss, default=`1.0`
    * `metric_weight` (float) - weight for the proportion of metric loss, default=`0.0`
    * `lr` (float) - starting learning rate, default=`1.0e-4`
    * `scheduler_gamma` (float) - value used to decay lr, default=`0.999994`
  
    * `use_cached_features` --optional-- [ONLY if features were CACHED] (bool) - flag that enables training with cached features

* `evaluation`: 
    * `epipolar_dist_threshold` (float) - threshold for epipolar distance metric, default=`5.0e-4` 
    * `camera_auc_thresholds` (List[float,...]) - thresholds for area under the curve metric, pose error in degrees, default=`[5.0, 10.0, 20.0]`  
    * `camera_auc_ransac_inliers_threshold`(float) - sampson error, default=`2.0` 


* `inference`:
  * `match_threshold` (float) - threshold for a match, default=`0.2`
    
    
* `superglue`: 
    * `descriptor_dim` (int) - dimensionality of descriptors, default=`128` Note: make sure this is set prior to running inference and after training. This should correspond with the same field in the features extractor config file that was used for training, which will be saved in the experiment folder
    * `laf_to_sideinfo_method` (str) - ability to include geometry info from detector for each keypoint in positional encoding, options: `[none, rotation, scale, scale_rotation, affine]`, default= `none`
    * `positional_encoding`: 
        * `hidden_layers_sizes` (List[int, ...]) - input shape for hidden layers in MLP net for positional encoding, default=`[32, 64, 128]`
        * `output_size` (int) - dimensionality of returned output, in most cases should correspond descriptor_dim, default=`128` Note: Make sure this is set prior to running inference after training
    * `attention_gnn`:
        * `num_stages` (int) - number of attention stages (layers), 1 stage = SELF-attn + CROSS-attn, default=`12` 
        * `num_heads` (int) - number of attention heads, default=`4` 
        * `embed_dim` (int) - corresponds to descriptor_dim, default=`128` Note: Make sure this is set prior to running inference after training
        * `attention` (str) - method for attention, options: `[linear, softmax]`, default=`'linear'` 
        * `use_offset` (bool) - flag for usage of offset attention https://arxiv.org/abs/2012.09688, default=`False`
    * `dustbin_score_init` (float) - dustbin score, default=`1.0` 
    * `otp`:
        * `num_iters`(int) - number of iterations for differentiable Optimal Transport solver (Sinkhorn matrix scaling algorithm), default=`20`
        * `reg` (float) - regularization value for Sinkhorn, default=`1.0`
    * `residual` (bool) - flag for enabling combining local descriptor with context-aware descriptor, default=`True` 
    
### This part is set in seperate config files from `config/features` and `config/features_online`
For each feature extractor, options: `[OPENCV_SIFT, SuperPointNet, SuperPointNetBn, OPENCVDoGAffNetHardNet]`, default=`'OPENCV_SIFT'`, this section varies, so please look in yaml files for more details.

Example of the general setup for SuperPoint case:
* `name` (str) - method name for descriptor
* `max_keypoints` (int) - maximum number of keypoints, default=`1024` Note: if you are using pre-extraction, changing this value will affect the runtime of preextraction, and it is reccommended that this value be consistent with the corresponding value used in the training step. The corresponding value used in training with cached options should be less than or equal to this value in the feature extractor configuration used for pre-extraction.
* `descriptor_dim` (int) - dimensionality of descriptors 
* `nms_kernel` (int) - size of the kernel for non-maximum suppression convolution, default=`3`
* `remove_borders_size` (int) - the number of border-neighboring pixels to skip for keypoint detection, default=`4`
* `keypoint_threshold` (float) - threshold of score confidence for keypoint to be considered, default=`0.0`
* `weights` (str) - path to the weights, option for pretrained SuperPoint weights Note: I have included in the repository the weight files for the 3 superpoint models referenced by the config files included in config/features, and modiied this value to point to them with a relative path. If you add different weights files for feature extraction, move them, or are creating a configuration for a new feature extractor, you may need to change this value to reflect that.
