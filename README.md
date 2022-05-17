# UVADSA_MRCNN
UVA DSA MRCNN Implementation for the RAVEN dataset

Using file-formatter.py
* Call rename_files and pass in the following arguments
   * The location of the folder containing your images or JSON files to be renamed
   * The destination of the renamed files
   * True if the path contains images and False if it contains JSON files
* Call train-test-split with the following arguments
  * Percentage of your data for the training set
  * Percentage of your data for the test set
  * Percentage of your data for the validation set
  * Path of your renamed images (made in rename-files)

* Once you have called these methods you should now have the following folders
  * ./train/images
  * ./train/json
  * ./test/images
  * ./test/json
  * ./validation/images
  * ./validation/json
* Which will be used by the MRCNN for training

Using mrcnn-implementation.py
* First call train() (make sure train() is uncommented and visualize_images is commented)
* Comment train() out and uncomment visualize_images()
* Check the logs folder and you should see a folder containing the most recent weights 
   * Change SURGERY_WEIGHTS_PATH in visualize_images to contain the path to "mask_rcnn_maskrcnn_config_0002.h5"
   * e.g.  "./logs/maskrcnn_config20220516T2057/mask_rcnn_maskrcnn_config_0002.h5"
* Call visualize_images() and you should have 2 figures, the first one contains the MRCNN's predictions on an image from the validation set 
