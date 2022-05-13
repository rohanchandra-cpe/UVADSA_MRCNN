# UVADSA_MRCNN
UVA DSA MRCNN Implementation for the JIGSAW dataset

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

Now you can begin training with the MRCNN in mrcnn_implementation.py

Test Message
