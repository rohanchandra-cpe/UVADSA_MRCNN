# UVADSA_MRCNN
This is a project of the UVA Dependable Systems and Analytics Lab, aimed at training a [Mask-RCNN](https://arxiv.org/abs/1703.06870) model to perform object detection and image segmentation on images from the popular [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) dataset. 

This repository includes
* A requirements.txt file containing all the packages needed to run this code
* mrcnn_implementation.py, which contains the Mask RCNN model
* file_formatter.py, used to transform raw Cogito data into data usable by Mask RCNN
* train_small, test_small, val_small, which each contain 10 images and can be used to determine if the Mask RCNN if functioning properly
* train, test, and validation, which contain 40%, 40%, and 20% of the full dataset respectively
   * These percentages can be changed by running file_formatter.py
* Knot_Tying_Images_Unprocessed and Knot_Tying_JSON_Unprocessed, containing the raw data from Cogito
* images/ and json/ which contain the unsorted, renamed data from Knot_Tying_Images_Unprocessed and Knot_Tying_JSON_Unprocessed

# Using file-formatter.py
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

# Using mrcnn-implementation.py
*  Clone the [Mask-RCNN repository](https://github.com/matterport/Mask_RCNN)
   * Run setup.py
   ```bash
   py setup.py install
   ```
*  Install dependencies
```bash
pip install -r requirements.txt
```
* Train the model starting with the COCO weights
```bash
python mrcnn_implementation.py train --train_dataset=./train_small --val_dataset=./val_small --weights=coco  
```
* Train the model starting with custom weights
```bash
python mrcnn_implementation.py train --train_dataset=./train_small --val_dataset=./val_small --weights=./path/to/weights.h5 
```
* Generate visualizations on a random image
```bash
python mrcnn_implementation.py visualize --val_dataset=./val_small --weights=coco --image=random 
```
* Generate visualizations with a custom image
```bash
python mrcnn_implementation.py visualize --val_dataset=./val_small --weights=coco --image=./path/to/image.png
```
