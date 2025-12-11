Data Acquisition & Preprocessing Summary
Member 1 Task
By: Kareem Hamed

As the Data Acquisition & Preprocessing Lead, I prepared all datasets required for model training. My work included collecting the datasets, organizing them, preprocessing all images, and delivering the final structured data for the training team.

1. Dataset Acquisition

I downloaded the LFW dataset and the VGGFace2 dataset (train and test sets) from Kaggle.
The datasets were organized clearly inside the project to prepare them for preprocessing.

2. Preprocessing Pipeline
Face Detection and Cleaning

I applied face detection using MTCNN on all images.
Images without faces or with detection errors were removed to keep only valid samples.

Face Cropping and Alignment

Each detected face was cropped and aligned to standardize pose, orientation, and position across the dataset.

Image Resizing

All aligned face images were resized to a unified resolution to ensure consistency for model training.

Train/Test Separation

The VGGFace2 identities were separated into training and testing subsets after preprocessing to maintain the original dataset structure.

3. Data Augmentation

To increase dataset diversity and reduce overfitting, I applied augmentation using several transformations including rotation, flipping, brightness/contrast adjustments, noise, and scaling.
Augmentation was applied to the prepared data to generate multiple additional training samples.

4. Output Structure

The preprocessing pipeline produced:

Cleaned and aligned face images

Clearly separated train and test sets

Augmented training samples

The final dataset is organized, consistent, and ready for model training and evaluation.

5. Summary of Completed Work

Downloaded and organized all required datasets

Performed face detection, alignment, cropping, and resizing

Removed unusable images

Separated training and testing identities

Applied advanced data augmentation

Delivered all processed datasets and preprocessing scripts to the team

Completion Status

All data acquisition and preprocessing tasks have been fully completed.
The dataset is now ready for model training, evaluation, and further project development.

Important Note
                  ALL the data is on this drive link

https://drive.google.com/drive/folders/17pN9OrDXG-s7PUlv3tDM7k0DJxOenD35?usp=drive_link
