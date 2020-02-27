# xview2 1st place solution
1st place solution for "xView2: Assess Building Damage" challenge. https://www.xview2.org

# Introduction to Solution

Solution developed using this environment:
 - Python 3 (based on Anaconda installation)
 - Pytorch 1.1.0+ and torchvision 0.3.0+ 
 - Nvidia apex https://github.com/NVIDIA/apex
 - https://github.com/skvark/opencv-python
 - https://github.com/aleju/imgaug


Hardware:
Current training batch size requires at least 2 GPUs with 12GB each. (Initially trained on Titan V GPUs). For 1 GPU batch size and learning rate should be found in practice and changed accordingly.

"train", "tier3" and "test" folders from competition dataset should be placed to the current folder.

Use "train.sh" script to train all the models. (~7 days on 2 GPUs).
To generate predictions/submission file use "predict.sh".
"evalution-docker-container" folder contains code for docker container used for final evalution on hold out set (CPU version).

# Trained models
Trained model weights available here: https://vdurnov.s3.amazonaws.com/xview2_1st_weights.zip

(Please Note: the code was developed during the competition and designed to perform separate experiments on different models. So, published as is without additional refactoring to provide fully training reproducibility).


# Data Cleaning Techniques

Dataset for this competition well prepared and I have not found any problems with it.
Training masks generated using json files, "un-classified" type treated as "no-damage" (create_masks.py). "masks" folders will be created in "train" and "tier3" folders.

The problem with different nadirs and small shifts between "pre" and "post" images solved on models level:
 - Frist, localization models trained using only "pre" images to ignore this additional noise from "post" images. Simple UNet-like segmentation Encoder-Decoder Neural Network architectures used here.
 - Then, already pretrained localization models converted to classification Siamese Neural Network. So, "pre" and "post" images shared common weights from localization model and the features from the last Decoder layer concatenated to predict damage level for each pixel. This allowed Neural Network to look at "pre" and "post" separately in the same way and helped to ignore these shifts and different nadirs as well.
 - Morphological dilation with 5*5 kernel applied to classification masks. Dilated masks made predictions more "bold" - this improved accuracy on borders and also helped with shifts and nadirs.


# Data Processing Techniques

Models trained on different crops sizes from (448, 448) for heavy encoder to (736, 736) for light encoder.
Augmentations used for training:
 - Flip (often)
 - Rotation (often)
 - Scale (often)
 - Color shifts (rare)
 - Clahe / Blur / Noise (rare)
 - Saturation / Brightness / Contrast (rare)
 - ElasticTransformation (rare)

Inference goes on full image size (1024, 1024) with 4 simple test-time augmentations (original, filp left-right, flip up-down, rotation to 180).


# Details on Modeling Tools and Techniques

All models trained with Train/Validation random split 90%/10% with fixed seeds (3 folds). Only checkpoints from epoches with best validation score used.

For localization models 4 different pretrained encoders used:
from torchvision.models:
 - ResNet34
from https://github.com/Cadene/pretrained-models.pytorch:
 - se_resnext50_32x4d
 - SeNet154
 - Dpn92

Localization models trained on "pre" images, "post" images used in very rare cases as additional augmentation.

Localization training parameters:
Loss: Dice + Focal
Validation metric: Dice
Optimizer: AdamW

Classification models initilized using weights from corresponding localization model and fold number. They are Siamese Neural Networks with whole localization model shared between "pre" and "post" input images. Features from last Decoder layer combined together for classification. Pretrained weights are not frozen.
Using pretrained weights from localization models allowed to train classification models much faster and to have better accuracy. Features from "pre" and "post" images connected at the very end of the Decoder in bottleneck part, this helping not to overfit and get better generalizing model.

Classification training parameters:
Loss: Dice + Focal + CrossEntropyLoss. Larger coefficient for CrossEntropyLoss and 2-4 damage classes.
Validation metric: competition metric
Optimizer: AdamW
Sampling: classes 2-4 sampled 2 times to give them more attention.

Almost all checkpoints finally finetuned on full train data for few epoches using low learning rate and less augmentations.

Predictions averaged with equal coefficients for both localization and classification models separately.

Different thresholds for localization used for damaged and undamaged classes (lower for damaged).


# Conclusion and Acknowledgments

Thank you to xView2 team for creating and releasing this amazing dataset and opportunity to invent a solution that can help to response to the global natural disasters faster. I really hope it will be usefull and the idea will be improved further.

# References
 - Competition and Dataset: https://www.xview2.org
 - UNet: https://arxiv.org/pdf/1505.04597.pdf
 - Pretrained models for Pytorch: https://github.com/Cadene/pretrained-models.pytorch
 - My 1st place solution from "SpaceNet 4: Off-Nadir Building Footprint Detection Challenge" (some ideas came from here): https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/cannab