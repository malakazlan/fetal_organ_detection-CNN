# fetal_organ_detection-CNN
fetal organ detection using RESNET model CNN
Project Title:
Fetal Organ Detection and Segmentation Using 
Convolutional Neural Networks (CNNs)
Group Members:
1. Azlan Malik
2. Laraib Irfan
3. Abdul Subhan Tariq
Research Paper Selection:
We have selected the following research papers as the foundation for our project:
1. "Transfer learning for accurate fetal organ classification from ultrasound images: A potential 
tool for maternal healthcare providers"
o Published: October 20, 2023
o Source: PMC PubMed Center
o URL: https://www.nature.com/articles/s41598-023-44689-0
2. "Computer-aided classification of fetal images from ultrasound data"
o URL: https://link.springer.com/article/10.1007/s00330-007-0604-3
3. "Segmentation of ultrasound images using deep learning"
o URL: https://ieeexplore.ieee.org/abstract/document/7065897
Summary of the Research Papers:
The selected research papers explore the use of Convolutional Neural Networks (CNNs) for detecting 
and segmenting fetal organs from ultrasound images. They highlight challenges such as variability in 
imaging conditions (e.g., fetal position, maternal anatomy, device settings) and present methodologies 
to overcome these issues.
Key Contributions:
1. Architecture Design:
o The studies propose CNN architectures that balance complexity and accuracy using 
convolutional layers, max pooling, and dropout techniques.
2. Preprocessing and Augmentation:
o The research emphasizes preprocessing techniques (e.g., resizing, normalization) and 
data augmentation (e.g., flipping, rotation) to increase model robustness.
3. Evaluation Metrics:
o The papers utilize metrics such as accuracy, precision, recall, and F1-score to measure 
the effectiveness of their models.
4. Results:
o Achieving classification and segmentation accuracies above 90%, the models 
demonstrate their utility for real-world medical imaging applications.
Objective of the Proposal:
The primary objective of this project is to reproduce and extend the results from the selected research 
papers. We aim to implement a pipeline for fetal organ detection and segmentation using CNNs, 
focusing on:
• Accurate classification of fetal organs (e.g., brain, thorax, abdomen, femur).
• Implementation of preprocessing, augmentation, and segmentation techniques.
• Evaluation using relevant metrics to assess model performance.
Why Use ResNet Model?
We propose using ResNet (Residual Network) as our CNN backbone for the following reasons:
1. Avoiding Vanishing Gradient Problems:
ResNet introduces residual connections, which allow gradients to flow directly through the 
network, solving the vanishing gradient problem in deeper architectures.
2. State-of-the-Art Performance:
ResNet has demonstrated exceptional performance in medical imaging tasks, especially for 
tasks requiring high accuracy, such as fetal organ segmentation.
3. Efficient Feature Extraction:
Its deep architecture efficiently extracts high-level features, which is critical for detecting 
subtle differences in ultrasound images.
4. Transfer Learning Capabilities:
Pretrained ResNet models can be fine-tuned on our ultrasound dataset, saving training time 
and improving performance.
Proposed Implementation Details:
1. Dataset:
o A labeled dataset containing fetal ultrasound images of organs such as the brain, 
thorax, abdomen, and femur.
o The dataset will be split into training, validation, and testing subsets.
2. CNN Architecture:
o We will implement a ResNet-based architecture that includes:
▪ Residual connections to improve gradient flow.
▪ Convolutional layers with ReLU activation for feature extraction.
▪ Batch normalization for faster convergence and stability.
▪ Max pooling for dimensionality reduction.
▪ Dense layers with dropout for classification.
3. Software and Tools:
o Framework: TensorFlow
o Language: Python
o Development Environment: Google Colab, Jupyter Notebook, or a local GPU-enabled 
system.
4. Evaluation Metrics:
o Accuracy
o Precision
o Recall
o F1-Score
Conclusion:
By implementing this project, we aim to gain practical experience with CNN architectures, particularly 
ResNet, for medical imaging applications. This work will enhance our understanding of deep learning 
techniques and contribute to the development of precise fetal organ detection and segmentation 
systems. Furthermore, by replicating and building on the selected research, we aim to provide a robust 
pipeline that supports maternal healthcare providers in critical diagnostic tasks.
