## Trusted Artificial Intelligence - Homework #1

**University of Tehran, College of Engineering** **Department of Electrical and Computer Engineering** 

**Instructor:** Dr. Mostafa Tavasolipour 

**Date:** March 2024 (Esfand 1402) 

---

## Introduction

Welcome to the first assignment of the Trusted AI course. This exercise focuses on the fundamental concepts of **Generalization** and **Robustness**. We will explore essential aspects of building AI models that generalize well to new, unseen data while maintaining stability (robustness) against uncertainties and adversarial inputs.

---

## Part 1: Generalization

Generalization refers to a model's ability to accurately predict new data based on patterns learned during training. In this section, you will apply various techniques to improve the generalization of a neural network for classification.

### 1.1 Dataset and Model Setup

You will use a **ResNet18** model (to be implemented manually, not using `torchvision.models`).

* **Training Dataset:** SVHN (Street View House Numbers). It contains ~76k training and ~26k test images (3-channel,  pixels).


* **Testing Dataset:** MNIST. It contains 60k training and 10k test images (single-channel).


* 
**Task:** Train on SVHN and test on both SVHN and MNIST test sets. Mention how you handle the channel difference (1-channel vs 3-channel) in your report.


* **Training Details:** Use SGD with momentum for at least 10 epochs. Report the loss plot and analyze results. **(15 Points)** 



### 1.2 Improving Generalization

Apply the following techniques independently to the baseline model and report the results:

* **Model Architecture:**
1. Briefly explain (max 1 paragraph each) how **Dropout** and **Batch Normalization** improve generalization based on the provided references. **(2 Points)** 


2. Remove all Batch Normalization layers from your ResNet18, retrain, and compare results. **(4 Points)** 




* 
**Loss Function:** 1.  Explain the **Label Smoothing Regularization** technique and how it affects Cross Entropy. **(2 Points)** 
2.  Implement Label Smoothing Cross Entropy () and report the impact. **(8 Points)** 


* 
**Data Augmentation:** Identify which augmentations are unsuitable for this specific data (and why). Find a set of augmentations through trial and error that improves generalization to the unseen dataset. **(8 Points)** 


* **Feature Extraction (Pre-trained Models):**
Use a ResNet18 pre-trained on ImageNet (via `torchvision.models` with `pretrained=True`). Report and analyze its performance as a powerful feature extractor. **(4 Points)** 


* **Optimizer:**
Repeat the training using the **Adam** optimizer while keeping other settings constant. **(4 Points)** 



### 1.3 Reverse Training

* **Unsupervised:** Use your best settings to train on **MNIST** and test on **SVHN**. Compare accuracy with the previous order. **(5 Points)** 


* 
**Supervised (Fine-tuning):** Fine-tune the model (trained on MNIST) using a small subset (500-1000 samples) of SVHN. **Freeze the convolutional layers** and only fine-tune the final classifier. **(8 Points)** 



---

## Part 2: Robustness

This section explores **Adversarial Attacks** and improving model resistance using **Circle Loss** and **Adversarial Training** on the **CIFAR10** dataset.

### 2.1 Adversarial Attacks

* 
**Setup:** Use a pre-trained ResNet18 on CIFAR10. Split data: 20% for training and 80% for fitting, ensuring class balance.


* 
**Methods:** Briefly explain **FGSM** and **PGD** attacks.


* **Task:** Generate adversarial examples using FGSM () and random pixel noise. Display some of these images. **(7 Points)** 



### 2.2 Evaluation and Visualization

For all following cases, report test accuracy, loss plots, and visualize 512-dimensional features in 2D using **UMAP**:

* 
**A) Baseline:** Train with standard Cross Entropy and original data. Evaluate on original vs. adversarial test data. **(10 Points)** 


* 
**B) Augmented Training:** Train with 50% probability of adversarial perturbation per data point using Cross Entropy. Evaluate on original vs. adversarial test data. **(8 Points)** 


* 
**C) Circle Loss Theory:** Briefly explain the benefits of **Circle Loss** and what problems it solves compared to previous methods. **(6 Points)** 


* 
**D) Circle Loss Training:** Train using Circle Loss and evaluate on original vs. adversarial test data. **(9 Points)** 



**Comparison:** Compare and analyze results from parts A, B, and D. **(10 Points)** 

---

## Submission Guidelines

* 
**Deadline:** Monday, April 8, 2024 (20 Farvardin). No extensions, but grace time is available.


* **Format:** One-person project. Submit a `.zip` file named `HW1_[Lastname]_[StudentNumber].zip` containing your Python code and report.


* 
**Integrity:** Any similarity in code or reports will be considered plagiarism. Using external code without citation is prohibited.


* 
**Report:** Must be typed (not handwritten), include numbered captions for all figures/tables, and detail the problem-solving process.



**Happy Holidays and Good Luck!** 
