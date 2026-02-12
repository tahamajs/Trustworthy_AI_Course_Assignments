
## Part 1: Interpretability for Tabular Data

The goal of this section is to investigate the interpretability of Neural Network models using various methods on a tabular dataset containing health information for diabetes prediction.

### Section 1: Data Loading and Model Training

1. **Exploratory Data Analysis (EDA):**
* Load the `diabetes.csv` file.


* Analyze dependencies using a **Correlation Matrix** and **Pairplot**. Identify which two pairs of features (other than the 'Outcome' column) are highly correlated.


* Plot the distribution of healthy vs. diabetic individuals and discuss the **data balance**.


* Check for **outliers** and display data dispersion. Discuss if these affect model accuracy or analysis.




2. 
**Preprocessing:** Normalize data using `sklearn.preprocessing`. Split the data into **Train (70%)**, **Validation (10%)**, and **Test (20%)** sets with identical distributions.


3. 
**Model Training:** Design and train an MLP (Multi-Layer Perceptron) based on the following architecture:



| Layer | Configuration |
| --- | --- |
| **Linear** | Input: 8, Output: 100, Activation: ReLU |
| **Batch Norm** | Size: 100 |
| **Linear** | Input: 100, Output: 50, Activation: ReLU |
| **Dropout** |  |
| **Linear** | Input: 50, Output: 50, Activation: ReLU |
| **Linear** | Input: 50, Output: 20, Activation: ReLU |
| **Linear** | Input: 20 (Note: doc says 10, likely typo), Output: 1 |

* Report **Accuracy, Recall, F1-score, and Confusion Matrix** on the test set.



### Section 2: Model Interpretation

4. 
**LIME Method:** Use the `LIME` library and `LimeTabularExplainer` to analyze the model for three random test samples.


5. **SHAP Method:** Use the `shap` library and `KernelExplainer` to analyze the same three samples. Generate `force_plot` visualizations.


6. **Comparison:** Compare LIME and SHAP results. Discuss similarities/differences and which method appears more accurate.


7. 
**Correlation Analysis:** Relate the outputs of LIME and SHAP to the Correlation Matrix from Step 1.



### Section 3: Neural Additive Models (NAM)

8. 
**NAM Implementation:** * Read the paper "Neural Additive Models". Explain the differences between NAM and black-box models, including pros/cons regarding performance and interpretability.


* Design a `NAMClassifier` for the dataset.


* Analyze if NAM improves interpretability compared to LIME and SHAP.





### Bonus Section: GRACE

Investigate the **GRACE** method for generating contrastive samples. Analyze how probability predictions change when features are modified, and use SHAP to explain these changes.

---

## Part 2: Interpretability in Computer Vision

This section focuses on **Pixel Attribution** (identifying important pixels via Heatmaps/Saliency Maps) and **Feature Visualization** (understanding what specific layers/filters have learned).

### General Requirements:

* Use a pre-trained **VGG16** model on **ImageNet**.


* Select **6 images** from different classes that the model classifies correctly.



### 1. Grad-CAM

* Explain the idea and mathematical relationships of **Grad-CAM** based on the referenced paper.


* Implement it on your chosen images and report the saliency maps.



### 2. Guided Grad-CAM

* Research **Guided Backpropagation** and its advantages over standard backpropagation for saliency maps.


* Implement Guided Backpropagation.


* Explain and implement **Guided Grad-CAM** (the fusion of Grad-CAM and Guided Backpropagation).



### 3. SmoothGrad

* Explain the concept of **SmoothGrad** (adding noise to remove noise).


* Implement **SmoothGrad + Guided Backpropagation**.


* Implement **SmoothGrad + Guided Grad-CAM** and report results.



### 4. Adversarial Perturbation

* Select one image and apply an attack (e.g., **PGD, FGSM**) to change the predicted class.


* Compare the saliency map of the original image with the adversarial image for the original class.



### 5. Feature Visualization

* Use **Activation Maximization** to find an image that maximizes the "Hen" class logits in VGG16.


* Explain why the initial generated image is not "meaningful".


* Apply **Total Variance Regularization** and **Random Shifts** to produce 5 meaningful images. Explain why these techniques help.



---

## Submission Guidelines

* 
**Deadline:** Friday, May 10, 2024 (21 Ordibehesht).


* 
**Format:** One `.zip` file named `HW2_[Lastname]_[StudentNumber].zip`.


* 
**Language:** Python.


* **Originality:** This is an individual assignment. Plagiarism or using uncredited ready-made code will result in a grade of zero.


* **Report:** Must be typed (no handwriting). Include captions and numbers for all tables and images.



