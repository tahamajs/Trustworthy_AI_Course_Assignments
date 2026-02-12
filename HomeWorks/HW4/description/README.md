This document is Homework Assignment #4 for the **Trustworthy Artificial Intelligence** course at the **University of Tehran**, Faculty of Engineering, Department of Electrical and Computer Engineering. The course is taught by Dr. Mostafa Tavasolipour, and this assignment was issued in **June 2024 (Khordad 1403)**.

---

## Table of Contents

* 
**Question 1: Security** 


* Part 1: Trigger Identification (Reverse Engineering and Label Identification) 


* Part 2: Model Cleansing and Mitigation 




* 
**Question 2: Privacy** 


* 
**Question 3: Fairness** 


* 
**Bonus Section** 


* 
**References & Submission Guidelines** 



---

## Question 1: Security

This section focuses on **Backdoor Attacks**, where a trigger is placed in training data to cause a specific misclassification while maintaining normal performance on clean data. You are provided with a convolutional model trained on MNIST that has been attacked. Using the **Neural Cleanse** paper , you must identify the trigger and cleanse the model.

### Part 1: Trigger Identification

* 
**Assumptions:** You have access to the attacked model and a small set of clean data (MNIST test set).


* 
**Model Weights:** Load the weights corresponding to the last digit of your student ID.


* 
**Model Architecture:** * `conv1`: Conv2d(1, 16, 5x5)  ReLU  AvgPool2d 


* 
`conv2`: Conv2d(16, 32, 5x5)  ReLU  AvgPool2d 


* 
`fc1`: Linear(512, 512)  ReLU 


* 
`fc2`: Linear(512, 10)  Softmax 


* 
`dropout`:  





### Sub-sections:

1. 
**Reverse Engineering the Trigger:** Explain the two terms in the optimization function and implement the reconstruction for all labels.


2. 
**Identifying the Attacked Label:** Use the **Median Absolute Deviation (MAD)** method to detect the outlier (attacked) label and display its reconstructed trigger.



### Part 2: Model Cleansing

1. Explain the three mitigation methods from the paper.


2. Implement the **Unlearning** method. Report the model accuracy and Attack Success Rate (ASR) before and after cleansing.


* 
*Note:* Use the MNIST test set and apply the reconstructed trigger to 20% of the data for one epoch of unlearning.





---

## Question 2: Privacy

This section covers **Differential Privacy (DP)** using the Laplace mechanism.

### Part 1: Laplace Mechanism

* 
**Scenario:** Protecting mean and total income for a community of 500 people.


* 
**Query 1 (Average Income):** Sensitivity , .


* 
**Query 2 (Total Income):** Sensitivity , .


* 
**Tasks:** * Calculate the scale parameter () for each query.


* Calculate privacy-preserving values given specific noise samples.


* Explain the effect of **Composition** (splitting  into  and ).





### Part 2: Sequential Queries

* 
**Scenario:** 92 counting queries on database  where each response is in  and sensitivity .


* 
**Parameters:** , .


* **Tasks:**
* Calculate the Laplace scale parameter.


* Calculate the probability that a noisy response for  exceeds 505 using the CDF.


* Recalculate for sequential queries where  and .


* Adjust for **Unbounded DP** if  percent of the population is added/removed.





---

## Question 3: Fairness

You are a data scientist predicting if employees earn  or , ensuring no gender bias.

### Part 1: Data and Evaluation

* Load `data.csv`.


* Implement evaluation metrics: **Accuracy**, **Zemel Fairness**, and **Disparate Impact**.


* 
**Zemel Fairness:** .


* 
**Disparate Impact:** .





### Part 2: Base Model

* Split data (70% train / 30% test) and train a classifier.


* Analyze if the model is accurate, fair, and if removing the sensitive feature helps.



### Part 3: Fair Model Implementation

* Implement a **bias mitigation** technique:


1. Rank men with income  by ascending probability (Promotion - CP).


2. Rank women with income  by descending probability (Demotion - CD).


3. Swap the labels for the top  rows in each category and retrain.





### Part 4 & 5: Comparison and Bonus

* Compare the Base and Fair models in a table regarding accuracy and fairness.


* 
**Bonus:** Propose and implement a different fairness method and compare results.



---

## Submission Guidelines

* 
**Deadline:** Friday, July 5th (15 Tir). No "Grace" period allowed.


* 
**Format:** Python code and a report in a `.zip` file named `HW4_[Lastname]_[StudentNumber].zip`.


* 
**Integrity:** Individual work only; zero tolerance for plagiarism or using source code without attribution.


* **Contact (TAs):**
* Q1 (Security): Mahyar Maleki (mahyar.maleki@ut.ac.ir) 


* Q2 (Privacy): Mehdi Dehshiri (mhdhshcom.gmail@ri) 


* Q3 (Fairness): Farzaneh Hatami (farzaneh.hatami@ut.ac.ir) 





