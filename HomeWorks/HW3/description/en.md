## Trusted Artificial Intelligence - Homework #3

**University of Tehran** 

**College of Engineering** 

**Department of Electrical and Computer Engineering** 

**Instructor:** Dr. Mostafa Tavasolipour 

**Date:** May 2024 (Ordibehesht 1403) 

---

### Introduction

The main objective of this exercise is to better understand **causal relationships** between variables and to interpret models using these relationships. By applying causality in interpretability, we can gain a deeper understanding of complex phenomena and make more appropriate decisions for potential interventions.

---

### Question 1 (10 Points)

Assume  represents the size of a patient's kidney stone, where  indicates a large stone and  indicates a regular stone.
 represents the patient's treatment status:  for the new treatment and  for the old treatment.
 represents whether the stone was removed after one month:  for removal and  for retention.

Based on the decomposition of  according to the DAG (Figure 1) and the following conditional distributions: 

* 
 


* 
 


* 
 


* 
 


* 
 


* 
 


* 
 



**Tasks:**

1. 
**Subsection 1:** Calculate the values of  and .


2. 
**Subsection 2:** Calculate the values of  and .



---

### Question 2 (12 Points)

Two individuals, A and B, have their loan applications rejected by a bank. Using the provided **Structural Causal Model (SCM)** and the **Causal Algorithmic Recourse** problem, find the optimal state in which the loan would be granted and the cost associated with each change.

**Individual Features:**

* 
 


* 
 



**Model Parameters:**

* 
 


* 
 


* 
 (Annual Salary) 


* 
 (Bank Balance) 


* Classifier:  



---

### Question 3 (20 Points)

This question uses a dataset from an airline’s operations management system. Features include:

* 
**Booking_Mode:** Indicates if a booking occurred due to a specific reason (e.g., holidays).


* 
**Marketing_Budget:** Spending on advertising.


* 
**Website_Visits:** Daily visitors.


* 
**Ticket_Price:** Average price per ticket.


* 
**Tickets_Sold:** Total tickets sold.


* 
**Sales_Revenue:** Income from tickets.


* 
**Operating_Expenses:** Operational costs (salaries, rent, etc.).


* 
**Profit:** Revenue minus operating expenses.



**Tasks:**

1. 
**Subsection 1:** Draw the causal graph using the `networkx` library.


2. 
**Subsection 2:** Model a **Structural Causal Model (SCM)** where each variable is a function of its parents plus noise. Use linear regression, decision trees, or other suitable models. Fit the model parameters using the dataset.


3. 
**Subsection 3:** Calculate the variance of profit. Determine whether Revenue or Operating Expenses has a greater impact on profit variance and explain if the results are logical.


4. 
**Subsection 4:** Identify which system factor contributes most to the overall profit variance.


5. 
**Subsection 5:** Analyze the provided table for the first day of the new year. Determine if profit increased or decreased compared to the previous year and explain the reasons.



---

### Question 4 (22 Points)

Estimate the effect of **Insulin ()** on **Blood Glucose ()** considering **Age ()** and **Blood Pressure ()**.

**Tasks:**

1. Using `health.csv` and logistic regression, calculate:
* 
 


* 
 


* 
 




2. Prove which of the above represents the **causal effect** of Insulin on Blood Glucose.



---

### Question 5 (20 Points)

Design a classifier to separate "Healthy" (1) and "Unhealthy" (0) groups using `health.csv`. Then, compare **Nearest Counterfactual Explanation** and **Causal Algorithmic Recourse** to find the most cost-effective intervention to change a patient's status from unhealthy to healthy.

**Subsections:**

1. Complete `process_health_data` in `data-utils.py` so only **Insulin** and **Blood Glucose** are "actionable".


2. Run `main.py` for 10 unhealthy individuals and report the cost.


3. Complete the `Health_SCM` class in `scm.py` with specific constant and actionable features.


4. Implement the `get_Jacobian` function for the SCM.


5. Uncomment the SCM code in `utils.py`, re-run `main.py` for the 10 individuals, and report the cost.


6. Compare costs from steps B and E. Explain which is lower and why.



---

### Question 6 (16 Points)

Review the paper *"On the Adversarial Robustness of Causal Algorithmic Recourse"* and answer: 

1. Under what conditions is robustness guaranteed in the classifier and SCM for Causal Algorithmic Recourse? Explain intuitively.


2. Based on **Proposition 4** of the paper, explain the intuitive reasons for Equation 5.



---

### Submission Notes

* **Deadline:** May 30, 2024 (10 Khordad). No extensions, but grace time is available.


* **Format:** Implement in **Python**. Submit code and a report in a `.zip` file named: `HW3_[Lastname]_[StudentNumber].zip`.


* **Integrity:** This is an individual assignment. Any similarity in code or reports will be considered plagiarism.


* **Quality:** Handwritten submissions are not accepted. All figures and tables must have captions and numbers.



