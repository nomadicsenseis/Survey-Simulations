# Simulations: Project Overview

This project focuses on the robust generation of a survey population that meets specified targets for key descriptive variables. The objectives correspond to the satisfaction levels in the sample (% of responses ≥ 8) for each of the customer experience touchpoints of interest to Iberia, including punctuality, check-in, food, and others.

The main goal is to simulate a population of surveys that accurately reflects these satisfaction levels while maintaining consistency and realism in the data.

# Can We Simulate the Population of Customers That Meets Specific Targets?

### Key Questions:

1. **"The Population"**:  
   - How can we determine if the population that meets the targets is unique?  
   - If the population is not unique, what approach can we take to generate a *probable* population?

2. **"Simulate"**:  
   - Is there a robust and generalizable methodology to create populations that reliably meet the given targets?  

---

## Defining the Problem

To address these questions, the first step is to create a controlled test environment—our **simulation laboratory**—to isolate and validate the quality of the simulations. The goal of this laboratory is to establish a methodology robust enough to provide confidence in the results and ultimately generalize the approach to real-world scenarios.  

---

## Laboratory and Verification Tools

### Historical Data as a Baseline:  
To ensure realism, we define the simulation targets based on historical data rather than arbitrarily creating them. This approach leverages known patterns and relationships in customer feedback, providing a reliable foundation for the simulation.

### The Key Question:  
Given fixed, realistic targets—where relationships between variables are known—**can we simulate a sample population that achieves these targets while maintaining consistency with the historical data?**

---

## Verification Methodology

Since we are developing a methodology that will later be generalized, we can directly compare the simulated sample against the historical population associated with the selected targets. This comparison allows us to assess the fidelity of the simulated population. But how can we effectively compare these samples?  

### Comparison Approach:  
In the **Explanatory Drivers** project, we developed a methodology to predict and explain the Net Promoter Score (NPS) at the individual customer level. This methodology is based on **two binary classifiers** trained on historical survey data:  

1. **Promoter Classifier**: Predicts the probability that a customer is a promoter.  
2. **Detractor Classifier**: Predicts the probability that a customer is a detractor.  

From the difference between these two probabilities, we calculate the predicted NPS for each customer. Additionally, both classifiers are designed to provide **explainability** by identifying the contribution of each touchpoint to the respective predictions (promoter or detractor). This dual approach allows us to derive not only a predicted NPS but also an **explanation of the NPS** for each individual customer.  

### Comparing Simulated vs. Historical Populations:  
To evaluate whether the simulated population aligns with the historical data, we aggregate the individual-level characterizations (predicted NPS and explainability scores) for both the simulated and historical populations. If the averaged results differ significantly between the two populations, it indicates that the individual-level profiles (customers or surveys) in the simulated sample are not consistent with the historical sample.

This rigorous comparison ensures that the simulated population matches not only the overall targets but also the underlying distribution of individual characteristics, providing confidence in the quality of the simulation.

---

## Initial Findings

Our initial experiments reveal a fundamental insight:  
There are, in fact, **multiple possible customer populations (n)** that can achieve the same targets. This non-uniqueness highlights the need for additional criteria or constraints to guide the simulation process toward the most *probable* or *useful* populations.


1. **Soft simulations**: The original NPS does not change in an initial check when customers are unrealistic but not ...  
2. **Hard simulations**.

This result raises Question 2: **How do we define and narrow down a probable population, case by case?** The aggregated model gives us an approximation to the correct population. We attempt to vary each customer more to cause more variation in NPS and check with a **hard check**.




