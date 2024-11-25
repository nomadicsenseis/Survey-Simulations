# Simulations: Project Overview

This project focuses on the robust generation of a survey population that meets specified targets for key descriptive variables. The objectives correspond to the satisfaction levels in the sample (% of responses ≥ 8) for each of the customer experience touchpoints of interest to Iberia, including punctuality, check-in, food, and others.

The main goal is to simulate a population of surveys that accurately reflects these satisfaction levels while maintaining consistency and realism in the data.

## Summary Diagram

To visually summarize the simulation process and its challenges, refer to the following diagram:

![Simulations Schema](src/Simulations_basics.png)

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

Our initial experiments provided valuable insights into the uniqueness and realism of simulated populations:

### Experiment 1: Soft Simulations
To test whether there can be more than one population of customers meeting the same targets, we conducted a quick experiment called **soft simulations**.  
- **Methodology**: We selected a historical period as the target population and used the same period from the previous year as a baseline. Then, we minimally altered each customer’s satisfaction scores to meet the targets. Specifically, we:
  - Increased some scores of `7` to `8`, or
  - Decreased some scores of `8` to `7`.  
  Each customer had at most one score adjusted, either up or down, to change the satisfaction levels.  

- **Results**:  
  The simulated population met the satisfaction targets, but the predicted NPS for the simulated sample differed significantly from the target population’s NPS. Surprisingly, the simulated NPS was much closer to the original (unaltered) population's NPS. This indicated that:
  1. **The individual customers in the simulated sample, though not real, were realistic** in terms of their profiles and characteristics.  
  2. The issue lay in the **overall population**, which was no longer realistic, despite the individual customers being plausible.  

This revealed a critical insight: there can be more than one population of realistic customers (e.g., the soft-simulated and original populations) that meet the same satisfaction targets but yield different NPS values.  

### Experiment 2: Hard Simulations
To further explore the variability of simulated populations, we developed a **hard simulation** algorithm:  
- **Methodology**: Instead of altering only one score per customer, we simultaneously increased or decreased scores for multiple variables (e.g., several touchpoints). This introduced more substantial changes to customer profiles while still ensuring the satisfaction targets were met.  

- **Results**:  
  The NPS of the hard-simulated population exhibited greater variability compared to the soft simulations, confirming that:
  1. **Different simulation methods produce populations with varying NPS values**, even when satisfaction targets are met.  
  2. The realism of the population as a whole depends not only on individual customer adjustments but also on the relationships between those adjustments across the entire dataset.

---

These experiments underscore the existence of **multiple possible customer populations (n)** that meet the same satisfaction targets. The challenge lies in defining additional constraints or selecting a methodology that prioritizes the most probable or useful population for real-world applications.


This result raises Question 2: **How do we define and narrow down a probable population, case by case?** The aggregated model gives us an approximation to the correct population. We attempt to vary each customer more to cause more variation in NPS and check with a **hard check**.

# Technical Lanscape

To visually summarize the simulation process and its challenges, refer to the following diagram:

![Simulations Landscape](src/Simulations_landscape.png)

# Technical Landscape

In this project, we faced a complex technical landscape for simulating survey populations that meet specific satisfaction targets. The diagram below summarizes the main approaches we explored:

![Simulations Schema](src/Simulations_schema.png)

## Approaches Explored

### 1. Survey Repositories
We categorized the sources of survey data into two main types:  
- **Artificial**:  
  - **Copulas**: Used to generate synthetic data based on probabilistic dependencies between variables.  
  - **GANs (Generative Adversarial Networks)**: Applied to create synthetic survey data that closely mimics historical patterns.  

- **Historical Data**: Leveraging existing survey datasets as a realistic baseline to ensure the simulated populations reflect real-world distributions.  

---

### 2. Convergence Algorithms
The algorithms for simulating populations that meet satisfaction targets are based on **two key pillars**:
1. **Client Characterization**: Understanding and segmenting clients into meaningful groups based on their behavior.  
2. **Optimization Algorithms**: Adjusting satisfaction scores or reshaping the population to meet specific targets.

#### 2.1. Client Characterization
Characterization is crucial for ensuring that simulated populations remain realistic. We explored two complementary approaches:

- **Client Model (NPS + SHAPs)**:  
  - This approach models each individual client by predicting their NPS and explaining the contribution of each touchpoint to that score using **SHAP (SHapley Additive exPlanations)** values.  
  - The predicted NPS provides a quantitative understanding of customer sentiment, while SHAP values offer a detailed breakdown of how each touchpoint influences the overall score.  

- **K-means Clustering**:  
  - Using the **elbow method**, we identified that segmenting clients into **k = 3** clusters provided well-defined groups, which align with our hypothesis of biases in the survey data:  
    1. **Loyalty Bias**: Captures clients whose behavior is consistent with promoters, generalizing their loyalty across multiple touchpoints.  
    2. **Incidence Bias**: Represents clients whose dissatisfaction stems from issues across multiple touchpoints, aligning with detractors.  
    3. **Objective Clients**: Represents neutral clients, without strong biases toward loyalty or incidences.  
  - While K-means itself is agnostic to these biases, our **causal interpretation** of the clusters allows us to operationalize them in subsequent optimization steps.  

These two characterization methods (NPS + SHAPs and K-means) complement each other:
- The **Client Model** provides individual-level insights into NPS and touchpoint contributions.
- **K-means Clustering** groups clients into interpretable segments based on multidimensional patterns, enabling targeted adjustments at a group level. It could be trained using the Client Model characterization.

#### 2.2. Optimization Algorithms
Optimization algorithms modify satisfaction levels or reshape populations to converge toward the specified targets. These can operate at different levels:

- **Survey-Level Algorithms**: Operate on individual clients within the population:  
  - **Soft Simulations**: Adjust one touchpoint score (e.g., increase a `7` to `8` or decrease an `8` to `7`) per client. This naive approach preserves individual realism but introduces minimal variability at the population level.  
  - **Hard Simulations**: Simultaneously adjust multiple touchpoints for each client. While this increases variability at the population level, it risks reducing individual realism.  
  - **Causal Swapping**:  
    - **Core Idea**: This algorithm asumes historical biases and adjusts satisfaction levels by "swapping" clients between groups to generate a population that meets targets while preserving those biases.  
    - **Implementation Without K-means**: Causal swapping could operate directly on NPS labels (promoter or detractor), but this approach might fail to generalize loyalty and incidence biases across multiple touchpoints.  
    - **Implementation with K-means**: Using the clusters from K-means enhances causal swapping by leveraging a multidimensional understanding of biases, ensuring more realistic and consistent adjustments.

- **Population-Level Algorithms**: Modify or generate entirely new populations in each iteration:  
  - **Naive Approaches**: Downsampling or other simplistic methods to filter data based on predefined criteria.  
  - **Complex Algorithms**:  
    - **Genetic Algorithms**: Simulate evolution by iteratively improving populations.  
    - **Bayesian Optimization**: Efficiently searches for optimal population configurations.  
    - **Monte Carlo Simulations**: Use random sampling to model and refine populations.  

---

### 3. Combined Approach: Client Model, K-means, and Causal Swapping
The combination of client characterization (via NPS + SHAPs and K-means) and causal swapping provides a structured simulation process:

1. **Client Model for Individual Insights**:  
   - The NPS + SHAP-based client model offers granular information about how each touchpoint influences individual satisfaction, ensuring adjustments align with customer behavior.  

2. **K-means for Validation and Segmentation**:  
   - K-means clustering confirms the existence of three well-defined client groups, which align with our hypothesized biases.  
   - This segmentation generalizes the concepts of promoters and detractors into a multidimensional space of touchpoints, improving the realism and plausibility of adjustments.  

3. **Causal Swapping for Population Adjustment**:  
   - Clients are swapped between clusters (e.g., from incidence-biased to loyalty-biased or vice-versa) to adjust satisfaction levels while respecting historical biases.  
   - This process ensures that the population meets satisfaction targets without compromising realism or consistency.  

4. **Improved Realism**:  
   - Combining causal swapping with these characterization methods results in more realistic and plausible populations compared to using simpler approaches.

---

### 4. Challenges
- Ensuring the **realism of individual clients** while achieving the satisfaction targets.
- Validating the **causal interpretation** of clusters to confirm alignment with loyalty and incidence biases.
- Balancing the trade-offs between **soft**, **hard**, and **causal swapping** simulations.
- Scaling optimization techniques to larger datasets while maintaining computational efficiency.

---

This landscape highlights how the combination of client characterization (e.g., NPS + SHAPs, K-means) and optimization algorithms (e.g., causal swapping) enables us to balance realism, target alignment, and computational feasibility in simulating survey populations.
