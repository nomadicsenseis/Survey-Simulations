# Simulations

The following is a tool developed at Iberia for the robust simulation of survey answers to derive statistical estimates from them. Therefore, the complexity to achieve robustness is double: robust inference and robust simulations.

The idea is that, for any given set of satisfactions ()

# Can we simulate the population of customers that generates specific targets?

1. **"The population"**: How do we know this population is unique? If it is not unique, how can we generate a probable population?

2. **"Simulate"**: Is there a robust and generalizable methodology to generate populations that meet the targets?

---

To answer any of these questions, the first step is to create a test environment where we can isolate and verify the quality of our simulations. The goal will be to design a methodology that gives us enough confidence in this laboratory to generalize it to the real world.

---

## Laboratory and Verification Tools

What better way to verify than by trying to simulate historical samples? When simulation targets are defined, one of the challenges is that these targets may attempt to represent relationships between variables that go beyond historical patterns and new initiatives. This effect is an additional complexity to the exercise that, for now, we aim to avoid.

At this stage, the question is: given fixed targets (perfectly realistic and with known relationships between variables), **are we capable of simulating the sample that generates them?**

How do we verify whether the real and simulated populations are similar? The answer to this question lies in comparing the accuracy of NPS and exploitability at the customer level with the model.

---

The first result obtained from this setup is that there are, INDEED, multiple customer populations (number **n**) that produce the same **tp**:

1. **Soft simulations**: The original NPS does not change in an initial check when customers are unrealistic but not ...  
2. **Hard simulations**.

This result raises Question 2: **How do we define and narrow down a probable population, case by case?** The aggregated model gives us an approximation to the correct population. We attempt to vary each customer more to cause more variation in NPS and check with a **hard check**.

---

## NPS: P, N, and D in the Context of the Survey

### NPS:

- S1, S2, S3 ... Sm

**Associative product** between P, N, and D.


