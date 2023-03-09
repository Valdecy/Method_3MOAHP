# 3MOAHP

## Introduction
The multi-criteria technique analytic hierarchy process (AHP) has a significant drawback. If the pairwise comparison matrix (PCM) has inconsistent comparisons, in other words, a consistency ratio (CR) above the value of 0.1, the final solution cannot be validated. Many studies have been developed to treat the inconsistency problem, but few of them tried to satisfy different quality measures. The selected quality measures are: minimum inconsistency (**fMI**), the total number of adjusted pairwise comparisons (**fNC**), original rank preservation (**fKT**), minimum average weights adjustment (**fWA**) and finally, minimum L1 matrix norm between the original PCM and the adjusted PCM (**fLM**).

Our approach is defined in four steps: 
- 1) The decision-maker should choose which **quality measures** she/he wishes to use, ranging from one to all quality measures. 
- 2) The authors encode the PCM to be used in a many-objective optimization algorithm (MOOA), and **each pairwise comparison can be adjusted individually**. 
- 3) The authors generate consistent solutions from the obtained Pareto optimal front that carry the desired quality measures in the third step. 
- 4) The decision-maker selects the most suitable solution for her/his problem. Remarkably, the decision-maker can choose one (mono-objective), two (multi-objective), three or more (many-objectives) quality measures.

It's worth noting that our implementation can deal with AHP and Fuzzy AHP. The Fuzzy AHP needs a fuzzy triangular scale to work, and although the user can define his scale, we have implemented a default fuzzy triangular scale that can be used in most problems:

| Crisp Number |   Fuzzy Number  | 
|--------------|-----------------|
|     1/9      | (1/9, 1/9, 1/9) |
|     1/8      | (1/9, 1/8, 1/7) |
|     1/7      | (1/8, 1/7, 1/6) |
|     1/6      | (1/7, 1/6, 1/5) |
|     1/5      | (1/6, 1/5, 1/4) |
|     1/4      | (1/5, 1/4, 1/3) |
|     1/3      | (1/4, 1/3, 1/2) |
|     1/2      | (1/3, 1/2,   1) |
|       1      | (  1,   1,   1) |
|       2      | (  1,   2,   3) |
|       3      | (  2,   3,   4) |
|       4      | (  3,   4,   5) |
|       5      | (  4,   5,   6) |
|       6      | (  5,   6,   7) |
|       7      | (  6,   7,   8) |
|       8      | (  7,   8,   9) |
|       9      | (  9,   9,   9) |
 
## Citation
Floriano, C.M., Pereira, V. and Rodrigues, B.e.S. (2022), **3MO-AHP: An Inconsistency Reduction Approach through Mono-, Multi- or Many-objective Quality Measures**, 
Data Technologies and Applications, Vol. 56 No. 5, pp. 645-670. https://doi.org/10.1108/DTA-11-2021-0315 

## Usage
1. Install

```bash
pip install method_3mo_ahp
```

2. Try it in **Colab**:

AHP: 
- Example 01 - One Objective: ([ Colab Demo ](https://colab.research.google.com/drive/17UC74CW_Bvjk7ZQkvcLli6vF0iL2JAWh?usp=sharing))
- Example 02 - Two Objectives: ([ Colab Demo ](https://colab.research.google.com/drive/1J6nRuXY4TQK_6HXaUtNJyEXh1BION8iA?usp=sharing))
- Example 03 - Three Objectives: ([ Colab Demo ](https://colab.research.google.com/drive/1gI-ZAp9XnLjrKvT_tzzkOJHnmmsDBb4n?usp=sharing))
- Example 04 - Five Objectives: ([ Colab Demo ](https://colab.research.google.com/drive/1ckBaGoD5uglkfwDzJtqPztFyDeDJkVil?usp=sharing))

Fuzzy AHP: 
- Example 05 - One Objective: ([ Colab Demo ](https://colab.research.google.com/drive/1Td-_m2rGTz1tcY3D7e8CcXvBJNZFHXgj?usp=sharing))
- Example 06 - Two Objectives: ([ Colab Demo ](https://colab.research.google.com/drive/1PYZXH_NmKf6IeH7imRaCl0TCcxt24ezh?usp=sharing))
- Example 07 - Three Objectives: ([ Colab Demo ](https://colab.research.google.com/drive/1EMjmYHivEcm7W7RggKPbNf3Fw58E2XqH?usp=sharing))
- Example 08 - Five Objectives: ([ Colab Demo ](https://colab.research.google.com/drive/1NNKvt-tGIxuXFGvWKcxXcsl9-jbfwnPQ?usp=sharing))
- Example 09 - Custom Fuzzy Scale: ([ Colab Demo ](https://colab.research.google.com/drive/1uWfN804d2fIoznx-SCAXNw3ux2d45-fS?usp=sharing))

3. Others
- [pyDecision](https://github.com/Valdecy/pyDecision) - A library for many MCDA methods
- [pyMissingAHP](https://github.com/Valdecy/pyMissingAHP) - A Method to Infer AHP Missing Pairwise Comparisons
- [ELECTRE-Tree](https://github.com/Valdecy/ELECTRE-Tree) - Algorithm to infer the ELECTRE Tri-B method parameters
- [Ranking-Trees](https://github.com/Valdecy/Ranking-Trees) - Algorithm to infer the ELECTRE II, III, IV and PROMETHEE I, II, III, IV method parameters
