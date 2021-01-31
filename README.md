# AIES_2021_Metric_Supplemental
Supplementary materials for "Measuring Group Advantage: A Comparative Study of Fair Ranking Metrics", in submission at AIES 2021

Additional analysis of bias functions that do not meet are assumptions is given in "metric_supp.pdf"

Code: 

All code is available in the "fair_rank" directory.

fair_rank/Correlation_analysis.ipynb -> Correlation analysis plots on synthetic data for all ranking metrics described in the paper

fair_rank/Evaluate_Bias_Functions.ipynb -> Plots of fairness score for all metrics with various levels of group advantage. Includes plots for all metrics considered in the paper, on both advantage functions that meet our assumptions and on those that do not

fair_rank/FantasyPros.ipynb -> Analysis on the Fantasy Football dataset. Specifically, on we perform the test described in our paper to determine which weeks of data meet our advantage function assumptions and which do not. 

standard python libraries:

numpy

scipy

pandas

matplotlib

seaborn

Fair ranking libraries:

https://github.com/DataResponsibly/FairRank/

https://github.com/fair-search/fairsearch-fair-python

Installation:

Metrics by Yang + Stoyanavich already copied into this repo

pip install fare

pip install fairsearhcore

Datasets:

pip install openml
