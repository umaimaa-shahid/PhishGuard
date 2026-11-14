# PhishGuard: Explainable Phishing URL Detection

## Overview

PhishGuard is a machine learning project focused on detecting phishing URLs using URL-based features. It investigates how well phishing detection models distinguish legitimate and phishing URLs, particularly when evaluating URLs from registered domains that were not seen during training.

The project also explores explainability techniques to identify and interpret the URL characteristics that contribute to model predictions.

## Research Question

Can a phishing URL classifier generalize to domains never seen during training, and can its predictions be explained using meaningful URL characteristics?

## Objectives

- Develop machine learning models for phishing URL detection using URL-based features.
- Investigate model generalization to previously unseen registered domains.
- Compare model performance under random and domain-disjoint evaluation settings.
- Identify URL characteristics that contribute to phishing predictions.
- Apply explainability techniques to interpret model decisions.

## Dataset

**Dataset:** [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
**Source:** UCI Machine Learning Repository (Dataset ID: 967)

The original dataset contains 235,795 records and 56 columns. It includes URL-related features and a target label distinguishing
legitimate URLs from phishing URLs. 

After data cleaning and duplicate removal, the dataset contains 235,370 records. The cleaned dataset is used for subsequent exploratory analysis and preprocessing.

**Target variable:** `is_phishing`

- `0`: Legitimate
- `1`: Phishing

The project focuses on URL-based features and does not fetch or analyze live webpage content.

## Phase 1: Data Preparation and Analysis

Phase 1 prepares the dataset for model development and evaluation through three stages: data cleaning, exploratory data analysis, and registered-domain preprocessing.

### 1. Data Cleaning

**Notebook:** `notebooks/01_cleaning.ipynb`
**Input:** raw PhiUSIIL CSV (downloaded from UCI)
**Output:** `data/cleaned.csv`, `data/feature_groups.json`, `data/cleaning_log.json`
 
| Step | Result | Action |
| ---- | ------ | ------ |
| Target recoding | Raw `label` is inverted (1 = legitimate) | Replaced with `is_phishing` (1 = phishing) |
| Missing values | None | Nothing dropped or imputed |
| Formatting | No fixes needed; 2 mojibake URLs | Kept as they are |
| Validity checks | 493 rows where `IsHTTPS` did not match the URL scheme | Corrected from the scheme |
| Duplicate URLs | 425 repeated URLs, all phishing, no label conflicts | Kept first copy, dropped 425 rows |
| Outliers | Mostly phishing (e.g. very long URLs) | Kept; transform later using training data only |
| Feature selection | 30 webpage/HTML columns, `Title` and `FILENAME` | Dropped; only URL-based features kept |
 
**Leakage check:** all legitimate URLs are bare `https://www.<domain>` homepages with HTTPS, while phishing URLs are far more varied. `URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb` and `IsHTTPS` are therefore tagged `leak_suspect` in `feature_groups.json`.
 
**Result:** 235,795 rows and 56 columns reduced to 235,370 rows and 26 columns (100,520 phishing, 134,850 legitimate).

**Output:** `data/cleaned.csv`

### 2. Exploratory Data Analysis (EDA)

**Notebook:** `notebooks/02_eda.ipynb`
**Input:** `data/cleaned.csv`
**Output:** plots in `images/`

The EDA examines the 22 numeric URL-only features of the cleaned dataset (235,370 records) before modelling. The notebook has a written observation under each plot.

#### Analyses performed

| Analysis | Figure |
| -------- | ------ |
| Class balance | `images/01_class_balance.png` |
| Feature distributions by class | `images/02_distributions.png` |
| Boxplots of key features by class | `images/03_boxplots.png` |
| Correlation heatmap (Pearson) | `images/04_correlation_heatmap.png` |
| Single-feature ROC-AUC | `images/05_single_feature_auc.png` |

#### Key findings

- **Class balance:** 134,850 legitimate (57.3%) and 100,520 phishing (42.7%) URLs. The imbalance is mild, so evaluation uses stratified splits and class-aware metrics (phishing recall, F1, PR-AUC) and not accuracy.
- **Discriminative features:** `URLSimilarityIndex` separates the classes almost completely (single-feature AUC 0.996, Pearson r = -0.86 with `is_phishing`). Next are `NoOfOtherSpecialCharsInURL` (0.823), `SpacialCharRatioInURL` (0.807), `URLCharProb` (0.768), `IsHTTPS` (0.757) and `CharContinuationRate` (0.743). Phishing URLs tend to be longer and contain more digits and special characters.
- **Weak features:** obfuscation features, `IsDomainIP`, `NoOfAmpersandInURL` and `NoOfSubDomain` have AUC near 0.50 individually.
- **Redundancy:** `NoOfLettersInURL` and `URLLength` are the only pair above |r| = 0.85 (0.956). The count features move together, and the three obfuscation features are near-duplicates.
- **Suspiciously strong features:** `URLSimilarityIndex`, `IsHTTPS`, `URLCharProb` and `TLDLegitimateProb` may reflect how the dataset was built and not phishing behaviour. They are candidates for the Phase 2 ablation, and models should be reported with and without them.
- **Domain repetition:** at hostname level (the `Domain` column) there are 220,086 unique domains across 235,370 rows, and 54 hostnames appear with both classes. Repeated domains are concentrated in file-hosting and gateway services such as `ipfs.io` and `docs.google.com`. Registered-domain level analysis is in Section 3 and gives different counts because registered domains group subdomains together.
- **TLD skew:** `.app`, `.co`, `.io` and `.ru` are mostly phishing, while `.uk`, `.org`, `.au` and `.de` are mostly legitimate.

### 3. Registered-Domain Preprocessing

This stage extends the cleaned dataset by extracting registered domains from URLs. The extracted domains are used for domain-level analysis and to support domain-disjoint evaluation in Phase 2.

**Notebook:** `notebooks/03_preprocessing.ipynb`

#### Preprocessing Steps

1. **TLD standardization:** Converted TLD values to lowercase and removed surrounding whitespace.
2. **Registered-domain extraction:** Used `tldextract` and the Public Suffix List, including its private-domain entries, to extract registered domains.
3. **IP address handling:** Preserved IP addresses when URLs used IP addresses instead of domain names.
4. **Validation:** Checked for missing or empty extracted domains and identified URLs using IP addresses.
5. **Domain analysis:** Examined domain frequency, repeated domains and domains associated with both target classes.

The original URL values were retained, and the extracted values were added as a separate `registered_domain` column.

#### Domain Analysis Findings

The processed dataset contains the following domain-level observations:

| Observation                           | Result  |
| ------------------------------------- | ------- |
| URL records                           | 235,370 |
| Unique extracted domains              | 197,301 |
| Domains appearing more than once      | 7,463   |
| Domains associated with both classes  | 106     |
| URLs using IP addresses               | 606     |
| Records belonging to repeated domains | 19.34%  |

These findings show that multiple URLs can belong to the same extracted domain, and some domains occur in both target classes.

A random train-test split may place URLs from the same domain in both training and testing sets. Domain-disjoint evaluation is therefore important for assessing generalization to previously unseen domains.

#### Output

`data/cleaned_with_domains.csv`

The processed dataset retains the cleaned features and adds the `registered_domain` column. This column is intended for grouping and evaluation, rather than as a predictive input feature.

## Repository Structure

```text
PhishGuard/
├── data/
│   ├── cleaned.csv
│   ├── cleaned_with_domains.csv
│   ├── cleaning_log.json
│   └── feature_groups.json
├── images/
│   ├── 01_class_balance.png
│   ├── 02_distributions.png
│   ├── 03_boxplots.png
│   ├── 04_correlation_heatmap.png
│   └── 05_single_feature_auc.png
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_preprocessing.ipynb
├── README.md
└── requirements.txt
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/umaimaa-shahid/PhishGuard.git
cd PhishGuard
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

## Reproducing the Results

Run the notebooks in the following order:

| Order | Notebook                   | Description                                                         |
| ----- | -------------------------- | ------------------------------------------------------------------- |
| 1     | `01_cleaning.ipynb`      | Cleans the original dataset and generates the cleaned dataset.    |
| 2     | `02_eda.ipynb`           | Performs exploratory data analysis and generates visualizations. |
| 3     | `03_preprocessing.ipynb` | Extracts registered domains and performs domain-level analysis.   |


Before running the notebooks:

1. Obtain the original PhiUSIIL dataset from the UCI link in the Dataset section.
2. Place the downloaded dataset at the input path expected by `01_cleaning.ipynb`.
3. Run each notebook from top to bottom, in the order listed above. Ensure that each notebook's generated output is available at the input path expected by the next notebook.

The expected outputs include the cleaned dataset, EDA visualizations, and the processed dataset containing the `registered_domain` column.

## Project Scope and Limitations

- Detection is based on URL-related features.
- The project does not fetch or analyze live webpage content.
- Registered-domain information is used for data grouping and evaluation, not as a predictive input feature.
- Phase 1 focuses on data preparation and analysis. Model development and evaluation are planned for Phase 2.

## Future Work

Phase 2 will focus on:

- Training Logistic Regression, Random Forest, and XGBoost classifiers.
- Comparing random and registered-domain-disjoint evaluation.
- Evaluating classification performance using appropriate metrics, including precision, recall, F1-score, and PR-AUC.
- Applying SHAP to interpret model predictions.
- Performing feature-group ablation to investigate the contribution of different URL characteristics.

## Team

| Name          | Student ID | Contribution                                                       |
| ------------- | ---------- | ------------------------------------------------------------------ |
| Ayesha Arshad | 23L-2596   | Data cleaning                                                      |
| Esar Fatima   | 23L-0888   | Exploratory Data Analysis                                         |
| Umaima Shahid | 23L-0893   | Registered-domain extraction, preprocessing and domain analysis. |
