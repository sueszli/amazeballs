---
title: "Amazon Dataset"
output: pdf_document
documentclass: article
papersize: a4
pagestyle: empty
geometry:
    - top=5mm
    - bottom=5mm
    - left=5mm
    - right=5mm
header-includes:
    # title
    - \usepackage{titling}
    - \setlength{\droptitle}{-15pt}
    - \pretitle{\vspace{-30pt}\begin{center}\LARGE}
    - \posttitle{\end{center}\vspace{-50pt}}    
    # content
    - \usepackage{scrextend}
    - \changefontsizes[8pt]{8pt}
    # code
    - \usepackage{fancyvrb}
    - \fvset{fontsize=\fontsize{6pt}{6pt}\selectfont}
    - \usepackage{listings}
    - \lstset{basicstyle=\fontsize{6pt}{6pt}\selectfont\ttfamily}
    # code output
    - \DefineVerbatimEnvironment{verbatim}{Verbatim}{fontsize=\fontsize{6pt}{6pt}}
---


<!-- 

- exercise: https://tuwel.tuwien.ac.at/pluginfile.php/4247741/mod_resource/content/1/DOPP2024_Exercise2.pdf
        - task 21: amazon dataset
- amazon dataset: https://amazon-reviews-2023.github.io/ (most recent, huggingface for easy download)
- algorithms benchmark: https://paperswithcode.com/dataset/amazon-review

-->

# intro

this is just as much about the "data science process" as it is about the results.

this is an open ended task.

we're free to pick whatever dataset we like, and modify questions with supervisor's approval.

*data science process*

- 1 – ask research question
	- define variables, metrics, build hypothesis
- 2 – get the data
	- sample, preprocess, ensure privacy
- 3 – explore the data
	- plot, find patterns and anomalies
- 4 – model the data
	- fit a model, validate
	- bias (inaccurate) vs. variance (overfitting)
- 5 – communicate findings
	- report, visualize
	- correlation ≠ causation

*crisp-dm*

- = cross-industry standard process for data mining
- 1 – business understanding
	- Refine questions (on confirmation with supervisor in review meeting)
	- Beware of biases
- 2 – data understanding
	- Understand what is in the data — are the data measurements or estimates? How accurate are these measurements or estimates? Are there biases in the data (e.g. in the data gathering process)? If you use estimates to make new estimates, how accurate are the new estimates?
- 3 – data preparation
	- Clean the data
        - Check for missing data points – decide what to do about them
        - Check for outliers – decide what to do about them
        - Check for inconsistencies – decide what to do about them
        - Calculate descriptive statistics
        - Transform the data (e.g. changing units of measurements)
        - Check if the necessary data is there to answer the questions. If not, then you could (1) combine columns in some way to generate the necessary data (2) Find the necessary data in another dataset (3) Change the questions asked (in this case you have the freedom to do this, but this may not be the case if someone else is asking the questions)
- 4 – modeling
	- Calculate correlations
- 5 – evaluation
	- Visualize the data
	- Check predictions
	- Answer questions
- 6 – deployment

# research questions

Question 21:

- RQ1: Are reviews for some categories of product on Amazon overall more positive than for other categories?
- RQ2: Are reviews more subjective for some classes of products than for others?
- RQ3: Which aspects of different classes of products are the most important in the reviews?
- RQ4: Can one predict the star rating from the review text?

# methodology

introduce the task we have chosen and why we chose it.

introduce our research questions.

# results
