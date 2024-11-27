---
title: "Report: Amazon Reviews'23"
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

https://tuwel.tuwien.ac.at/pluginfile.php/4247741/mod_resource/content/1/DOPP2024_Exercise2.pdf

deliverables:

- plan / review meeting document (1 page)
        - research questions
        - datasets planned to use
        - methodology to answer questions
        - division of work
- report (2 pages)
        - management summary document
        - main insights
- a single jupyter notebook
        - like a more verbose version of the report
- presentation (10min)

-->

this is just as much about the "data science process" as it is about the results.

this is an open ended task.

we're free to pick whatever dataset we like, and modify questions with supervisor's approval.

# intro

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

motivation on why this matters

Question 21:

- RQ1: Are reviews for some categories of product on Amazon overall more positive than for other categories? (sentiment analysis: polarity)
- RQ2: Are reviews more subjective for some classes of products than for others? (sentiment analysis: subjectivity)
- RQ3: Which aspects of different classes of products are the most important in the reviews? (topic modeling, aspect extraction)
- RQ4: Can one predict the star rating from the review text? (sentence to star rating classification)

# methodology

- algorithms benchmark: https://paperswithcode.com/dataset/amazon-review

Which dataset(s) did you choose? Why?

- https://amazon-reviews-2023.github.io/
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
        - easier to download from huggingface
        - Larger Dataset: We collected 571.54M reviews, 245.2% larger than the last version;
        - Newer Interactions: Current interactions range from May. 1996 to Sep. 2023;
        - Richer Metadata: More descriptive features in item metadata;
        - Fine-grained Timestamp: Interaction timestamp at the second or finer level;
        - Cleaner Processing: Cleaner item metadata than previous versions;
        - Standard Splitting: Standard data splits to encourage RecSys benchmarking.

How did you clean/transform the data? Why?

How did you solve the problem of missing values? Why?

What questions did you ask of the data? Why were these good questions?

What were the answers to these questions? How did you obtain them? Do the answers make sense?

Were there any difficulties in analysing the data?

What were the key insights obtained?

What are potential biases in the data and analysis?

Which Data Science tools and techniques were learned during this exercise?

How was the work divided up between the members of the group?

# results
