---
title: "Report: Amazon Reviews'23"
subtitle: "Code: [`github.com/sueszli/amazeballs/`](https://github.com/sueszli/amazeballs/)"
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
    - \posttitle{\end{center}\vspace{-70pt}}    
    # content
    - \usepackage{scrextend}
    - \changefontsizes{8pt}
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

this is just as much about the "data science process" as it is about the results.

this is an open ended task.

we're free to pick whatever dataset we like, and modify questions with supervisor's approval.

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

#### Motivation

Understanding online reviews isn't just about interpreting customer opinions; it's a window into how people perceive and interact with products across diverse categories. The Amazon Dataset'23 offers a treasure trove of insights, allowing us to explore patterns in sentiment, subjectivity and the elements that matter most in consumer decision-making. By digging into this data, we aim to uncover the subtle relationships between what customers say and how they rate products, shedding light on the dynamics of trust, satisfaction and expectation in the digital marketplace.

Beyond the findings, this report highlights the (data science) process of turning raw, unstructured data into actionable knowledge. Through techniques like sentiment analysis, topic modeling and classification, we're not just addressing key questions about product reviews – we're also demonstrating the iterative, hands-on nature of data science itself.

#### Process

The process which we followed, is more formally known as CRISP-DM (Cross-Industry Standard Process for Data Mining). It begins with (1) business understanding, where we refine the research questions in consultation with a supervisor for our project, define variables and metrics and build hypotheses while being mindful of biases. Next, we move to (2) data understanding, where we sample and preprocess the data, ensuring privacy and assess the accuracy, biases and reliability of the measurements. In (3) data preparation, we clean the data by checking for missing values, outliers and inconsistencies, calculating descriptive statistics and transforming the data as needed. If the data is insufficient to answer the research questions, we may combine columns, look for additional datasets, or modify the questions. In (4) modeling, we calculate correlations and build models to explore the relationships between variables. During (5) evaluation, we plot the data, identify patterns and anomalies, visualize the findings and check predictions to assess if the models answer the original questions. Finally, (6) deployment involves using the results to make decisions or share insights with stakeholders.

# Methodology

#### Research Questions

First we define the research questions that we aim to answer. Our team has selected task 21 from the list provided by the course team and did not further modify it. The research questions are as follows:

- RQ1: Are reviews for some categories of product on Amazon overall more positive than for other categories?
- RQ2: Are reviews more subjective for some classes of products than for others?
- RQ3: Which aspects of different classes of products are the most important in the reviews?
- RQ4: Can one predict the star rating from the review text?

The first research question is a comparison of sentiment across categories, the second is a comparison of subjectivity across categories, the third is a topic modeling task, commonly referred to as aspect-based sentiment analysis and the fourth is a classification task to predict the star rating from the review text.

<!-- Which dataset(s) did you choose? Why? -->

#### Dataset Selection

To answer these questions, we chose the [Amazon Reviews'23 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) which is the standard dataset for the Amazon product reviews in the RecSys and NLP communities. This dataset is a collection of 571.54M reviews, 245.2% larger than the last version, with interactions ranging from May 1996 to September 2023. It includes richer metadata, fine-grained timestamps and cleaner processing, making it an ideal choice for our analysis. We decided to choose it 

- https://amazon-reviews-2023.github.io/
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
        - easier to download from huggingface
        - Larger Dataset: We collected 571.54M reviews, 245.2% larger than the last version;
        - Newer Interactions: Current interactions range from May. 1996 to Sep. 2023;
        - Richer Metadata: More descriptive features in item metadata;
        - Fine-grained Timestamp: Interaction timestamp at the second or finer level;
        - Cleaner Processing: Cleaner item metadata than previous versions;
        - Standard Splitting: Standard data splits to encourage RecSys benchmarking.

<!-- How did you clean/transform the data? Why? -->

- sample because too large
        - 100,000 samples per category (2.92 GB): doesn't fit in memory for plotting
        - 10,000 samples per category (0.33 GB): inference would take 8 days (339880 items with 2it/s)
        - 1,000 samples per category (0.03 GB): inference would take 19 hours (33994 items with 2it/s)
        - 100 samples per category (<0.00 GB): inference would take 2 hours (3399 items with 2it/s) — this is what we used
- lots of languages, so models had to be multilingual
        - some of them were, others weren't

<!-- How did you solve the problem of missing values? Why? -->

<!-- What questions did you ask of the data? Why were these good questions? -->

<!-- What were the answers to these questions? How did you obtain them? Do the answers make sense? -->

<!-- Were there any difficulties in analysing the data? -->

<!-- What were the key insights obtained? -->

<!-- What are potential biases in the data and analysis? -->

<!-- Which Data Science tools and techniques were learned during this exercise? -->

<!-- How was the work divided up between the members of the group? -->

# Findings
