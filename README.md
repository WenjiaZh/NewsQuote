# NewsQuote
This repository is configured for paper: 
[**Newsquote: A Dataset Built on Quote Extraction and Attribution for Expert Recommendation**](https://arxiv.org/abs/2305.04825)

Wenjia Zhang, Lin Gui, Rob Procter, Yulan He

To enhance the ability to find credible evidence in news articles, we propose a novel task of expert recommendation, which aims to identify trustworthy experts on a specific news topic. To achieve the aim, we describe the construction of a novel NewsQuote dataset consisting of 24,031 quote-speaker pairs that appeared on a COVID-19 news corpus. We demonstrate an automatic pipeline for speaker and quote extraction via a BERT-based Question Answering model. Then, we formulate expert recommendations as document retrieval task by retrieving relevant quotes first as an intermediate step for expert identification, and expert retrieval by directly retrieving sources based on the probability of a query conditional on a candidate expert. Experimental results on NewsQuote show that document retrieval is more effective in identifying relevant experts for a given news topic compared to expert retrieval.


[Plot(a) describes the QA pipeline, the sequence labelling and the Rule-based Quota Annotator used for quote-source extraction.](quoteextract.pdf)

<img src="https://github.com/WenjiaZh/NewsQuote/blob/main/expranking.pdf" width="500">
 Plot(b) introduces the document retrieval approach for expert recommendation, and plot(c) presents the expert retrieval approach for expert recommendation.

# Code
We do not release the data directly due to the restrictions imposed by the data owner, but will release the scripts for data collection and pre-processing.
The COVID-19 news corpus is from: https://aylien.com/resources/datasets/coronavirus-dataset 







