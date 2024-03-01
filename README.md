# PubMedSummarizer
## Purpose
Automatically search for information in PubMed, obtain articles and/or abstracts and generate summary with links to information sources from them using GPT-3.5.

## Pipeline
1. User inputs their query (optional: and list of PMIDs using `--pmid_list` flag, in that case the searching part is skipped)
2. GPT-3.5 processes it to make up to 3 optimized queries for PubMed
3. The program passes this optimal queries to PubMed search and returns top 10 results
4. GPT-3.5 "reads" this abstracts, picks only relevant ones and returns their PMIDs
5. The program finds full articles of these  PMIDs provided by GPT-3.5 in PMC and Sci-Hub if not found in PMC
6. The program tokenizes articles by sentences and embeds it as well as query provided in the 2nd point and performs semantic search, returning top 5 found context chunks with their score (cosine similarity) from each of the articles. Also if can't find/download article, it just returns the abstract of corresponding article.
7. GPT-3.5 gets all these article chunks, cosine similarity scores, and abstracts along with their PMIDs and in response generates brief summary with relevant information (obtained only from these abstracts and articles), answering to the initial query (point 1) and provides corresponding PMIDs for each piece of information it writes in the answer.
