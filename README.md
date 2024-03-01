# PubMedSummarizer
## Purpose
Automatically search for information in PubMed, obtain articles and/or abstracts and generate summary with links to information sources from them using GPT-3.5.

## Pipeline
1. User inputs their query (optional: and list of PMIDs using `--pmid_list` flag, in that case the searching part is skipped)
2. GPT-3.5 processes it to make up to 3 optimized queries for PubMed
3. The program passes this optimal queries to PubMed search and returns abstracts of top 10 results
4. GPT-3.5 "reads" these abstracts, picks only relevant ones and returns their PMIDs
5. The program finds full articles of these relevant PMIDs in PMC and Sci-Hub if not found in PMC
6. The program tokenizes articles by sentences and embeds it as well as query provided in the 2nd point and performs semantic search, returning top 5 found context chunks with their score (cosine similarity) from each of the articles. Also if can't find/download article, it just returns the abstract of corresponding article.
7. GPT-3.5 gets all these article chunks, cosine similarity scores, and abstracts along with their PMIDs and in response generates brief summary with relevant information (obtained only from these abstracts and articles), answering to the initial query (point 1) and provides corresponding PMIDs for each piece of information it writes in the answer.
8. After that GPT works just as a chat-bot with all these context

## Examples
### AI applications in medicine
#### Input
`python PubMedSummarizer.py "AI applications in medicine"`
#### Optimized query
`"Artificial Intelligence AND Medicine"`
#### Abstracts fetched from PubMed
[here](example/abstracts.txt)
#### Relevant articles
`33811123 31939856 32250084 33509373`
#### Context chunks from articles
[here](example/context_chunks.txt)
#### Output

"Artificial intelligence (AI) is revolutionizing cancer research and personalized clinical care by leveraging high-dimensional datasets and innovative deep learning architectures. This advancement in AI has led to applications in detecting and classifying cancer, molecular characterization of tumors, drug discovery, repurposing, and predicting treatment outcomes (PMID: 33811123). The field of anesthesiology stands to benefit from AI through applications in perioperative and intensive care, pain management, and drug delivery and discovery. AI technologies have the potential to augment clinical judgment and skills, improving clinical endpoints like depth of anesthesia (PMID: 31939856).

Moreover, in forensic medicine, the integration of AI technology has shown feasibility in various identification tasks such as face recognition, DNA analysis, postmortem interval estimation, and injury identification. This approach not only brings new vitality to forensic medicine but also poses new challenges that need to be addressed for the collaborative development of "artificial intelligence plus forensic medicine" in the era of big data (PMID: 32250084).

In nuclear medicine imaging, AI and machine learning approaches are increasingly utilized for automated image reading, correlation with clinical outcomes, and technological applications in image processing. These methods aid in tumor imaging, quantification of tumor volume, segmentation, and classification of pathological foci, ultimately improving clinical decision-making and patient prognosis (PMID: 33509373).
Overall, the integration of AI in medicine showcases promising advancements in cancer research, anesthesiology, forensic medicine, and nuclear medicine imaging, indicating a transformative shift towards more efficient, precise, and personalized healthcare practices."

### digital mental health interventions
#### Input
`python PubMedSummarizer.py "digital mental health interventions"`
#### Intermediate steps
[here](example/mental_health.txt)
#### Output
"Digital mental health interventions, including those delivered via the web and apps, have shown promise in improving access to mental health treatment for various populations, such as college students and caregivers of older adults (PMID: 31333198, 32411643). These interventions have been effective in addressing common mental health issues like depression, anxiety, stress, and improving psychological well-being (PMID: 31333198, 37001253). The use of digital tools, such as cognitive behavioral therapy delivered online, has demonstrated positive outcomes in reducing symptoms and improving satisfaction with life (PMID: 37001253).

Additionally, the effectiveness of digital mental health interventions for adolescents and young people has been explored, showing potential in addressing mental health disorders early in life and reaching underserved clinical populations (PMID: 33913817). Computerized cognitive behavioral therapy has been effective in managing anxiety and depression in this population, emphasizing the importance of interventions with an in-person element or professional involvement for greater effectiveness (PMID: 33913817).

While digital mental health interventions offer benefits such as increased accessibility, reduced stigma, and improved user engagement, there are also challenges to address, including concerns about privacy, developer credibility, and the need for more rigorous studies to confirm their effectiveness and cost-effectiveness, especially in low-resource settings (PMID: 31333198, 35612165, 33913817). Continued research and development in this field are necessary to optimize the user experience, assess the long-term effectiveness, and ensure sustainable implementation of digital mental health interventions across diverse populations (PMID: 31333198, 35612165, 33913817)."
