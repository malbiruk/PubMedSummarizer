'''
an attempt to search for info in pubmed using GPT
at first it fetches 10 first relevant articles and reads their abstracts
if they are relevant, it tries to get and read full text
after that it generates an answer to the initial query with links
'''

import argparse
import logging
import logging.config
import os
import re
import time
from datetime import datetime
from typing import List
from urllib.request import urlretrieve

import metapub
import nltk
import textract
import torch
from Bio import Entrez, Medline
from my_api_keys import OPENAI_API_KEY
from openai import OpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           ProgressColumn, SpinnerColumn, Text, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich_argparse import RichHelpFormatter
from scidownl import scihub_download
from sentence_transformers import SentenceTransformer, util

console = Console(soft_wrap=True)

# logging.root.handlers = []


class SpeedColumn(ProgressColumn):
    '''speed column for progress bar'''

    def render(self, task: 'Task') -> Text:
        if task.speed is None:
            return Text('- it/s', style='red')
        return Text(f'{task.speed:.2f} it/s', style='red')


progress_bar = Progress(
    TextColumn('[bold]{task.description}'),
    SpinnerColumn('simpleDots'),
    TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn('|'),
    SpeedColumn(),
    TextColumn('|'),
    TimeElapsedColumn(),
    TextColumn('|'),
    TimeRemainingColumn(),
)

GPT_MODEL = 'gpt-3.5-turbo-0125'
CLIENT = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDER = SentenceTransformer('msmarco-distilbert-base-v4')
# embedder_mpnet = SentenceTransformer('all-mpnet-base-v2')
CURRENT_DATE = datetime.now().strftime('%B %-d, %Y')

PROMPT = f"""
You are a part of the program which gets a search query or question as an \
input, performs optimized search for scientific articles in PubMed, \
retrieves abstracts and/or full texts of relevant articles from the search \
results and returns a brief precise comprehensive answer to the \
initial query/question with the links to sources -- PMIDs of articles, \
where each part of information was taken from.

Each time you will be called, you should perform one of the following actions \
(the action will be specified in the beginning of the prompt):
CREATE_QUERY
IDENTIFY_RELEVANT
WRITE_SUMMARY
CONTINUE_CHAT

Now I will describe each action.

CREATE_QUERY
If a message starts with CREATE_QUERY, user input will follow it. Your task \
will be to convert this input (could be a question, query or keywords) to \
up to 3 optimized queries for PubMed. You can specify publication type \
[PT] (for example, review, systematic review, clicnical trial) or you can specify \
publication date range, if the search will benefit from it or if the user asks so. \
Notice that current date is {CURRENT_DATE} for searches \
like "last n years".
Answer with up to 3 optimized queries separated by newlines.

Example 1:
input: "CREATE_QUERY
Find me Paroxetine clinical trials of the last 10 years"
your answer: "Paroxetine" AND "Clinical Trial"[PT] AND \
("2012/01/01"[PDat] : "2022/12/31"[PDat])

Example 2:
input: "CREATE_QUERY
why dinosaurs extinct"
your answer: "Why did dinosarus become extinct
Dinosaur extinction causes
"

IDENTIFY_RELEVANT
If a message starts with IDENTIFY_RELEVANT, one of the queries you created \
previously will follow it, then the next structure will be passed to you: \
PMID, line break, abstract of the article with this PMID, two line breaks, \
next PMID with abstract and so on.
Your task will be to identify the most relevant articles to the provided \
query by the abstracts. YOUR ANSWER WILL CONTAIN ONLY PMIDs SEPARATED BY SPACES. \
No other text, just:
PMID1 PMID2 PMID3

Example:
input: "IDENTIFY_RELEVANT
Query: Why did dinosarus become extinct

PMID: 30911383
Abstract: Palaeontological deductions from the fossil remnants of extinct dinosaurs tell us much about their classification into species as well as about their physiological and behavioural characteristics. Geological evidence indicates that dinosaurs became extinct at the boundary between the Cretaceous and Paleogene eras, about 66 million years ago, at a time when there was worldwide environmental change resulting from the impact of a large celestial object with the Earth and/or from vast volcanic eruptions. However, apart from the presumption that climate change and interference with food supply contributed to their extinction, no biological mechanism has been suggested to explain why such a diverse range of terrestrial vertebrates ceased to exist. One of perhaps several contributing mechanisms comes by extrapolating from the physiology of the avian descendants of dinosaurs. This raises the possibility that cholecalciferol (vitamin D3) deficiency of developing embryos in dinosaur eggs could have caused their death before hatching, thus extinguishing the entire family of dinosaurs through failure to reproduce.

PMID: 30816905
Abstract: Evolution is both a fact and a theory. Evolution is widely observable in laboratory and natural populations as they change over time. The fact that we need annual flu vaccines is one example of observable evolution. At the same time, evolutionary theory explains more than observations, as the succession on the fossil record. Hence, evolution is also the scientific theory that embodies biology, including all organisms and their characteristics. In this paper, we emphasize why evolution is the most important theory in biology. Evolution explains every biological detail, similar to how history explains many aspects of a current political situation. Only evolution explains the patterns observed in the fossil record. Examples include the succession in the fossil record; we cannot find the easily fossilized mammals before 300 million years ago; after the extinction of the dinosaurs, the fossil record indicates that mammals and birds radiated throughout the planet. Additionally, the fact that we are able to construct fairly consistent phylogenetic trees using distinct genetic markers in the genome is only explained by evolutionary theory. Finally, we show that the processes that drive evolution, both on short and long time scales, are observable facts.

PMID: 34188028
Abstract: The question why non-avian dinosaurs went extinct 66 million years ago (Ma) remains unresolved because of the coarseness of the fossil record. A sudden extinction caused by an asteroid is the most accepted hypothesis but it is debated whether dinosaurs were in decline or not before the impact. We analyse the speciation-extinction dynamics for six key dinosaur families, and find a decline across dinosaurs, where diversification shifted to a declining-diversity pattern ~76 Ma. We investigate the influence of ecological and physical factors, and find that the decline of dinosaurs was likely driven by global climate cooling and herbivorous diversity drop. The latter is likely due to hadrosaurs outcompeting other herbivores. We also estimate that extinction risk is related to species age during the decline, suggesting a lack of evolutionary novelty or adaptation to changing environments. These results support an environmentally driven decline of non-avian dinosaurs well before the asteroid impact."

Your answer: "30911383 34188028"

WRITE_SUMMARY
If a message starts with WRITE_SUMMARY, you will be provided with the initial \
user query, optimized queries, PMIDs, and context chunks or abstracts \
from corresponding articles and cosine similarity scores of that context \
chunks to the queries you generated in one of the previous steps.
The input message will have the following structure:
WRITE_SUMMARY
User query: input query

Optimized query: optimized query 1

PMID: PMID
Context Chunks:
context chunk 1
context chunk 2
...
Cosine Similarity Scores: [score for chunk 1, score for chunk 2, ...]

PMID: PMID
Abstract:
Abstract

PMID: PMID
...

Optimized query: optimized query 2

PMID: PMID
Context Chunks:
context chunk 1
context chunk 2
...
Cosine Similarity Scores: [score for chunk 1, score for chunk 2, ...]

...

Your task will be to briefluy and precisely summarize the information \
provided in these contet chunks and abstracts and answer to the intial user query \
like a scientist writing literature review. You also should provide sources \
i.e. PMIDs of the articles from which you took particluar pieces of information \
for your summary/answer. YOUR SUMMARY SHOULD BE ABOUT 250-300 WORDS. \
USE ONLY INFORMATION PROVIDED IN THE INPUT.
Example of your final answer format:
Although non-avian dinosaurs dominated terrestrial ecosystems until the end-Cretaceous, both a marked increase of extinction and a decrease in their ability to replace extinct species led dinosaurs to decline well before the K/Pg extinction (PMID: 34188028). Even though the latest Cretaceous dinosaur fossil record is geographically dominated by Laurasian taxa, the diversity patterns observed here are based on continent-scale samples that reflect a substantial part of latest Cretaceous dinosaur global diversity. Long-term environmental changes led to restructuring of terrestrial ecosystems that made dinosaurs particularly prone to extinction (PMID: 23112149). These results are also consistent with modelling studies of ecological food-webs (PMID: 22549833) and suggest that loss of key herbivorous dinosaurs would have made terminal Maastrichtian ecosystems—in contrast with ecosystems from earlier in the Late Cretaceous (Campanian)—more susceptible to cascading extinctions by an external forcing mechanism. We propose that a combination of global climate cooling, the diversity of herbivores, and age-dependent extinction had a negative impact on dinosaur extinction in the Late Cretaceous; these factors impeded their recovery from the final catastrophic event (PMID: 34188028).

CONTINUE_CHAT
If a message starts with CONTINUE_CHAT, than just support the dialogue using context you already have.
""".strip()


def initialize_logging(level=logging.INFO, folder: str = '.') -> None:
    '''
    initialize logging (default to file 'out.log' in folder +
    rich to command line)
    '''
    logger_ = logging.getLogger(__name__)
    logger_.setLevel(level)
    filehandler = logging.FileHandler(f'{folder}/out.log')
    filehandler.setFormatter(logging.Formatter())
    filehandler.setLevel(level)
    richhandler = RichHandler(show_path=False)
    richhandler.setFormatter(logging.Formatter())
    richhandler.setLevel(level)
    logger_.addHandler(filehandler)
    logger_.addHandler(richhandler)
    return logger_


logger = initialize_logging()


def search(query: str,
           n_res: int,
           email: str
           ) -> dict:
    '''
    initialize search
    '''
    Entrez.email = email
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=str(n_res),
                            retmode='xml',
                            term=query)
    return Entrez.read(handle)


def fetch_details(id_list: List[int],
                  email: str
                  ) -> dict:
    '''
    fetch data by id from initial search
    '''
    ids = ','.join(id_list)
    Entrez.email = email
    handle = Entrez.efetch(db='pubmed',
                           rettype="medline",
                           retmode="text",
                           id=ids)
    return Medline.parse(handle)


def get_abstracts(query: str,
                  n_res: int = 10,
                  email: str = 'fiyefiyefiye@gmail.com',
                  pmid_list: list = None,
                  reviews: bool = False
                  ) -> dict:
    '''
    get abstracts by search query
    returns dict PMID: abstract
    '''
    query = query + ' AND "review"[PT]' if reviews else query
    if pmid_list is None:
        logger.info('getting abstracts for query "%s"...', query)
        results = search(query, n_res, email)
    id_list = results['IdList'] if pmid_list is None else pmid_list
    papers = fetch_details(id_list, email)
    return {paper.get('PMID'): paper.get('AB') for paper in papers
            if paper.get('PMID') is not None and paper.get('AB') is not None}


def process_article(article_text: str) -> List[str]:
    '''
    split article to sentences and return part between introduction
    or keywords asnd references

    also remove references like (someone et al., 20..) from text

    returns tokenized by sentences article
    '''
    # Define keywords for identifying abstract and references
    references_keywords = ['references', 'bibliography']
    references_keywords = (references_keywords +
                           [word.title() for word in references_keywords]
                           + [word.upper() for word in references_keywords])

    # Tokenize the text into sentences
    # Convert to lowercase for case-insensitive matching
    article_text = article_text.replace('\n', ' ')  # remove newlines
    article_text = re.sub(r'(\s){2,}', ' ', article_text)  # and double spaces
    # remove citations
    article_text = re.sub(r'\([^()\d]*\d[^()]*\)', '', article_text)
    sentences = nltk.sent_tokenize(article_text)

    # Identify and remove references
    references_start = len(sentences)
    for i, sentence in enumerate(sentences[::-1]):
        if any(keyword in sentence for keyword in references_keywords):
            references_start = len(sentences) - i - 1
            # Include the sentence containing the keyword
            break

    # remove short sentences (less than 5 words)
    sentences = [sentence for sentence in sentences
                 if len(nltk.word_tokenize(sentence)) >= 5]

    clean_text = ' '.join(sentences[:references_start])
    return clean_text


def get_article_texts(id_list: List[str]) -> dict:
    '''
    retrieve full text of articles by PMIDs
    '''
    result = {}
    with progress_bar as p:
        for pmid in p.track(id_list,
                            description='downloading articles'):
            if os.path.exists(f'cache/{pmid}.pdf'):
                logger.info('%s already downloaded', pmid)

            else:
                url = metapub.FindIt(pmid).url

                if not url is None:
                    logger.info('downloading %s from PMC', pmid)
                    urlretrieve(url, f'cache/{pmid}.pdf')
                    time.sleep(2)
                else:
                    logger.info('downloading %s from Sci-Hub', pmid)
                    scihub_download(pmid, paper_type='pmid',
                                    out=f'cache/{pmid}.pdf')

            if os.path.exists(f'cache/{pmid}.pdf'):
                article_text = textract.process(
                    f'cache/{pmid}.pdf',
                    extension='pdf',
                    method='pdftotext',
                    encoding="utf_8",
                ).decode()

                result[pmid] = process_article(article_text)
            else:
                result[pmid] = None

    return result


def chunk_text(text, chunk_size=3, overlap=1) -> list:
    '''
    chunk text to chunks size of chunk_size sentences
    which are overlapping by overlap
    '''
    sentences = nltk.sent_tokenize(text)

    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def get_context_from_articles(query: str,
                              embedder,
                              pmid_to_article: dict,
                              n_res: int = 10,
                              chunk_size: int = 3,
                              overlap: int = 1) -> tuple:
    '''
    embed sentences from articles (input: {PMID: article_text})
    (actually embed by chunks of sentences with overlap)
    embed query
    semantic search for top n_res relevant sentences
    return them as strings in list, where key is PMID of corresponding article
    '''
    # chunk my texts
    logger.info('embedding articles and retrieving context...')
    pmid_to_embeddings = {}

    for pmid, v in pmid_to_article.items():
        if os.path.exists(f'cache/{pmid}.pt'):
            logger.info('%s already embedded', pmid)
            pmid_to_embeddings[pmid] = torch.load(f'cache/{pmid}.pt')
        else:
            if v is not None:
                pmid_to_embeddings[pmid] = embedder.encode(
                    chunk_text(v, chunk_size, overlap),
                    convert_to_tensor=True)
                torch.save(pmid_to_embeddings[pmid], f'cache/{pmid}.pt')
            else:
                pmid_to_embeddings[pmid] = v

    pmid_to_chunked = {k: chunk_text(v, chunk_size, overlap)
                       if v is not None else v
                       for k, v in pmid_to_article.items()}

    # pmid_to_embeddings = {
    #     k: embedder.encode(v, convert_to_tensor=True)
    #     if v is not None else v
    #     for k, v in pmid_to_chunked.items()
    # }

    query_embedding = embedder.encode(query, convert_to_tensor=True)

    pmid_to_context = {}
    pmid_to_cos = {}

    for k, v in pmid_to_embeddings.items():
        if v is not None:
            hits = util.semantic_search(query_embedding,
                                        v,
                                        top_k=n_res)
            pmid_to_context[k] = [
                pmid_to_chunked[k][hit['corpus_id']] for hit in hits[0]]
            pmid_to_cos[k] = [hit['score'] for hit in hits[0]]
        else:
            pmid_to_context[k] = None
            pmid_to_cos[k] = None
    return pmid_to_context, pmid_to_cos


def chat_completion_request(messages,
                            model=GPT_MODEL,
                            temperature=.7):
    '''
    gpt request function
    '''
    try:
        completion = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error("Unable to generate ChatCompletion response")
        logger.error("Exception: %s", e)


def gpt_process_query(messages: list,
                      user_query: str,
                      model=GPT_MODEL,
                      temperature=.7) -> List[dict]:
    '''
    optimize user query to multiple searches
    '''
    prompt = f'CREATE_QUERY\n{user_query}'
    logger.info('optimizing query %s...', user_query)
    messages.append({'role': 'user', 'content': prompt})
    answer = chat_completion_request(messages, model, temperature)
    messages.append({'role': 'assistant', 'content': answer})
    logger.info('new queries: %s', answer)
    return answer.split('\n')


def gpt_identify_relevant(messages: list,
                          query: str,
                          pmid_to_abstract: dict,
                          model=GPT_MODEL,
                          temperature=.7) -> list:
    '''
    identify relevant articles to the query by abstracts
    '''
    logger.info('identifying relevant articles for query: "%s"...', query)
    prompt = f'IDENTIFY_RELEVANT\nQuery: {query}\n\n'
    for k, v in pmid_to_abstract.items():
        prompt += f'PMID: {k}\n'
        prompt += f'Abstract: {v}\n\n'
    messages.append({'role': 'user', 'content': prompt})
    answer = chat_completion_request(messages, model, temperature)
    messages.append({'role': 'assistant', 'content': answer})
    logger.info('relevant articles: "%s"', answer)
    return answer.split() if answer is not None else []


def gpt_generate_summary(messages: list,
                         user_query: str,
                         query_to_context: dict,
                         model=GPT_MODEL,
                         temperature=.7) -> tuple:
    '''
    summarize info from contexts and different queries
    '''
    logger.info('summarizing info: "%s"...', user_query)
    prompt = f'WRITE_SUMMARY\nUser query: {user_query}\n\n'

    for k, v in query_to_context.items():
        prompt += f'Optimized query: {k}\n\n'
        for pmid, cont in v[0].items():
            prompt += f'PMID: {pmid}\n'
            if isinstance(cont, list):
                prompt += 'Context Chunks:\n'
                prompt += '\n'.join(cont)
                prompt += f'\nCosine Similarity Scores: {v[1][pmid]}\n\n'
            else:
                prompt += f'Abstract:\n{cont}\n\n'

    # available context space
    av_cont_space = int(16300 * 4 - (len(PROMPT) + 300) / 4)
    if len(prompt) > av_cont_space:
        prompt = prompt[:av_cont_space]
        logger.warning('Truncated prompt in order to fit in context window.')

    messages = [messages[0]]
    messages.append({'role': 'user', 'content': prompt})
    answer = chat_completion_request(messages, model, temperature)
    messages.append({'role': 'assistant', 'content': answer})
    # logger.info('got summary: "%s"', answer)
    return answer, messages


def extract_terms(pubmed_query):
    '''
    extract search terms from pubmed query
    '''
    # Define a pattern to match the text within double quotes and parentheses
    pattern = re.compile(r'"([^"]+)"|\(([^)]+)\)|(\S+)')

    # Find all matches in the PubMed query
    matches = pattern.findall(pubmed_query)

    # Flatten the matches and filter out empty strings
    text_terms = [group[0] if group[0] else group[1] if group[1] else group[2]
                  for group in matches if any(group)]

    # Join the text terms into a single string
    result = ' '.join(text_terms)

    result = (result
              .replace(' AND ', ' ')
              .replace(' : ', '')
              .replace(' OR ', ' ')
              )
    pattern = re.compile(r'"[^"]+"\[[^\]]+\]')

    result = re.sub(pattern, '', result).strip()

    return result


def messages_to_human_readable(messages: list) -> str:
    '''
    get only content of messages without system prompt and with nice separators
    '''
    return '\n\n----------\n\n'.join(
        [i['content'].strip() for i in messages[1:]])


def gpt_continue_chat(messages: list,
                      model=GPT_MODEL,
                      temperature=.7) -> list:
    '''
    just support the dialogue using input() from user
    '''
    try:
        while True:
            contents = []
            line = console.input("[bold magenta]User:[/] ")
            contents.append(line)
            prompt = 'CONTINUE_CHAT\n' + '\n'.join(contents)
            messages.append({'role': 'user', 'content': prompt})
            console.print('\ngetting response...', style='yellow')
            answer = chat_completion_request(messages, model, temperature)
            console.print(f'\n[bold green]GPT:[/] {answer}\n')
            messages.append({'role': 'assistant', 'content': answer})
    except KeyboardInterrupt:
        console.print('\nExiting...', style='yellow')
        return messages


def main(user_query: str,
         pmid_list: list = None,
         reviews: bool = False) -> None:
    '''
    Search and summarize info from PubMed using GPT.
    '''
    if not os.path.exists('cache'):
        os.makedirs('cache')

    messages = [{'role': 'system', 'content': PROMPT}]
    optimized_queries = gpt_process_query(messages, user_query)
    query_to_context = {}

    if pmid_list is not None:
        pmid_list = [str(i) for i in pmid_list]

    for q in optimized_queries:
        abstracts = get_abstracts(q,
                                  n_res=10,
                                  email='2601074@gmail.com',
                                  pmid_list=pmid_list,
                                  reviews=reviews)
        if not abstracts:
            continue

        q = q.replace('"', '')
        query = extract_terms(q)

        relevant_pmids = (gpt_identify_relevant(messages, query, abstracts)
                          if pmid_list is None else pmid_list)

        if not relevant_pmids:
            continue

        articles = get_article_texts(relevant_pmids)
        context_chunks, cosine_similarity_scores = (
            get_context_from_articles(query,
                                      EMBEDDER,
                                      articles,
                                      n_res=5,
                                      chunk_size=6,
                                      overlap=2))
        contexts = {k: abstracts[k] if v is None else v
                    for k, v in context_chunks.items()}
        query_to_context[query] = (contexts, cosine_similarity_scores)

    with open('gpt_messages.txt', 'w', encoding='utf-8') as f:
        f.write(messages_to_human_readable(messages)
                + '\n\n----------\n\n')

    gpt_summary, messages = gpt_generate_summary(messages,
                                                 user_query,
                                                 query_to_context)

    with open('gpt_summary.txt', 'w', encoding='utf-8') as f:
        f.write(gpt_summary)

    console.print()
    console.print(f'[bold green]GPT:[/] {gpt_summary}')
    console.print()
    messages = gpt_continue_chat(messages)
    with open('gpt_messages.txt', 'a', encoding='utf-8') as f:
        f.write(messages_to_human_readable(messages))


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search and summarize info from PubMed using GPT.',
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("user_query", type=str, help="User query string")
    parser.add_argument("--pmid_list", nargs="*", type=int, default=None,
                        help="List of PMIDs")
    parser.add_argument('--reviews', action='store_true',
                        help='find only review articles')
    args = parser.parse_args()
    main(**vars(args))
