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
from config import EMBEDDING_MODEL, GPT_MODEL, PROMPT, TEMPERATURE
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

TOKENS_USED = {"completion_tokens": 0,
               "prompt_tokens": 0,
               "total_tokens": 0}

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


CLIENT = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
# embedder_mpnet = SentenceTransformer('all-mpnet-base-v2')


def initialize_logging(level=logging.INFO, folder: str = '.') -> None:
    '''
    initialize logging (default to file 'out.log' in folder +
    rich to command line)
    '''
    logger_ = logging.getLogger(__name__)
    logger_.setLevel(level)
    filehandler = logging.FileHandler(f'{folder}/out.log')
    filehandler.setLevel(level)
    richhandler = RichHandler(show_path=False)
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
                  reviews: bool = False,
                  pub_types: list = None,
                  from_year: int = None,
                  ) -> dict:
    '''
    get abstracts by search query
    returns dict PMID: abstract
    '''
    query = query + ' AND "review"[PT]' if reviews else query
    if pub_types:
        query = (query
                 + ' AND ('
                 + ' OR '.join(f'"{i}"[PT]' for i in pub_types)
                 + ')')
    if from_year:
        current_date = datetime.now().strftime('%Y/%m/%d')
        query = (query
                 + f' AND ("{from_year}/01/01"[PDat] : "{current_date}"[PDat])')

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
        emb_file_name = (
            f'cache/{pmid}-{EMBEDDING_MODEL.rsplit("/", maxsplit=1)[-1]}'
            f'-{chunk_size}-{overlap}.pt')
        if os.path.exists(emb_file_name):
            logger.info('%s already embedded', pmid)
            pmid_to_embeddings[pmid] = torch.load(emb_file_name)
        else:
            if v is not None:
                pmid_to_embeddings[pmid] = embedder.encode(
                    chunk_text(v, chunk_size, overlap),
                    convert_to_tensor=True)
                torch.save(pmid_to_embeddings[pmid], emb_file_name)
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
                            temperature=TEMPERATURE,
                            stream=False):
    '''
    gpt request function
    '''
    try:
        completion = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream
        )
        if stream:
            def iter_func():
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return iter_func()

        for key, value in dict(completion.usage).items():
            if key in TOKENS_USED:
                TOKENS_USED[key] += value

        return completion.choices[0].message.content
    except Exception as e:
        logger.error("Unable to generate ChatCompletion response")
        logger.error("Exception: %s", e)


def gpt_process_query(messages: list,
                      user_query: str,
                      model=GPT_MODEL,
                      temperature=TEMPERATURE) -> List[dict]:
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
                          temperature=TEMPERATURE) -> list:
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
                         temperature=TEMPERATURE,
                         prompt=PROMPT) -> tuple:
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
    max_tokens = 16385 if GPT_MODEL.startswith('gpt-3.5') else 128000

    av_cont_space = int(16300 * 4 - (len(prompt) + 300) / 4)
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


def tokens_to_prices(tokens: dict) -> dict:
    '''
    convert n_tokens to prices
    prices last updated on Mar 4, 2024
    '''
    final_price = {}
    prices = {'gpt-3.5-turbo-0125': {'input': .5, 'output': 1.5},
              'gpt-4': {'input': 30, 'output': 60}}
    if prices.get(GPT_MODEL):
        final_price['output'] = tokens['completion_tokens'] / \
            1000000 * prices[GPT_MODEL]['output']
        final_price['input'] = tokens['prompt_tokens'] / \
            1000000 * prices[GPT_MODEL]['input']
        final_price['total'] = final_price['output'] + \
            final_price['input']

        return final_price
    return


def gpt_continue_chat(messages: list,
                      model=GPT_MODEL,
                      temperature=TEMPERATURE,
                      show_money: bool = False) -> list:
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
        console.print('\nExiting...\n', style='yellow')

        if show_money:
            console.print(f'Tokens used: {TOKENS_USED}', style='yellow')
            prices = tokens_to_prices(TOKENS_USED)
            if prices:
                console.print(f'$ spent: {prices}',
                              style='yellow')
        return messages


def main(user_query: str,
         pmid_list: list = None,
         reviews: bool = False,
         show_money: bool = False,
         n_articles: int = 10,
         n_chunks: int = 5,
         chunk_size: int = 6,
         overlap: int = 2,
         email: str = 'fiyefiyefiye@gmail.com'
         ) -> None:
    '''
    Search and summarize info from PubMed using GPT.
    '''
    if not os.path.exists('cache'):
        os.makedirs('cache')
    current_date = datetime.now().strftime('%Y/%m/%d')
    prompt = PROMPT.replace('{N_QUERIES}', '3').replace(
        '{CURRENT_DATE}', current_date)
    messages = [{'role': 'system', 'content': prompt}]
    optimized_queries = gpt_process_query(messages, user_query)
    query_to_context = {}

    if pmid_list is not None:
        pmid_list = [str(i) for i in pmid_list]

    for q in optimized_queries:
        abstracts = get_abstracts(q,
                                  n_res=n_articles,
                                  email=email,
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
                                      n_res=n_chunks,
                                      chunk_size=chunk_size,
                                      overlap=overlap))
        contexts = {k: abstracts[k] if v is None else v
                    for k, v in context_chunks.items()}
        query_to_context[query] = (contexts, cosine_similarity_scores)

    with open('gpt_messages.txt', 'w', encoding='utf-8') as f:
        f.write(messages_to_human_readable(messages)
                + '\n\n----------\n\n')

    gpt_summary, messages = gpt_generate_summary(messages,
                                                 user_query,
                                                 query_to_context,
                                                 prompt=prompt)

    with open('gpt_summary.txt', 'w', encoding='utf-8') as f:
        f.write(gpt_summary)

    console.print()
    console.print(f'[bold green]GPT:[/] {gpt_summary}')
    console.print()
    messages = gpt_continue_chat(messages, show_money)
    with open('gpt_messages.txt', 'a', encoding='utf-8') as f:
        f.write(messages_to_human_readable(messages))


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search and summarize info from PubMed using GPT.',
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("user_query", type=str, help="User query string")
    parser.add_argument("-p", "--pmid_list", nargs="*", type=int, default=None,
                        help="List of PMIDs")
    parser.add_argument('-r', '--reviews', action='store_true',
                        help='find only review articles')
    parser.add_argument('-m', '--show_money', action='store_true',
                        help='show tokens and $ usage')
    parser.add_argument(
        '-a', '--n_articles', type=int, default=10,
        help='retrieve top {n} articles from PubMed per query (default=10)')
    parser.add_argument(
        '-n', '--n_chunks', type=int, default=5,
        help='pass top {n} chunks from semantic search in article to context '
        '(default=5)')
    parser.add_argument('-с', '--chunk_size', type=int, default=6,
                        help='context chunk - {n} sentences (default=6)')
    parser.add_argument(
        '-o', '--overlap', type=int, default=2,
        help='overlap between chunks (should be 20-30%% of chunk size) '
        '(default=2)')
    parser.add_argument(
        '-e', '--email', type=str, default='fiyefiyefiye@gmail.com',
        help='email address for Entrez calls')

    args = parser.parse_args()
    main(**vars(args))
