'''
this is a web-interface for PubMedSummarizer.py
should be run with `streamlit run web_interface.py`
'''

import re
from dataclasses import make_dataclass
from datetime import datetime
from typing import TypeVar

import streamlit as st
from sentence_transformers import SentenceTransformer

import config
from PubMedSummarizer import (chat_completion_request, extract_terms,
                              get_abstracts, get_article_texts,
                              get_context_from_articles, gpt_generate_summary,
                              gpt_identify_relevant, gpt_process_query,
                              initialize_cache, messages_to_human_readable)

Settings = TypeVar('Settings')


def customize_page_appearance() -> None:
    '''
    remove streamlit's red bar and "deploy" button at the top
    '''
    st.markdown('''
        <style>
            [data-testid='stDecoration'] {
                display: none;
            }
            .stDeployButton {
                visibility: hidden;
            }
        </style>
        ''', unsafe_allow_html=True
                )


def click_button() -> None:
    # pylint: disable=missing-function-docstring
    st.session_state.clicked = True


def convert_pmids_to_links(text: str, markdown: bool = True) -> str:
    '''
    convert 7 or 8 digits words to links to pubmed (markdown format)
    '''
    pattern = r'(?<!\[)(?<!\d)(\d{7,8})(?!\d)(?!\])'
    if markdown:
        replacement = r'[\1](https://pubmed.ncbi.nlm.nih.gov/\1/)'
    else:
        replacement = r'https://pubmed.ncbi.nlm.nih.gov/\1/'
    result = re.sub(pattern, replacement, text)
    return result


def sidebar() -> object:
    '''
    creates sidebar with settings, returns settings as a dataclass
    (basically a dict, where keys are attributes)
    '''
    # pylint: disable=too-many-locals
    with st.sidebar:
        # st.title('Settings')

        st.title('Search settings')

        # initialize variables
        from_year = None
        pub_types = None
        pmid_list = None
        email = 'fiyefiyefiye@gmail.com'
        embedder_name = None
        n_chunks = None
        chunk_size = None
        overlap = None
        filter_by_year = False
        filter_by_pub_type = False

        use_full_article_texts = st.toggle(
            'Use full articles\' texts',
            help='By default only articles\' abstracts are used. '
            'This option implies downloading and embedding articles on-the-fly'
            ' followed by semantic search. '
            'It may provide more acccurate results and context, but '
            'keep in mind that it may take more than 10 minutes per run '
            '(depends on n queries and n articles). For limited list of '
            'articles it may be quite useful.',
        )
        use_pmids_list = st.toggle('Provide a list of PMIDs')
        if use_pmids_list:
            pmid_list = st.text_input(
                'List of PMIDs (separated by spaces)').split()
            n_articles = None

        n_queries = st.slider('Generate ___n___ optimized queries '
                              'from input query',
                              1, 10, 3)

        if not use_pmids_list:
            n_articles = st.slider('Retrieve top ___n___ articles per query',
                                   5, 60, 20, 5)
            filter_by_year = st.toggle('Filter by publication date')
            if filter_by_year:
                from_year = st.slider(
                    'Start year',
                    1990, datetime.now().year, datetime.now().year - 10,
                    disabled=not filter_by_year)
            filter_by_pub_type = st.toggle('Filter by publication type')
            if filter_by_pub_type:
                all_pub_types = [
                    'Review', 'Systematic Review',
                    'Clinical Trial', 'Meta-Analysis']
                pub_types = []
                for pub_type in all_pub_types:
                    if st.checkbox(pub_type):
                        pub_types.append(pub_type)
        # email = st.text_input('Email for Entrez', 'fiyefiyefiye@gmail.com')

        if use_full_article_texts:
            st.header(
                'Embedding',
                help='When full articles\' texts are used, '
                'they are split into chunks consisting '
                'of several sentences. After that, these chunks are '
                'embedded, cosine similarity search '
                'by input query is performed, '
                'and the model gets top n chunks as a context '
                'from each article.')
            embedder_name = st.selectbox(
                'Embedding model',
                ['dmis-lab/biobert-base-cased-v1.2',
                 'msmarco-distilbert-base-v4'])

            n_chunks = st.slider('Retrieve top ___n___ chunks per article',
                                 1, 20, 5)
            chunk_size = st.slider('Chunk size, sentences',
                                   1, 20, 6)
            overlap = st.slider('Chunks overlap, %', 0, 50, 30, 5)
            overlap = round(overlap / 100 * chunk_size)

    st.header('Model', anchor=False)
    col1, col2 = st.columns([1,1])
    with col1:
        model_name = st.selectbox('GPT model',
                                  ['gpt-3.5-turbo (16k)',
                                   # 'gpt-4-32k (32k)',
                                   # 'gpt-4 (8k)',
                                   'gpt-4-turbo-preview (128k)',
                                   ])

    model_name = model_name.split(maxsplit=1)[0]

    with col2:
    # if st.toggle('Modify temperature (defaul=0.2)'):
        temperature = st.number_input('Temperature', 0., 1., .2, .1)

    prompt = config.PROMPT
    if st.toggle('Edit system prompt'):
        prompt = st.text_area('System prompt', prompt, height=300)

    settings = {
        'from_year': from_year,
        'pub_types': pub_types,
        'use_pmids_list': use_pmids_list,
        'pmid_list': pmid_list,
        'email': email,
        'use_full_article_texts': use_full_article_texts,
        'n_articles': n_articles,
        'n_queries': n_queries,
        'filter_by_year': filter_by_year,
        'filter_by_pub_type': filter_by_pub_type,
        'model_name': model_name,
        'temperature': temperature,
        'prompt': prompt,
        'embedder_name': embedder_name,
        'n_chunks': n_chunks,
        'chunk_size': chunk_size,
        'overlap': overlap,
    }
    settings_class = make_dataclass('Settings', settings.keys())
    return settings_class(**settings)


def initialize_session_state() -> None:
    '''
    create default attributes in session state
    '''
    if "gpt_summary" not in st.session_state:
        st.session_state.gpt_summary = None

    if 'opt_queries_to_pmids' not in st.session_state:
        st.session_state.opt_queries_to_pmids = {}

    if 'truncated' not in st.session_state:
        st.session_state.truncated = False

    # starts search and analysis of publications
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    # is needed to run search on user_query change
    if 'prev_query' not in st.session_state:
        st.session_state.prev_query = ''

    # messages which are displayed in chat box
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # messages which are used by GPT to engage in conversation
    if 'messages_with_context' not in st.session_state:
        st.session_state.messages_with_context = []

    # messages and responses during search and analysis
    if 'script_messages' not in st.session_state:
        st.session_state.script_messages = []

    # the reason why script_messages and messages_with_context are separate
    # is because in gpt_summary() all previous messages are deleted,
    # and model is provided only with system prompt and obtained context
    # (to use context window more effectively).


def publications_search_and_analysis(session_state: st.session_state,
                                     user_query: str,
                                     settings: Settings,
                                     ) -> None:
    '''
    runs publications search, analysis, and summary generation
    '''
    session_state.prev_query = user_query
    session_state.opt_queries_to_pmids = {}
    session_state.truncated = False
    truncated_1 = False
    truncated_2 = False

    with st.status('Optimizing query...') as status:
        st.write('Optimizing query...')
        if settings.use_full_article_texts:
            embedder = SentenceTransformer(settings.embedder_name)
        current_date = datetime.now().strftime('%Y/%m/%d')
        updated_prompt = settings.prompt.replace(
            '{N_QUERIES}', str(settings.n_queries)).replace(
            '{CURRENT_DATE}', current_date)
        messages = [{'role': 'system', 'content': updated_prompt}]
        optimized_queries = gpt_process_query(
            messages, user_query,
            model=settings.model_name,
            temperature=settings.temperature)
        query_to_context = {}

        if settings.use_pmids_list:
            pmid_list = [str(i) for i in settings.pmid_list]
            status.update(label='Downloading abstracts...')
            st.write('Downloading abstracts...')

        all_pmids = []  # ensure unique pmids only between queries
        for q in optimized_queries:
            session_state.opt_queries_to_pmids[
                q.strip()] = None
            if not settings.use_pmids_list:
                status.update(label=f'Searching articles by query {q}...')
                st.write(f'Searching articles by query {q}...')

            abstracts = get_abstracts(
                q,
                n_res=settings.n_articles,
                email=settings.email,
                pmid_list=pmid_list if settings.use_pmids_list else None,
                from_year=settings.from_year,
                pub_types=settings.pub_types
            )
            if not abstracts:
                st.warning(f'No results found for query "{q}"')
                continue

            q = q.strip()
            query = extract_terms(q.replace('"', ''))

            if settings.use_pmids_list:
                status.update(label=f'Applying query {query}...')
                st.write(f'Applying query {query}...')
            else:
                status.update(label='Getting relevant PMIDs...')
                st.write('Getting relevant PMIDs...')

            relevant_pmids, truncated_1 = (
                pmid_list if settings.use_pmids_list
                else gpt_identify_relevant(
                    messages,
                    query,
                    abstracts,
                    settings.model_name,
                    settings.temperature))

            if relevant_pmids:
                relevant_pmids = [i for i in relevant_pmids
                                  if i not in all_pmids]
                session_state.opt_queries_to_pmids[q] = [
                    int(pmid)
                    for pmid in relevant_pmids]
                all_pmids.extend(relevant_pmids)
            else:
                st.warning('No relevant articles were selected'
                           f' for query "{q}"')
                continue

            if settings.use_full_article_texts:
                status.update(label='Downloading articles\' texts...')
                st.write('Downloading articles\' texts...')
                articles = get_article_texts(relevant_pmids)

                status.update(
                    label='Embedding articles\' texts and '
                    'performing semantic search...')
                st.write('Embedding articles\' texts and '
                         'performing semantic search...')
                context_chunks, cosine_similarity_scores = (
                    get_context_from_articles(
                        query,
                        embedder,
                        articles,
                        n_res=settings.n_chunks,
                        chunk_size=settings.chunk_size,
                        overlap=settings.overlap))
                contexts = {k: abstracts[k] if v is None else v
                            for k, v in context_chunks.items()}
                query_to_context[query] = (contexts,
                                           cosine_similarity_scores)
            else:
                query_to_context[query] = (
                    {k: abstracts[k] for k in relevant_pmids},
                    None
                )

        status.update(label='Generating summary using provided context...')
        st.write('Generating summary using provided context...')
        session_state.script_messages = messages
        gpt_summary, messages, truncated_2 = gpt_generate_summary(
            messages,
            user_query,
            query_to_context,
            settings.model_name,
            settings.temperature)

        status.update(label='Done!', state='complete', expanded=False)
        st.write('Done!')

    session_state.truncated = any((truncated_1, truncated_2))
    session_state.gpt_summary = gpt_summary
    session_state.messages = [{'role': 'assistant',
                               'content': st.session_state.gpt_summary}]
    session_state.messages_with_context = messages
    session_state.clicked = False


def chat_window(session_state: st.session_state, settings: Settings) -> None:
    '''
    create chat window and allow user to chat with the model
    '''
    st.header('Chat', anchor=False)
    with st.container():
        messages_box = st.container(height=500)
        for message in st.session_state.messages:
            messages_box.chat_message(
                message["role"],
                avatar='🧑‍💻' if message["role"] == 'user' else '🤖'
            ).write(convert_pmids_to_links(message["content"]))

        if user_message := st.chat_input("Your message"):
            messages_box.chat_message(
                "user",
                avatar='🧑‍💻').markdown(user_message)
            session_state.messages.append(
                {"role": "user", "content": user_message})

            session_state.messages_with_context.append(
                {'role': 'user',
                 'content': 'CONTINUE_CHAT\n' + user_message})
            response = chat_completion_request(
                session_state.messages_with_context,
                settings.model_name,
                settings.temperature, stream=True)
            complete_response = messages_box.chat_message(
                "assistant", avatar='🤖').write_stream(response)
            session_state.messages.append(
                {"role": "assistant", "content": complete_response})
            session_state.messages_with_context.append(
                {"role": "assistant", "content": complete_response})


def main():
    '''
    represents main logic of program:
    runs functions from PubMedSummarizer along with creating the interface
    '''
    initialize_cache()
    st.set_page_config(page_title='PubMedSummarizer',
                       page_icon='https://images.emojiterra.com/google/'
                       'noto-emoji/unicode-15/color/1024px/1f4d1.png')
    customize_page_appearance()
    st.title('PubMedSummarizer', anchor=False)
    settings = sidebar()

    initialize_session_state()

    st.header('Search', anchor=False)
    # st.write('')

    col1, col2 = st.columns([10, 1])
    with col1:
        user_query = st.text_input(
            'Search',
            placeholder='Enter your scientific question, '
            'query or keywords',
            label_visibility='collapsed')
    with col2:
        st.button('Run', on_click=click_button, type='primary')

    if user_query and (st.session_state.prev_query != user_query):
        click_button()

    if st.session_state.clicked:
        publications_search_and_analysis(st.session_state,
                                         user_query,
                                         settings)

    # chat should be displayed after first run
    if st.session_state.gpt_summary:
        with st.popover('Show optimized queries and found relevant articles'):
            for query, pmids in st.session_state.opt_queries_to_pmids.items():
                st.write(f'__{query}__')
                if pmids:
                    st.write(convert_pmids_to_links(
                        ', '.join([str(i) for i in pmids])))
                else:
                    st.write('None')

        if not all(
            value is None
                for value in st.session_state.opt_queries_to_pmids.values()):

            chat_window(st.session_state, settings)

            if st.session_state.truncated:
                st.warning(
                    '__Some data was truncated in order '
                    'to fit in the context window.__\n\n'
                    'Try to reduce one of the following: '
                    'n optimized queries, n articles per query, '
                    'n chunks per article, chunk size, '
                    'or choose a model with bigger '
                    'context window or modify your query and try again.')

            text_contents = messages_to_human_readable(
                st.session_state.script_messages
                + st.session_state.messages_with_context[1:])
            st.download_button('Download full chat history', text_contents,
                               'PubMedSummarizer_messages.txt',
                               help='Includes messages generated during '
                               'the search process')

        else:
            st.error('__Nothing was found.__\n\nTry to adjust '
                     'your search query and/or modify filters.')


if __name__ == '__main__':
    main()
