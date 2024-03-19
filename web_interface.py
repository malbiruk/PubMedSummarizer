import os
import re
from datetime import datetime

import config
import streamlit as st
from PubMedSummarizer import (chat_completion_request, extract_terms,
                              get_abstracts, get_article_texts,
                              get_context_from_articles, gpt_generate_summary,
                              gpt_identify_relevant, gpt_process_query,
                              messages_to_human_readable)
from sentence_transformers import SentenceTransformer


def customize_page_appearance():
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


def click_button():
    st.session_state.clicked = True


def convert_pmids_to_links(text: str) -> str:
    pattern = r'(?<!\[)(?<!\d)(\d{7,8})(?!\d)(?!\])'
    replacement = r'[\1](https://pubmed.ncbi.nlm.nih.gov/\1/)'
    result = re.sub(pattern, replacement, text)
    return result


def main():
    st.set_page_config(page_title='PubMedSummarizer',
                       page_icon=':bookmark_tabs:')
    customize_page_appearance()
    st.title('PubMedSummarizer', anchor=False)

    with st.sidebar:
        st.title('Settings')

        st.header('Search settings')
        from_year = None
        pub_types = None
        pmid_list = None
        email = 'fiyefiyefiye@gmail.com'
        use_full_article_texts = st.toggle(
            'Use full articles\' texts',
            help='This implies downloading and embedding articles on-the-fly. '
            'It may provide more acccurate results and context, but '
            'keep in mind that it may take about 10 minutes per run.',
        )
        use_pmids_list = st.toggle('Use provided list of PMIDs')
        if not use_pmids_list:
            n_queries = st.slider('Generate ___n___ optimized queries '
                                  'from input query',
                                  1, 10, 3)
            n_articles = st.slider('Retrieve top ___n___ articles per query',
                                   1, 25, 10)
            filter_by_year = st.toggle('Filter by publication date')
            if filter_by_year:
                from_year = st.slider(
                    'Start year',
                    1990, datetime.now().year, datetime.now().year - 10,
                    disabled=not filter_by_year)
            filter_by_pub_type = st.toggle('Filter by publication type')
            if filter_by_pub_type:
                all_pub_types = [
                    'Classical Article', 'Review', 'Systematic Review',
                    'Clinical Trial', 'Meta-Analysis']
                pub_types = []
                for pub_type in all_pub_types:
                    if st.checkbox(pub_type):
                        pub_types.append(pub_type)
        else:
            pmid_list = st.text_input('List of PMIDs (separated by spaces)')
        # email = st.text_input('Email for Entrez', 'fiyefiyefiye@gmail.com')

        st.header('Models')
        model_name = st.selectbox('GPT model',
                                  ['gpt-3.5-turbo-0125', 'gpt-4-0125-preview'])
        if st.toggle('Modify temperature (defaul=0.2)'):
            temperature = st.slider('Temperature', 0., 1., .2, .1)
        else:
            temperature = .2
        embedder_name = st.selectbox('Embedding model',
                                     ['dmis-lab/biobert-base-cased-v1.2',
                                      'msmarco-distilbert-base-v4'])
        if use_full_article_texts:
            st.header('Chunking')
            n_chunks = st.slider('Retrieve top ___n___ chunks per article',
                                 1, 20, 5)
            chunk_size = st.slider('Chunk size, sentences',
                                   1, 20, 6)
            overlap = st.slider('Chunks overlap, %', 0, 50, 30, 5)
            overlap = round(overlap / 100 * chunk_size)

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    if "gpt_summary" not in st.session_state:
        st.session_state.gpt_summary = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'messages_with_context' not in st.session_state:
        st.session_state.messages_with_context = []
    if 'script_messages' not in st.session_state:
        st.session_state.script_messages = []
    if 'prev_query' not in st.session_state:
        st.session_state.prev_query = ''

    st.write("")
    col1, col2 = st.columns([10, 1])
    with col1:
        user_query = st.text_input(
            'User query',
            placeholder='Enter your scientific question, '
            'query or keywords',
            label_visibility='collapsed')
    with col2:
        st.button('Run', on_click=click_button, type='primary')

    # st.write('This is sample text. ' * 20)
    if st.session_state.prev_query != user_query:
        st.session_state.clicked = True

    if st.session_state.clicked:
        st.session_state.prev_query = user_query
        with st.status('Optimizing query...') as status:
            st.write('Optimizing query...')
            embedder = SentenceTransformer(embedder_name)
            if not os.path.exists('cache'):
                os.makedirs('cache')
            current_date = datetime.now().strftime('%Y/%m/%d')
            prompt = config.PROMPT.replace(
                '{N_QUERIES}', str(n_queries)).replace(
                '{CURRENT_DATE}', current_date)
            messages = [{'role': 'system', 'content': prompt}]
            optimized_queries = gpt_process_query(messages, user_query,
                                                  model=model_name,
                                                  temperature=temperature)
            query_to_context = {}
            if pmid_list is not None:
                pmid_list = [str(i) for i in pmid_list]

            for q in optimized_queries:
                status.update(label=f'Searching articles by query {q}...')
                st.write(f'Searching articles by query {q}...')
                abstracts = get_abstracts(q,
                                          n_res=n_articles,
                                          email=email,
                                          pmid_list=pmid_list,
                                          from_year=from_year,
                                          pub_types=pub_types
                                          )
                if not abstracts:
                    continue

                q = q.replace('"', '')
                query = extract_terms(q)
                status.update(label='Getting relevant PMIDs...')
                st.write('Getting relevant PMIDs...')
                relevant_pmids = (gpt_identify_relevant(messages,
                                                        query,
                                                        abstracts,
                                                        model_name,
                                                        temperature)
                                  if pmid_list is None else pmid_list)

                if not relevant_pmids:
                    continue

                if use_full_article_texts:
                    status.update(label='Downloading articles\' texts...')
                    st.write('Downloading articles\' texts...')
                    articles = get_article_texts(relevant_pmids)

                    status.update(
                        label='Embedding articles\' texts and '
                        'performing semantic search...')
                    st.write('Embedding articles\' texts and '
                             'performing semantic search...')
                    context_chunks, cosine_similarity_scores = (
                        get_context_from_articles(query,
                                                  embedder,
                                                  articles,
                                                  n_res=n_chunks,
                                                  chunk_size=chunk_size,
                                                  overlap=overlap))
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
            st.session_state.script_messages = messages
            gpt_summary, messages = gpt_generate_summary(messages,
                                                         user_query,
                                                         query_to_context,
                                                         model_name,
                                                         temperature,
                                                         prompt)

            status.update(label='Done!', state='complete', expanded=False)
            st.write('Done!')

        st.session_state.gpt_summary = convert_pmids_to_links(gpt_summary)
        st.session_state.messages = [{'role': 'assistant',
                                      'content': st.session_state.gpt_summary}]
        st.session_state.messages_with_context = messages
        st.session_state.clicked = False

    if st.session_state.gpt_summary:
        # st.markdown(st.session_state.gpt_summary)
        # st.caption('Here you can chat with the model, using provided context')
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
                st.session_state.messages.append(
                    {"role": "user", "content": user_message})

                st.session_state.messages_with_context.append(
                    {'role': 'user',
                     'content': 'CONTINUE_CHAT\n' + user_message})
                response = chat_completion_request(
                    st.session_state.messages_with_context, model_name,
                    temperature, stream=True)
                complete_response = messages_box.chat_message(
                    "assistant", avatar='🤖').write_stream(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": complete_response})
                st.session_state.messages_with_context.append(
                    {"role": "assistant", "content": complete_response})

        if None in [i['content'] for i in (
                st.session_state.script_messages
                + st.session_state.messages_with_context[1:])]:
            st.warning(
                '__Something went wrong, got some empty responses.__\n\n'
                'Try to reduce one of the following: '
                'n optimized queries, n articles per query, '
                'n chunks per article, chunk size, '
                'or choose a model with bigger '
                'context window or modify your query and try again.')

        text_contents = messages_to_human_readable(
            st.session_state.script_messages
            + st.session_state.messages_with_context[1:])
        st.download_button('Download full chat history', text_contents,
                           'PubMedSummarizer_messages.txt')


if __name__ == '__main__':
    main()
