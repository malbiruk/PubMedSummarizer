import colorsys
import json
import random

from openai import OpenAI
from pyvis.network import Network


def create_ner_prompt(entities: list) -> str:
    chemical_rules = ('EXLUDE protein and protein structures from CHEMICAL entity.\n'
                      if 'chemical' in entities else '')
    bioprocess_rules = ('Recognize biological processes mentions with morphological variations.\n'
                        if 'biological process' in entities else '')
    xenograft_rules = (
        'If the name of a xenograft contains tissue, then we annotate the Tissue; '
        'if it is an organism, then the Organism. The word “xenograft” is not annotated\n'
        'Example: [mouse] xenograft -> Organisms, [cardiac] xenografts -> Tissues\n'
        'Organoids are annotated as Cell lines. Viroids are annotated as Organisms. '
        'In the case of prions, we annotate the Gene (protein) as prion mentions usually '
        'come along with gene mentions\n'
        if (('tissue' in entities) and ('organism' in entities)
            and ('cell line' in entities)) else '')
    prompt = (
        ('You are a biomedical data labeler labeling data to be used in token/Named Entity Recognition.\n'
         'In the text below, give me the list of:\n')
        + '\n'.join([f'- {entity} named entity' for entity in entities])
        + ('\nIf there are full name of entity and its short form in the text, '
           'recognize both forms as separate entities and include in the list.\n'
           'Enteties should not overlap. If they are overlapping, choose bigger one.'
           'You can treat parts of words connected with hyphen as different words.\n')
        + chemical_rules + bioprocess_rules + xenograft_rules
        + 'Format the output in json with the following keys:\n'
        + '\n'.join([f'- {entity.upper()} for {entity} named entity' for entity in entities])
        + '\nText below:\n')
    return prompt


def ner_with_gpt(text: str,
                 gpt_model: str = "gpt-4-turbo-preview",
                 api_key: str = None,
                 entities: list = None):
    openai = OpenAI(api_key=api_key) if api_key else OpenAI()
    entities = entities if entities else ['gene', 'disease', 'drug']

    system_prompt = create_ner_prompt(entities)

    response = openai.chat.completions.create(
        model=gpt_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    generated_content = response.choices[0].message.content
    return json.loads(generated_content)


def knowledge_graph_with_gpt(text: str,
                             entities_json: dict,
                             relations: list = None,
                             gpt_model: str = "gpt-4-turbo-preview",
                             api_key: str = None):
    openai = OpenAI(api_key=api_key) if api_key else OpenAI()
    relations = relations if relations else [
        'associated with',
        'treats',
        'interacts with',
        'inhibits',
        'activates']

    relations_string = ', '.join(f"'{relation}'" for relation in relations)
    system_prompt = f"""
Generate a knowledge graph from the provided text and entities. \
Each relationship should be categorized with one of the following relations: \
{relations_string}. \
Use short names for GENE enteties and full names for CHEMICAL entities. \
Remove synonyms and antonyms to ensure clarity. Try to avoid orphans/islands in the graph.

The output should be a JSON object representing the knowledge graph, with the following structure:
""" + """
{
    "edges": [
        {"from": "entity1", "to": "entity2", "relation": "relation_type"},
        {"from": "entity2", "to": "entity3", "relation": "relation_type"},
        ...
    ]
}

Each edge in the edges list should represent a relationship between two entities, \
labeled with the appropriate relation type.
"""

    message = "Text:\n" + text + '\n\nEntities:\n' + json.dumps(entities_json, indent='\t')

    response = openai.chat.completions.create(
        model=gpt_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    )
    generated_content = response.choices[0].message.content
    return json.loads(generated_content)['edges']


def random_colors(N: int, value: float = 1.0, saturation: float = 0.3,
                  random_order=False) -> list:
    '''
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    '''
    hsv = [(i / N, saturation, value) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if random_order:
        random.shuffle(colors)
    return ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in colors]


def find_key_by_value(dictionary, search_value):
    for key, value in dictionary.items():
        if search_value in value:
            return key
    return None


def draw_knowledge_graph(edges: list,
                         entities_json: dict,
                         relation_to_color: dict = None) -> None:
    net = Network(height='600px', width='100%', directed=True)
    net.barnes_hut()
    relation_to_color = relation_to_color if relation_to_color else {'activates': 'palegreen',
                                                                     'inhibits': 'lightsalmon'}
    colors = dict(zip(list(entities_json.keys()), random_colors(len(entities_json.keys()))))

    for edge in edges:
        entity_from = find_key_by_value(entities_json, edge['from'].replace('\n', ''))
        net.add_node(edge['from'], edge['from'], title=entity_from,
                     color=colors.get(entity_from))
        entity_to = find_key_by_value(entities_json, edge['to'].replace('\n', ''))
        net.add_node(edge['to'], edge['to'], title=entity_to,
                     color=colors.get(entity_to))
        net.add_edge(edge["from"], edge["to"], title=edge["relation"], width=20,
                     arrowStrikethrough=False,
                     color=(relation_to_color[edge["relation"]]
                            if relation_to_color.get(edge["relation"])
                            else 'lightgray')
                     )

    for n in net.nodes:
        n["size"] = 150
        n["font"] = {"size": 100}

    for e in net.edges:
        if "label" in e:
            e["font"] = {"size": 100}

    net.write_html("knowledge_graph.html")
