import sys

import neuralcoref
import spacy

import requests


nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)


class LabelNotFound(Exception):
    pass


def _lookup_dbpedia(term):
    uri = """
        http://lookup.dbpedia.org/api/search/KeywordSearch?QueryString={}
    """
    uri = uri.format(term)
    response = requests.get(uri, headers={'Accept': 'application/json'})

    try:
        return response.json()['results'][0]['uri']
    except Exception:
        raise LabelNotFound


def _spotligth_dbpedia(doc, confidence=.7):
    uri = """
        http://model.dbpedia-spotlight.org/en/annotate?text={}&confidence={}
    """
    uri = uri.format(doc.text, confidence)
    response = requests.get(uri, headers={'Accept': 'application/json'})
    try:
        results = {
            r['@surfaceForm']: r['@URI']
            for r in response.json()['Resources']
        }
        return results
    except Exception:
        return {}


def filter_entity_names(ents):
    accepted_en_types = [
        'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
        'WORK_OF_ART', 'LAW'
    ]

    return [
        e for e in ents if e.label_ in accepted_en_types and len(e.text) > 2
    ]


def get_disambiguated_entities(doc, remove_discrepancy=True):
    from collections import defaultdict
    results = defaultdict(set)
    [results[k].add(v) for k, v in _spotligth_dbpedia(doc).items()]
    for e in filter_entity_names(doc.ents):
        try:
            label = _lookup_dbpedia(e)
            results[e.text].add(label)
        except LabelNotFound:
            pass

    if remove_discrepancy:
        results = {k: v for k, v in results.items() if len(v) == 1}
    return sorted(results.items())


def resolve_coref(text):
    doc = nlp(text)
    return doc._.coref_resolved


def annotate_text(text):
    doc = nlp(resolve_coref(text))
    ents = get_disambiguated_entities(doc)
    result = str(text)
    for i, (k, _) in enumerate(ents):
        result = result.replace(k, 'FN-TOKEN-{}'.format(i))
    for i, (_, v) in enumerate(ents):
        result = result.replace('FN-TOKEN-{}'.format(i), list(v)[0])

    return result, ents


if __name__ == "__main__":
    label = _lookup_dbpedia(sys.argv[1])
    print(label)
