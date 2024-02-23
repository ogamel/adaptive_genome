"""
Functions related to protein sequences.
"""

import requests
import re

def search_uniprotkb(query: str):
    """Search the UniprotKb database. Useful for retrieving protein info."""
    base_url = f'https://rest.uniprot.org/uniprotkb/search?query='
    r = requests.get(base_url + query)
    return r.json()['results']


def get_protein_substrings(id: str):
    """Get name of protein family, if any, from ensembl id."""
    # Note: there is a sub-subfamily
    pattern = r"'([^']*family[^']*)'"
    query_results = search_uniprotkb(id)
    return re.findall(pattern, str(query_results))


def get_protein_families(id: str):
    """Get name of protein family, if any, from ensembl id."""
    # Note: there is a sub-subfamily
    PREAMBLE = 'Belongs to the '
    substrings = get_protein_substrings(id)

    # TODO: consider using NLP to capture more general phrasing
    #
    superfamily, family, subfamily = '', '', ''
    for substring in substrings:
        # substring = substring.lower()
        if substring.startswith(PREAMBLE):
            for subsubstring in substring[len(PREAMBLE):].split('.'):
                if (ind := subsubstring.find(' superfamily')) != -1:
                    superfamily = subsubstring[:ind]
                elif (ind := subsubstring.find(' subfamily')) != -1:
                    subfamily = subsubstring[:ind]
                elif (ind := subsubstring.find(' family')) != -1:
                    family = subsubstring[:ind]
            break
    else:
        pass
        """
        Consider the stuff below
        ** family member ** if no family assigned yet - before it is needed
        family - (remove family) --> false positives
        """

    return superfamily, family, subfamily
