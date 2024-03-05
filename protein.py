"""
Functions related to protein sequences.
"""
import logging
import requests
import re

from collections import namedtuple

ProtFam = namedtuple('ProtFam', ['superfamily', 'family', 'subfamily'], defaults=('','',''))


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


def unique_protein_family(id):
    """Take raw merged id, which may have many features separated by space, and return a single ProtFam() object"""
    families = sorted(list(set([get_protein_families(ft_id.split(':')[-1]) for ft_id in id.split()])))
    # if len(families) > 2 or (len(families) == 2 and families[0] != ('', '', '')):

    # log if families includes non-congruent triplets
    for grp in zip(*families):
        grp_set = set(grp)
        grp_set.discard('')
        if len(grp_set) > 1:  # more than one unique nonempty (super//sub)family, i.e. incongruent
            logging.info(f'Feature with interesting protein families: {families}')

    # breakpoint()
    if families:
        superfamily, family, subfamily = families[-1]  # last unique family group, since first may be empty
        return ProtFam(superfamily=superfamily, family=family, subfamily=subfamily)
    return ProtFam()
