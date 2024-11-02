"""
Functions related to protein sequences.
"""
import os
import logging
import requests
import re
import pickle

from collections import namedtuple
from data.paths import PROT_CACHE_PATH

ProtFam = namedtuple('ProtFam', ['superfamily', 'family', 'subfamily'], defaults=('','',''))


class ProteinCache:
    def __init__(self, path=PROT_CACHE_PATH, max_size=None, clean_slate=False):
        self.path = path
        self.max_size = max_size  # TODO: implement LRU

        if os.path.exists(path) and os.path.getsize(path) > 0 and not clean_slate:
            with open(path, 'rb') as f:
                self._cache = pickle.load(f)
        else:
            self._cache = {}

    def __contains__(self, item):
        return item in self._cache

    def __delitem__(self, key):
        del self._cache[key]

    def __getitem__(self, item):
        return self._cache[item]

    def __len__(self):
        return len(self._cache)

    def __setitem__(self, key, value):
        self._cache[key] = value

    def clear_cache(self):
        self._cache = {}

    def save(self, path=None):
        path = path or self.path
        if path is None:
            raise ValueError("cache path must be given")
        with open(path, 'wb') as f:
            pickle.dump(self._cache, f)
        return


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


prot_cache = ProteinCache()
def get_protein_families(id: str):
    """Get name of protein family, if any, from ensembl id."""
    if id in prot_cache:
    # if id in prot_cache and not (')' in prot_cache[id][1] or '(' in prot_cache[id][1]):
        return prot_cache[id]
    # Note: there is a sub-subfamily
    PREAMBLE = 'Belongs to the '
    substrings = get_protein_substrings(id)

    # superfamily, family, subfamily = prot_cache[id]

    # TODO: consider using NLP to capture more general phrasing

    superfamily, family, subfamily = '', '', ''
    for substring in substrings:
        # substring = substring.lower()
        if substring.startswith(PREAMBLE):
            for subsubstring in substring[len(PREAMBLE):].split('. '):
                if (ind := subsubstring.find(' superfamily')) != -1:
                    superfamily = subsubstring[:ind].strip()
                elif (ind := subsubstring.find(' subfamily')) != -1:
                    subfamily = subsubstring[:ind].strip()
                elif (ind := subsubstring.find(' family')) != -1:
                    family = subsubstring[:ind].strip()
            break
    else:
        pass
        """
        Consider the stuff below
        ** family member ** if no family assigned yet - before it is needed
        family - (remove family) --> false positives
        """

    prot_cache[id] = superfamily, family, subfamily
    # print("contains brackets. id: ", id, "fam:", family, "sub str:", substrings)
    return prot_cache[id]


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
