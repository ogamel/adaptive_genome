"""
Module to simulate alternative "artificial genomes" so we may check their properties to benchmark those we find in real
biological genomes (of any species).
E.g. create a genetic code for a human language, and measure therein the same properties we measure for the genome.
Check to what extent a random corpus has these properties.
"""
import logging
import random
import pandas as pd
import numpy as np
import os
import re

from typing import Iterator, Callable, Iterable, Optional
from itertools import product
from collections import defaultdict
from score_collation import K_COL, KMER_COL, SEQNAME_COL, FRAME_COL, COUNT_COL

from data.paths import english_genome_dir
from genetic import NUCLEOTIDE_ALPHABET, code_frequency_proportions, code_frequencies
from util import periodic_logging


def gen_powers_with_sum(num, s, x=2, max_power=2):
    """Generate num powers of x, up to power max_power, that sum to s."""

    counts = [0] * (max_power+1)
    # initially fill counts with mostly the highest power
    for p in reversed(range(max_power+1)):
        counts[p] = s//(x**p)
        s = s % x**p

    i = 0
    while sum(counts) < num:
        ind = max_power - (i % max_power)  # source index
        if counts[ind] > counts[ind - 1]:
            counts[ind] -= 1
            counts[ind - 1] += x
        i += 1

    return sum([[x**i]*counts[i] for i in reversed(range(max_power+1))], [])


def generate_heurestic_code_frequencies(target_size):
    # start with standard genetic code codon frequencies, then sample a frequency count (weighted by frequency
    # counts) greater than 1, which is then split to two counts, until we have enough separate letters.
    # TODO: build a version that combines codons if the desired alphabet is shorter than standard genetic code
    freq_dict = code_frequencies()
    while (sz := sum(freq_dict.values())) < target_size:
        c = np.random.choice(list(freq_dict.keys()), p=np.array(list(freq_dict.values())) / sz)
        # split to c to two values
        if c == 1:
            continue
        elif c == 2:
            a = b = 1
        else:
            a = c // 2 + 1
            b = c - a
        freq_dict[c] -= 1
        freq_dict[a] += 1
        freq_dict[b] += 1

    return sum([[k] * v for k, v in freq_dict.items()], [])


def codon_tables(alphabet='abcdefghijklmnopqrstuvwxyz. ', mode='random', bases=NUCLEOTIDE_ALPHABET):
    """
    Creates the codon forward and back table for the artificial alphabet provided.
    Uses bases in bases. Codon lengths are the minimum that can accommodate the alphabet length.
    mode: one of 'random', 'sequential', 'natural'
        'random' assigns letters to codons randomly, such that each letter has at least one codon.
        'sequential' assigns letters to codons deterministically in sequence.
        'sequential2' assigns letters to codons deterministically in sequence, in blocks length of powers of two.
        'natural' follows some general heuristics present in the standard genetic code, such as putting most of the \
                    synonymous mutation in the last base.

    Returns:
    forward_table (dict): mapping from codon to alphabet member.
    back_table (dict): mapping from alphabet member to codon.
    """
    alphabet_size = len(alphabet)
    n = len(bases)
    k = int(np.ceil(np.log(alphabet_size)/np.log(n)))  # minimum codon length to capture the whole alphabet

    codons = [''.join(p) for p in product(bases, repeat=k)]
    forward_table = {codon: None for codon in codons}  # to fix order
    back_table = {letter: [] for letter in alphabet}

    if mode == 'random':
        random.shuffle(codons)
        alphabet_set = set(alphabet)
        for codon in codons:
            if alphabet_set:
                forward_table[codon] = alphabet_set.pop()
            else:
                forward_table[codon] = random.choice(alphabet)
    elif mode == 'sequential':
        ind = 0
        for codon in codons:
            forward_table[codon] = alphabet[ind]
            ind = (ind + 1) % alphabet_size
    elif mode in ['sequential2', 'natural']:
        if mode == 'sequential2':
            # generate alphabet_size powers of 2 that sum to n**k
            num_list = gen_powers_with_sum(alphabet_size, n**k, x=2, max_power=2)
        else:
            # generate alphabet_size numbers that roughly follow a similar pattern to natural genetic code
            num_list = generate_heurestic_code_frequencies(alphabet_size)

        ind = 0  # letter index in the alphabet
        c = 0  # index of the synonomous codon encoding the letter
        for codon in codons:
            forward_table[codon] = alphabet[ind]
            c += 1
            if c == num_list[0]:
                ind += 1
                c = 0
                num_list.pop(0)

    for codon, letter in forward_table.items():
        back_table[letter].append(codon)

    return forward_table, back_table


def artificial_stats_by_kmer(back_table, file_dir=english_genome_dir, codon_choice='random',
                             k_values: Iterable[int] = (2,)) -> pd.DataFrame:
    """
    Do basic analysis of frequency statistics by k-mer on text documents using an artificial genetic code.

    file_dir: directory of .txt files to use as the "genome".
    back_table: dictionary from letter to list of codons.
    codon_bias: how to choose the codon among synonmous ones. One of 'random', 'random_biased', 'sequential'.
    k_values list[int]: k values to analyze.

    Outputs a dataframe of codon, its frequency, frame, and position information.
    """

    files = os.listdir(file_dir)
    txt_files = [file for file in files if file.endswith('.txt')][:1]

    # Iterate over the .txt files and read their contents
    for txt_file in txt_files:
        file_path = os.path.join(file_dir, txt_file)

        with open(file_path, 'r') as file:
            text = file.read().lower()
            text = re.sub(r'[\n\t]', ' ', text)
            text = re.sub(r'[^a-z. ]', '', text)

            # "untranslate" text to genetic code
            logging.info(f'"Untranslating" {txt_file} of length {len(text)} characters.')
            genome = ''
            for i, letter in enumerate(text):
                periodic_logging(i, f'letter {i}')
                codons = back_table[letter]
                if codon_choice == 'random':
                    codon = np.random.choice(codons)
                elif codon_choice == 'random_biased':
                    codon = np.random.choice(codons, p=np.array())
                elif codon_choice == 'sequential':
                    codon = codons[i % len(codons)]

                genome += codon
            logging.info(f'Completed the "untranslation" of text of {len(text)} characters to genome of length '
                         f'{len(genome)}.')

            kmer_data = defaultdict(lambda: 0)
            for k in k_values:
                # loop through the text in a window of width k, shifting by one
                # nucleotide at a time, updating the current kmer, and values with each iteration
                cur_kmer = ''
                for ind in range(len(genome)):
                    next_nucleotide = genome[ind]

                    cur_kmer = (cur_kmer + next_nucleotide)[-k:]
                    if len(cur_kmer) < k:
                        continue

                    frame = ind % k
                    # track running count
                    kmer_data[(k, cur_kmer, txt_file, frame)] += 1

    # compute overall count, mean and standard deviation
    kmer_data_agg = []
    for key, count in kmer_data.items():
        k, cur_kmer, txt_file, frame = key
        kmer_data_agg.append(
            {
                K_COL: k,
                KMER_COL: cur_kmer,
                SEQNAME_COL: txt_file,
                FRAME_COL: frame,
                COUNT_COL: count,
            })

    # create output DataFrame
    kmer_base_df = pd.DataFrame(kmer_data_agg)
    kmer_base_df.sort_values(by=['k','kmer','frame'])

    logging.info(f'Computed artificial genome score stats by k-mer, on {len(kmer_base_df)} k-mer outputs, '
                 f'for files in {file_dir}.')

    return kmer_base_df
    # CONCLUSION: random genome has utterly no pattern of any kind. Count not the same in every frame.
