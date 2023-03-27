# Does the genome drive its own evolution?
Author: Omar Gamel

Underlying this project is a a simple idea, can one distill patterns in the genetic sequence itself that predict how and where the genome will vary 
between related sequences? That is, between species, or between different members of a gene / protein family. For example, are there particular 
genomic patterns that are more variable and / or conserved than others?

## Abstract
Analyzing human chromosome 17, we extend the known inversion symmetry between occurence counts of reverse complement k-mers of nucleotides to a measure 
of a nucleotide's conservation across a multispecies alignment. 
Namely, the [GERP score](http://mendel.stanford.edu/SidowLab/downloads/gerp/). This provides evidence that sequence itself influences how different 
it will be across species. That is, we have an indication that the genome drives its own evolution.

## Project Overview

Below we do such an analysis on a Human Chromosome, yielding very interesting discoveries.

Our modus operandi is to collect relevant data in pandas dataframe, then extract insight through various plots and correlations.
We rely on the GERP score to quantify the degree of conservation or variability of a nucleotide. A positive score indicates a nucleotide more conserved than expected under neutral sequence evolution, while a negative score indicates one more variable than expected.

The data is obtained from [ENSEMBL](ensembl.org). We use the softmasked genome, but effectively ignore lower case letters. Check data/paths.py for details.

Many thanks to Dr. Lynn Caporale for important ideas and help creating an overarching [research proposal](https://docs.google.com/document/d/18zZY_aS1gq4SWBPvaKIlqKkqmLpxg_wT0Vn8HHdJSmg). Thanks to the ENSEMBL team for valuable help acquiring the data.