"""
Paths to data files.

V0.1: Based on Ensembl release 109

Parent: https://ftp.ensembl.org/pub/release-109/

Whole Genome data:
Sequence (FASTA, 64 GB): https://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz
Annotation (GFF3, 540 MB): https://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/Homo_sapiens.GRCh38.109.gff3.gz
GERP (BigWig, 11GB): https://ftp.ensembl.org/pub/release-109/compara/conservation_scores/90_mammals.gerp_conservation_score/gerp_conservation_scores.homo_sapiens.GRCh38.bw

Chromosome 17:
Sequence (FASTA, 85 MB): https://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/Homo_sapiens.GRCh38.109.chromosome.17.gff3.gz
Annotation (GFF3, 32 MB): https://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/Homo_sapiens.GRCh38.109.gff3.gz
GERP: same as whole genome

We use semi masked sequence FASTA files (_sm suffix), where lowercase bases denote repetitive sequences.
"""
from collections import namedtuple

PROJ_PATH = '/Users/omar/Research/Adaptive Genome'
FIGS_PATH = f'{PROJ_PATH}/figs'

ProjectPaths = namedtuple('ProjectPaths', ['sequence', 'annotation', 'gerp'])

# Whole genome
wgs_paths = ProjectPaths(
    sequence = f'{PROJ_PATH}/data/human_genome/ensembl/fasta/Homo_sapiens.GRCh38.dna_sm.toplevel.fa',
    annotation = f'{PROJ_PATH}/data/human_genome/ensembl/gffs/Homo_sapiens.GRCh38.109.gff3',
    gerp = f'{PROJ_PATH}/data/gerp/gerp_conservation_scores.homo_sapiens.GRCh38.bw'
)

# Chromosome 17 only
chr17_paths = ProjectPaths(
    sequence = f'{PROJ_PATH}/data/human_genome/ensembl/fasta/Homo_sapiens.GRCh38.dna_sm.chromosome.17.fa',
    annotation = f'{PROJ_PATH}/data/human_genome/ensembl/gffs/Homo_sapiens.GRCh38.109.chromosome.17.gff3',
    gerp = f'{PROJ_PATH}/data/gerp/gerp_conservation_scores.homo_sapiens.GRCh38.bw'
)
