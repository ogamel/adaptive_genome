# Research Notes
  فتبارك الله احسن الخالقين

## Principles
The overarching goal is to find interesting patterns in the distribution and variation of the genome and its components. 
In doing so, one (i) organizes reasoning and (ii) clarifies questions that will be addressed. 
These are then answered through (iii) clean well-documented code, with (iv) clear visuals for each point.

## Active tasks: 
- Clean up code and codify discoveries
- Add feature to prediction model - by getting details on the protein coded (e.g. ENSP00000007708) - feature enrich
- Get protein family annotations, check GERP relation between them
ENSP00000471646 ENSP00000482950 ENSP00000513293 ENSP00000319210 ENSP00000479485 ENSP00000328977
## Conclusions
### Macro
- GERP Analysis by feature type
  - From GERP mean, we find CDS much more conserved than everything else, as expected. So we do our analysis on it
  - UTRs unconserved. 
  - Exon has intermediate GERP, which makes sense since exons = CDS + UTR. 
  - GERP variance does not seem useful, too high (turns out useful later).
  - We restrict most sequence analysis to CDS - within coding sequence of a single chromosome
  
- Masking
  - check with small (masked) letters, should I just ignore them? Frequencies: all caps 10000s. .. mixed caps and small 
  10s, (seems regional boundaries).. all small in 100s-1000s. We drop the mixed, compare the all caps with all small... 
  figure out difference in meaning .. 
  - We drop anything with small letters for initial analysis since it is infrequent relative to capitals

### Inversion symmetry
[Chargaff's Second rule]https://en.wikipedia.org/wiki/Chargaff%27s_rules)
Already known for frequencies, perhaps my finding can be to extend it to GERP, and find exceptions... 
 - what can be concluded from this, is it that Chargaff's rule has to do with variation, i.e. this is a novel finding. 
or is this just a trivial consequence? too sleep-deprived to tell.
   - Turns out neither. GERP uncorrelated with frequency. But the inversion symmetry of neither is trivial.
 In literature no relation known between Chargaff's and genetic code ... perhaps I should find one .... would be amazing, 
 إن شاء الله... اللهم افتح علينا

#### Breakdown by complementable columns   
الحمد لله تعالى, فقد علمني الله

Each K-mer within a CDS has almost identical GERP score as its reverse complement on a per POSITION basis, where the 
STRAND and FRAME are complemented, 
- Frame used is the correct one - i.e. subtracting phase in the right direction depending on strand
- Each of strand, frame, pos is necessary, as shown by diff analysis described in the next subsection. 
    - Slight exception is strand for k=1, where it seems even on same strand (not complemented) there is still 
       strong correlation this might be a hint of the mechanism
  
What does this mean? The above implies kmer is same as itself on either strand, when that strand is coding. That is,

    Each kmer has its own well defined position-wise GERP on the coding strand as a function of its frame.

Could this all somehow be trivial? No. It shows GERP (variation) is largely a function of kmer. 
How much? For k=3, std. dev. of all the means across (kmer, strand, frame, pos) tuples is about 1.275, while mean std. 
dev. within a single tuple is 2.4. So a good explanation, but most variation is still within a given tuple.
(TODO: Verify this: done with XGBoost)

For each kmer, first two codon positions have positive GERP, i.e. don't vary much, and the final one 
(or first in -1 strand) varies a lot with very negative GERP.

As for k-mer counts, basically just frame is complementable. Strand doesn't seem to be. phase no, ft_len no. 

    So we can further refine extended Chargaff's rule by frame, for coding regions.

q: What about noncoding? there is no frame there. So what does this mean then?



#### Differential analysis
Diff analysis results make sense. The differences between candidate complements are very small. They are calculated
as mean square difference between of the score or count between complements for a given k-mer and its RC, divided by
the same mean square difference for all kmers (TODO: review this definition).
If you shuffle each complementable column, the difference between candidate complements increases dramatically. 
Meaning each column is important to the establishment of the complement relationship.

Note: I started with  mean absolute difference - whose expected value randomly is 1/sqrt(pi) = 1.12
([ref](https://stats.stackexchange.com/questions/489075/what-is-the-mean-absolute-difference-between-values-in-a-normal-distribution)),
then switched to mean square difference, which is easier to relate to variance and std. dev.

Trying with entire genome, the differential analysis differs by genome. Chromosomes 19 and Y both do poorly on 
differential analysis, where the discovered pattern is weaker, though still much better than mixing up the features.
Chr_Y has very few CDS (86 vs the rest 3K-20K) so we can ignore it. Chr_19 has 12K CDS, but -0.317 score_mean vs 
the rest 0.333-0.768, It rms diff is 0.15, much higher than otherwise avg 0.05. Chr 20 also has 0.11 rms diff. Lowest is
<0.02 rms diff
Why would some chromosomes differ if it is essential to the variation mechanism?




#### Non-CDS features
Try outside CDS.
- lnc_RNA: frame doesn't matter, because it is ill-defined from the beginning of the feature
    - even strand doesn't matter as much as it did before ... ... but then may be I need to redefine frame here?
    - to some absolute? - ft_start doesn't matter. Frame here not relevant
    - score: position and strand. count: position 
- 3' UTR: score, strand and pos, but not for k=1. count: position
- 5' UTR: score, strand and pos, (weaker than 3') but not for k=1. count: position

### Mutual information vs dilation
 - Analysis for both score and count probability shows mutual information (or its proxy) rises at multiples of 3 
nucleotides, in CDS, and doesn't even decrease up to 18. What does that mean? 
That there is higher correlation with analogous position in downstream codons.
This is fascinating. SubhanAllah. needs plot by dilated 2-mer... for more clarity
What does it mean? that on average one can tell something about bps separated by multiple of three, about 1/5 of 
what one can tell about immediate adjacent. 
 - Can I increase this by looking at some subset? Does it increase for some specific 2-mers?
 - does it depend on the frame, i.e. 1st, 2nd, or 3rd position in codon? I am not combining dilation with frame, 
   perhaps I should

Split by strand and frame - get the period of 3. Peak at frame 2, for both score and count prob ... this is the most 
variable nucleotide - this means it also contains the most info the future one ... more than first and second nucleotide
This indicates the choice of the third nucleotide depends choice of third nucleotide in previous codons, and so does its 
score. 
While first and second don't depend much - second a bit dependence more than first - this seems to make sense since this 
needs to be free to determine the amine acid sequence - perhaps there is another "sequence within a sequence" given in 
the third nucleotide.
But the higher mutual info indicates it is not totally free, a previous one says something about the next.
It seems to me likely that score pattern is a consequence of the probability pattern. On the other hand, for frame = 2
the probability is slightly higher is negative strand while score is higher in positive strand.

### Genetic Code
Delve into genetic code, whether the symmetries above can shed light on its properties.
- Found interesting graph between amino acids where edges are reverse complement codons. Get two disconnected graphs, 
size 7 and 14. The larger can itself is two graphs sizes 8 and 5 connected only by stop codon.
- But what does that even mean? RC the codon only makes sense if both strands are transcribed. actually no, two 
connected amino acids can mean they have common reverse codons 

- plotted score and count by amino acid, for each frame and strand. Interestingly, 
- s=1, f=0 real has very few stops. 
  - stop, M, W have no negative GERP in any codon, very conserved. I understand stop and M (start), but why W? Could it
  be because it is "surrounded by Stop in the Genetic code?"
  - stop's last codon is most conserved 

#### Predict last bp in codon and its score
- seems just need first two bp in current codon to predict third with 0.476 accuracy. earlier codons add nothing
hypothesized and found it equivalent to simply knowing the proportions of each codon among those with same first two 
base pairs. Makes sense, because there is no other information in that one codon 
  - Last base in most prevalent codon is always G or C 
  - should try just 1 bp previous?
- predicting GERP score however seems to be helped by earlier codons, up to three codons, get 13% 
- bases+scores to score gives best performance 26% R^2
  - try changing their lengths independently
- All this is local effect, previous codons. What if I look into nonlocal effects. Can I boost the 13%/27%?
  
#### Codon prevalence and GERP
Generally rarer codon has lower GERP. true within a family of synonmous codons for last position - not clear further 
pattern.
Seems codon bias affecting expression is known in literature, as is 
[GERP variability](https://www.biorxiv.org/content/10.1101/688051v1.full).

### Artificial genome
I did analysis on "artificial genome", creating some random genetic code and reverse translating English text, then
doing K-mer analysis. As expected, there is no pattern. 2-mers independent of frame - makes sense, because the frame
for k=2 is out of sync with the 3-mer pattern, and will average out to the same.
for k=3 however, as expected, the count is heavily dependent on the frame, with 10 fold variation normal, and even
200 fold occurring once.

    Such massive change doesn't happen in actual genetic code, and genome. Rather it is minor change between frames.

Todo: check variation in count / score between frames, as compared to a random code and text. 
Is it optimized in this sense? How so?

### Protein Families
- Tyr protein kinase - receptor vs  Tyr protein kinase (presumably non receptor). the former  is more conserved. seems 
sensible, since it has extra receptor-like features an extracellular ligand-binding domain and a transmembrane-spanning 
region - keep looking

### Misc
- Standard error of GERP is tiny, and standard dev is quite large and doesn't change much, somewhat useless
  - It gets smaller with further breakdown  
- ft_start - seems irrelevant. Holding all else constant and varying it changes almost nothing. 
- Sanity check: `dfx = aggregate_over_additive_field(df3, 'kmer')`, passes. combinations with same frame + pos on the 
same strand have same score mean. 
- Something doesn't make sense, aggregating over columns that end up being irrelevant, e.g. ft_len, ft_start, makes the 
the score standard deviations artificially small. Probably because my std combination formula assumes independence, 
which is clearly incorrect - how to resolve? By taking into account dependence, explicitly .... or by construction 
calculating the std for every possible aggregation? But is this right? they are supposed to be independent actually 
- Read GERP Paper - done
- 

## Paths Forward
### Computations
- Split input k-fold, check how different the computed values are - I am especially interested in frames 1 and 2, which 
depend on how codons are used together.
- Find smallest averaging window that obeys these symmetries
  - The symmetry even between A and T, as well as a  and t .... presumably masking is region based ... this implies some 
  locality to the origin of the symmetry
- Do analysis within single feature - CDS, gene. E.g. kmer and its RC within same CDS
- Do within type of subfeatures that appears in the label or feature name (e.g. promoter)
- construct de brun graph with quantities from actual sequences ... measures some kind of robustness under frameshift 
  - for all codons, and translate them... 64 x 4 matrix ... since only 4 codons could follow another
  - [Frameshift and wild-type proteins are often highly similar because the genetic code and genomes were optimized for 
     frameshift tolerance](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-022-08435-6)
  - compare the observed values of said matrix with theoretical ones, for random, and what else?
  - compare sequence of amino acids in each frame
- Seems rarer codons are more variable between species ... find interpretation and meaning of this
  between other codons, any other pattern? scatterplot frequency vs gerp score.

### Theory
- What does reverse complement equality imply about the genetic code and the allowed sequences of 
amino acids? Given (similar?) frame-wise frequency, what does this impose on the aforementioned? This is profound. Compare it
to artificial genome. 
    - not much because it is reverse complement with the strand as well. So always the coding strand
- Perhaps non standard genetic code is reflected in the sequences themselves. That code and sequences have to go together 
to conserve these higher properties.
- Find nice symmetric order of rc pairs - rather than currently used canonical ordering
- Whatever properties one explores for genetic code, they could also apply for alternate genetic code - check there
  - Ways in which genetic code varies ... what is allowed? 
  - What is the best visual representation that shows how code can be varied and parametrized?
  - Does such parametrization vary with prevalence and arrangement of codons in actual sequences?
  - 

### Motif language analog
- Motifs vs GERP score https://en.wikipedia.org/wiki/Sequence_motif (seems I need more detailed annotation)
- Analogy with Quranic patterns ....
    - semilocal motifs that occur several times (slightly changed) throughout a given range
    - repeats and almost repeats with slightly different structure but same meaning in distant parts
    - rearranged pieces when same theme occurs in different part
- The above is only possible because I understand language, have concept of words, sentences, meanings, verbs, nouns, 
pronouns, subject, object .... in the genome we just have k-mer, codon, and annotation .... need to find intermediate 
concepts in the literature, and discover some myself ...fundamentally a data analysis problem.

### Protein families
- Check GERP score between families - does it change drastically between them? Even when stratified by codons?
- Is it closer between subfamilies (find hierarchies with common levels) - write code for it
- Where I have "subunit" information, are the GERP results consistent with function of subunit
- Families with common description e.g. "transporter", common name only differ in number TMEM##
- If I add protein family info as another feature, can I better predict GERP?

### Data expansion
- Whole human genome
- Mitochondria (since it apparently may not obey chargaff's - but is that only due to smaller size?)
- Analyze Camel DNA افلا ينظرون الى الابل كيف خلقت https://ftp.ensembl.org/pub/current_fasta/camelus_dromedarius/dna/

### New projects
- Variation between gametes - find gamete datasets
- Variation between gene/protein family members
- Directionality of variation - not simply this locus varies, what does it tend to change to? 
    does one k-mer tend to turn to another like CG becomes ...? 
    does that depend on region type? would be very interesting if so - and show it is a complex genetic function, not 
    just innate local tendency depending on conditions
    Would I need a dataset within a given species but in different conditions?


## Paths completed
- Distance vs mutual information: done in `score_stats_by_dilated_kmer` and `mutual_information_by_dilation`.
- Calculate degrees of freedom in determinig k-mer counts, given the intrinstic de Bruyne graph relationship and 
   the k-mer reverse complement relationship.
- Can the count symmetry, in conjunction with some other effect, be producing the GERP one somehow? No - I calculated
    the correlations between GERP score and counts, and they are non existent. Not to mention the later discovered fact
    that Count reverse complement relation holds only for frame, while GERP holds for frame, strand, and position
- Make frame out of 3 for all. 3 is only quantity for which frame is meaningful: done
- Conduct analysis within different features (CDS, GENE, UTR) - how do frequences or GERP scores change? - done: answers 
above
- Score correlations of adjacent (or dilated) nucleotides: similar pattern to score, 3-periodic, especially for frame 2
- Check the choice of the third nucleotide in CDS - making its own sequence - based on dilation ... mutual info
- Check synonmous codons, do differences in their GERP imply something about prevalence? yes, rarer have lower GERP in 
  pos 2.


## References
- [Dinucleotide Property Database](https://diprodb.leibniz-fli.de/ShowTable.php).
- [On Normalized Mutual Information](https://pdfs.semanticscholar.org/7914/8020ef42aff36f0649bccc94c9711f9b884f.pdf)
- [Evolution of k-mer frequencies and entropy](https://par.nsf.gov/servlets/purl/10152367)
- [Google Scholar](https://scholar.google.com/scholar?start=20&q=k+mer+distribution&hl=en&as_sdt=0,5)
- [Genomic DNA k-mer spectra](https://link.springer.com/article/10.1186/gb-2009-10-10-r108)
- [Estimation of genomic characteristics by analyzing k-mer frequency](https://arxiv.org/abs/1308.2012)
- [K-mer natural vector and its application to the phylogenetic analysis of genetic sequences](https://www.sciencedirect.com/science/article/abs/pii/S0378111914006064)
