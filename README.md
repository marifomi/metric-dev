1. Prepare data for evaluation

In case of ranking data:
Copy original data (in wmt format) to the working directory
class Ranking Data method read_data/write_data to concatenate multiple MT output files for different languages and systems
in a single file, and create the corresponding parallel reference file. In the resulting files each MT sentence correspond
to the human translation of the same source sentence in the reference. The outputs are sorted by dataset, language pair
and system name.
To do the same with parsed files use method write_parsed from utils.write_parsed.



