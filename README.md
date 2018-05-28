##Setting the environment

### NLTK downloads

```
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
>>> nltk.download('wordnet')
```

### Using the Stanford Parser (with nltk 3.2.1)

1. Download the stanford parser from the official site.
2. Unzip the file in STANFORD_FOLDER
3. Run:
```bash
$ export STANFORDTOOLSDIR=STANFORD_FOLDER
$ export CLASSPATH=$STANFORDTOOLSDIR/stanford-parser-full-XXXX-XX-XX/stanford-parser.jar:$STANFORDTOOLSDIR/stanford-parser-full-XXXX-XX-XX/stanford-parser-3.6.0-models.jar:$STANFORDTOOLSDIR/stanford-parser-full-XXXX-XX-XX/slf4j-api.jar
```

##Experiments

### Using the sequential classifiers

For brat annotations, run the following pipeline:

```bash
$ cd preprocess
$ python process_arg_essays_for_conll.py --input_dirpath INPUT_DIRPATH --output_file PICKLED_DOCUMENTS
```

INPUT_DIRPATH must contain the .txt and .ann files to process. This will
generate a pickled file PICKLED_DOCUMENTS with an internal representation of the text
files.

```bash
$ cd ../experiments
$ python crf_baseline.py --input_filename PICKLED_DOCUMENTS
```

Now check the file `logs/log-crf` for your results!

### Using the neural classifiers

To run the script, clone the repository
https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf.git
under the name ukplab_nets and add it the path to PYTHONPATH.

```bash
export PYTHONPATH=$PYTHONPATH:/home/.../path_to_ukplab_nets:/home/.../path_to_ukplab_nets/ukplab_nets
```

Install Keras 2.1.5 and Tensorflow 1.7 with pip

```bash
pip install keras==2.1.5
```

To run the preprocess use

python -m preprocess.ukpnets_process