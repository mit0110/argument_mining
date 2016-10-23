#### NLTK downloads

```
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
```

#### Using the Stanford Parser (with nltk 3.2.1)

1. Download the stanford parser from the official site.
2. Unzip the file in STANFORD_FOLDER
3. Run:
```
$ export STANFORDTOOLSDIR=STANFORD_FOLDER
$ export CLASSPATH=$STANFORDTOOLSDIR/stanford-parser-full-XXXX-XX-XX/stanford-parser.jar:$STANFORDTOOLSDIR/stanford-parser-full-XXXX-XX-XX/stanford-parser-3.6.0-models.jar:$STANFORDTOOLSDIR/stanford-parser-full-XXXX-XX-XX/slf4j-api.jar
```
