RELATIONS="_wrel"
SEPARATION_LEVEL="paragraph"
DATA_DIR="../../data/echr/annotation/for_training"

cd preprocess

echo "Creating intermidiate representation"

for PARTITION_DIR in $DATA_DIR/partition*
do
    echo "Processing $PARTITION_DIR"
    # These commands create a single list of AnnotatedDocuments from all the annotations
    # in the input_dirpath
    python arg_docs2conll.py --input_dirpath $PARTITION_DIR/train/ --output_file $PARTITION_DIR/train_docs${RELATIONS}.p
    python arg_docs2conll.py --input_dirpath $PARTITION_DIR/dev/ --output_file $PARTITION_DIR/dev_docs${RELATIONS}.p
    python arg_docs2conll.py --input_dirpath $PARTITION_DIR/test/ --output_file $PARTITION_DIR/test_docs${RELATIONS}.p
    # These commands write the AnnotatedDocuments as a single conll file
    # For script 03, they must be named train.txt, test.txt and dev.txt
    echo "Converting partition $PARTITION_DIR to conll format"
    mkdir $PARTITION_DIR/${SEPARATION_LEVEL}${RELATIONS}
    python build_conll.py --input_filename $PARTITION_DIR/train_docs${RELATIONS}.p --output_filename $PARTITION_DIR/${SEPARATION_LEVEL}${RELATIONS}/train.txt --separation $SEPARATION_LEVEL --include_relations
    python build_conll.py --input_filename $PARTITION_DIR/dev_docs${RELATIONS}.p --output_filename $PARTITION_DIR/${SEPARATION_LEVEL}${RELATIONS}/dev.txt --separation $SEPARATION_LEVEL --include_relations
    python build_conll.py --input_filename $PARTITION_DIR/test_docs${RELATIONS}.p --output_filename $PARTITION_DIR/${SEPARATION_LEVEL}${RELATIONS}/test.txt --separation $SEPARATION_LEVEL --include_relations
done
cd ..
