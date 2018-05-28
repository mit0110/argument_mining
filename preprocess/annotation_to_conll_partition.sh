CONFIG_NAME="config4"
DATA_DIR="../../data/echr/annotation/"
cd preprocess
echo "Creating intermidiate representation"
mkdir $DATA_DIR/$CONFIG_NAME/
python arg_docs2conll.py --input_dirpath $DATA_DIR/training/ --output_file $DATA_DIR/$CONFIG_NAME/train.p
python arg_docs2conll.py --input_dirpath $DATA_DIR/validation/ --output_file $DATA_DIR/$CONFIG_NAME/dev.p
python arg_docs2conll.py --input_dirpath $DATA_DIR/holdout/ --output_file $DATA_DIR/$CONFIG_NAME/test.p
echo "Converting to conll format"
python build_conll.py --input_filename $DATA_DIR/$CONFIG_NAME/train.p --output_filename $DATA_DIR/$CONFIG_NAME/train.txt --separation paragraph
python build_conll.py --input_filename $DATA_DIR/$CONFIG_NAME/dev.p --output_filename $DATA_DIR/$CONFIG_NAME/dev.txt --separation paragraph
python build_conll.py --input_filename $DATA_DIR/$CONFIG_NAME/test.p --output_filename $DATA_DIR/$CONFIG_NAME/test.txt --separation paragraph
cd ..
python preprocess/ukpnets_process.py --embeddings_path ../data/wordvectors/komninos_english_embeddings.gz --output_dirpath $DATA_DIR/$CONFIG_NAME/ --dataset $DATA_DIR/$CONFIG_NAME/