SEPARATION_LEVEL="paragraph"
DATA_DIR="../data/echr/annotation/for_training"

echo "Creating embedded representation"
for PARTITION_DIR in $(compgen -f $DATA_DIR/partition)
do
	echo "Processing partition $PARTITION_DIR"
    # This command creates the representation (with embeeddings and mappings) necessary
    # for the ukpnets inputs
    python preprocess/ukpnets_process.py --embeddings_path ../data/wordvectors/komninos_english_embeddings.gz --output_dirpath $PARTITION_DIR/$SEPARATION_LEVEL --dataset $PARTITION_DIR/$SEPARATION_LEVEL --name ukp
done
