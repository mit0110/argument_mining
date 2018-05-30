# Script to run the training in all configurations. Must be run from the
# argument mining folder

DATE=$(date +%y-%m-%d-%H-%M)
RESULT_DIRECTORY="../results/ukpnets"
EXPERIMENT_DIRECTORY="ongoing-"$DATE
SEPARATION_LEVEL="paragraph"
DATA_DIRECTORY="../data/echr/annotation/for_training/"

mkdir $RESULT_DIRECTORY/$SEPARATION_LEVEL/
mkdir $RESULT_DIRECTORY/$SEPARATION_LEVEL/$EXPERIMENT_DIRECTORY

for PARTITION_DIR in $DATA_DIR/partition*
do
    PARTITION=$(basename PARTITION_DIR)
    DATASET_NAME=$(compgen -f $PARTITION_DIR/$SEPARATION_LEVEL/ukp*.p)
    # The model resulting from this command will be saved in --output_dirpath + --experiment_name _model.h5
    python -u experiments/train_BiLSTM_CNN_CRF.py \
        --dataset $DATASET_NAME \
        --output_dirpath $RESULT_DIRECTORY/$SEPARATION_LEVEL/$EXPERIMENT_DIRECTORY \
        --experiment_name $PARTITION \
        --char_embedding lstm \
        --epochs 100 \
        --classifier CRF
        # --num_units 100,100 \
        # --dropout 0.5,0.5 \
        # --char_embedding_size 30 \
        # --batch_size 32 \
        # --patience 5 \
    # Now we need to evaluate the model
    MODEL_NAME=$(compgen -f $RESULT_DIRECTORY/$SEPARATION_LEVEL/$EXPERIMENT_DIRECTORY/$PARTITION/$experiment_name*h5)
    python -u experiments/run_BiLSTM_CNN_CRF.py \
        --classifier $MODEL_NAME --dataset $DATASET_NAME \
        --output_dirname $RESULT_DIRECTORY/$SEPARATION_LEVEL/$EXPERIMENT_DIRECTORY
done