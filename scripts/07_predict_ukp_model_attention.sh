# Script to read the model and apply it to all partitions.
# Replace EXPERIMENT_DIRECTORY with the name of the model.

RESULT_DIRECTORY="../results/ukpnets"
EXPERIMENT_DIRECTORY="18-08-30-03-41"
RELATIONS=""
SEPARATION_LEVEL="paragraph"
DATA_DIR="../data/echr/annotation/for_exploration"
echo "******** Starting classification with modell $EXPERIMENT_DIRECTORY"

ATTENTION_MODEL="feature_pre"

for PARTITION_DIR in $(compgen -f $DATA_DIR/partition)
do
    PARTITION=$(basename $PARTITION_DIR)
    DATASET_NAME=$(compgen -f $PARTITION_DIR/${SEPARATION_LEVEL}${RELATIONS}/ukp*.p)
    MODEL_NAME=$(compgen -f $RESULT_DIRECTORY/${SEPARATION_LEVEL}${RELATIONS}/$EXPERIMENT_DIRECTORY/$PARTITION*h5)
    echo "********* Evaluating model $MODEL_NAME"
    python -u experiments/run_BiLSTM_CNN_CRF.py \
        --classifier $MODEL_NAME --dataset $DATASET_NAME \
        --attention_model $ATTENTION_MODEL \
        --output_dirname $RESULT_DIRECTORY/${SEPARATION_LEVEL}${RELATIONS}/$EXPERIMENT_DIRECTORY \
        --experiment_name $PARTITION
done

echo "********** All experiments completed"
