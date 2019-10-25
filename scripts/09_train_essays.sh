
# 

#Script to run the training in all configurations. Must be run from the
# argument mining folder

DATE=$(date +%y-%m-%d-%H-%M)
RESULT_DIRECTORY="/home/jfrau/am/results/essays2.0"
EXPERIMENT_DIRECTORY="ongoing-"$DATE
SEPARATION_LEVEL="sentence"
DATA_DIR="/home/jfrau/am/data/processed_essays2.0"
echo "******** Starting experiment $DATE"
ATTENTION_MODEL=$7
ATTENTION_ACTIVATION=$8
echo "******** Using attention " $ATTENTION_MODEL " " $ATTENTION_ACTIVATION

mkdir $RESULT_DIRECTORY/
mkdir $RESULT_DIRECTORY/$EXPERIMENT_DIRECTORY

DATASET_NAME=$DATA_DIR/processed_essays2.0_komninos_e.p
# The model resulting from this command will be saved in --output_dirpath + --experiment_name _model.h5
python -u experiments/train_BiLSTM_CNN_CRF.py \
    --dataset $DATASET_NAME \
    --attention_model $ATTENTION_MODEL \
    --attention_activation $ATTENTION_ACTIVATION \
    --output_dirpath $RESULT_DIRECTORY/$EXPERIMENT_DIRECTORY \
    --experiment_name essays2.0 \
    --char_embedding $1 \
    --char_embedding_size $2 \
    --epochs 50 \
    --classifier $3 \
    --patience 10 \
    --dropout $4 $4 \
    --batch_size $5 \
    --num_units $6 $6
# Now we need to evaluate the model
MODEL_NAME=$(compgen -f $RESULT_DIRECTORY/$EXPERIMENT_DIRECTORY/essays*h5)
echo "********* Evaluating model $MODEL_NAME"
python -u experiments/run_BiLSTM_CNN_CRF.py \
    --classifier $MODEL_NAME --dataset $DATASET_NAME \
    --attention_model $ATTENTION_MODEL \
    --output_dirname $RESULT_DIRECTORY/$EXPERIMENT_DIRECTORY \
    --experiment_name essays2.0

echo "********** All experiments completed"
mv $RESULT_DIRECTORY/$EXPERIMENT_DIRECTORY $RESULT_DIRECTORY/$DATE

