# Script to run the training in all configurations. Must be run from the
# argument mining folder

DATE=$(date +%y-%m-%d-%H-%M)
RESULT_DIRECTORY="/home/jfrau/am/results/essays2.0"
EXPERIMENT_DIRECTORY="ongoing-"$DATE
DATA_DIR="/home/jfrau/am/data/processed_essays2.0"
echo "******** Starting experiment $DATE"
echo "******** Using self attention "

mkdir $RESULT_DIRECTORY/
mkdir $RESULT_DIRECTORY/$EXPERIMENT_DIRECTORY

DATASET_NAME=$DATA_DIR/processed_essays2.0_komninos_e.p
# The model resulting from this command will be saved in --output_dirpath + --experiment_name _model.h5
KERAS_BACKEND=tensorflow python -u experiments/train_SelfAtt_BiLSTM_CNN_CRF.py \
    --dataset $DATASET_NAME \
    --output_dirpath $RESULT_DIRECTORY/$EXPERIMENT_DIRECTORY \
    --experiment_name essays2.0_self_attention \
    --char_embedding $1 \
    --char_embedding_size $2 \
    --epochs 50 \
    --classifier $3 \
    --patience 10 \
    --dropout $4 $4 \
    --batch_size $5 \
    --num_units $6 $6 \
    --n_heads $7 \
    --attention_size $8

if [ $? -ne 0 ]; then { echo "******** Training failed" ; exit 1; } fi

echo "********** All experiments completed"
mv $RESULT_DIRECTORY/$EXPERIMENT_DIRECTORY $RESULT_DIRECTORY/$DATE

