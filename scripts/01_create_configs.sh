DATA_DIR="../../data/echr/annotation/for_training"

FILES=( $(ls $DATA_DIR | sed 's/.\{4\}$//' | uniq) )
PARTITION_NUMBER=0
echo $FILES
for test_filename in "${FILES[@]}"
do
    echo "Partition $PARTITION_NUMBER"
    mkdir "$DATA_DIR/partition$PARTITION_NUMBER"
    mkdir "$DATA_DIR/partition$PARTITION_NUMBER/test"
    cp $DATA_DIR/$test_filename.* "$DATA_DIR/partition$PARTITION_NUMBER/test"
    echo "Test filename $test_filename"

    mkdir "$DATA_DIR/partition$PARTITION_NUMBER/dev"
    rand=$[$RANDOM % ${#FILES[@]}]
    dev_filename=${FILES[$rand]}
    cp $DATA_DIR/$dev_filename.* "$DATA_DIR/partition$PARTITION_NUMBER/dev"
    echo "Dev filename $dev_filename"

    mkdir "$DATA_DIR/partition$PARTITION_NUMBER/train"

    for train_filename in "${FILES[@]}"
    do
        if [ $test_filename != $train_filename ] && [ $dev_filename != $train_filename ]; then
            echo "copying $train_filename for Training"
            cp $DATA_DIR/$train_filename* "$DATA_DIR/partition$PARTITION_NUMBER/train"
        fi
    done

    PARTITION_NUMBER=`expr $PARTITION_NUMBER + 1`
done

