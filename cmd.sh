#!/bin/bash

module add use.own
module load python/3.7.0
module load pytorch/1.1.0

IMPORTS=(
	directory.tar.xz
    # ml.tar.xz
	# Use your appropriate one
)

# Head Node -> /home2/asvs
# G Node -> /home2/asvs

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="/home2/$USER"

mkdir -p $LOCAL_ROOT/{data,checkpoints,results}

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints
RESULTS=$LOCAL_ROOT/results

# rsync -r /home/shashanks/ilci/ $DATA/ilci/

rsync -rvz /home2/$USER/checkpoints/checkpoint_best.pt $CHECKPOINTS/

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}

rsync -vz /home2/$USER/datasets/complete-en-ml/ $DATA/complete-en-ml/
mv $DATA/sample_check/a.txt $DATA/complete-en-ml/test.ml-en.en

function _export {
    ssh $USER@ada "mkdir -p ada:/share1/$USER/checkpoints/pib"
    rsync -rvz $CHECKPOINTS/checkpoint_best.pt ada:/share1/$USER/checkpoints/pib/
}

# trap "_export" SIGHUP
copy
export ILMULTI_CORPUS_ROOT=$DATA

python3 preprocess_cvit.py config.yaml


ARCH='transformer'
MAX_TOKENS=3500
LR=1e-3
UPDATE_FREQ=128
MAX_EPOCHS=200

set -x
function train {
    python3 train.py \
        --task shared-multilingual-translation \
        --share-all-embeddings \
        --num-workers 0 \
        --arch $ARCH \
        --max-tokens $MAX_TOKENS --lr $LR --min-lr 1e-9 \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --save-dir $CHECKPOINTS \
        --log-format simple --log-interval 200 \
        --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
        --lr-scheduler inverse_sqrt \
        --clip-norm 0.1 \
        --ddp-backend no_c10d \
        --update-freq $UPDATE_FREQ \
        --max-epoch $MAX_EPOCHS \
        --criterion label_smoothed_cross_entropy \
        config.yaml

}

    #    --reset-optimizer \
    #    --reset-lr-scheduler \

function _test {
    python3 generate.py config.yaml \
        --task shared-multilingual-translation  \
        --path $CHECKPOINTS/checkpoint_last.pt > ufal-gen.out
    cat ufal-gen.out \
        | grep "^H" | sed 's/^H-//g' | sort -n | cut -f 3 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > ufal-test.hyp
    cat ufal-gen.out \
        | grep "^T" | sed 's/^T-//g' | sort -n | cut -f 2 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > ufal-test.ref

    split -d -l 2000 ufal-test.hyp hyp.ufal.
    split -d -l 2000 ufal-test.ref ref.ufal.

    # perl multi-bleu.perl ref.ufal.00 < hyp.ufal.00 
    # perl multi-bleu.perl ref.ufal.01 < hyp.ufal.01 

    python3 -m indicnlp.contrib.wat.evaluate \
        --reference ref.ufal.00 --hypothesis hyp.ufal.00 
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference ref.ufal.01 --hypothesis hyp.ufal.01 

}

function _backtranslate {
	python3 generate.py config.yaml \
          --task shared-multilingual-translation  \
          --path $CHECKPOINTS/checkpoint_best.pt > $RESULTS/output.txt
}

# ARG=$1
# eval "$1"
# # _test

# wait
# _export
_backtranslate
