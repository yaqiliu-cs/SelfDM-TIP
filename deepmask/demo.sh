DEEPMASK=$PWD
export PYTHONPATH=$DEEPMASK:$PYTHONPATH
python tools/computeProposals.py --arch DeepMask --resume $DEEPMASK/pretrained/deepmask/DeepMask.pth.tar --img ./data/test.jpg
