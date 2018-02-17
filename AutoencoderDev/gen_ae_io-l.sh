rm -r /hmewald/Autoencoders/ModelIO
mkdir /hmewald/Autoencoders/ModelIO
mkdir /hmewald/Autoencoders/ModelIO/TrainingImages
mkdir /hmewald/Autoencoders/ModelIO/TestImages

num_max=$1
python trainAE.py io-l /hmewald/Autoencoders/ModelIO/ /hmewald/DatasetTraining/VideoIOL /hmewald/DatasetTest/VideoIOL $num_max
