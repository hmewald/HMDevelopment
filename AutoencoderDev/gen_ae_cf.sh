rm -r /hmewald/Autoencoders/ModelCF
mkdir /hmewald/Autoencoders/ModelCF
mkdir /hmewald/Autoencoders/ModelCF/TrainingImages
mkdir /hmewald/Autoencoders/ModelCF/TestImages

num_max=$1
python trainAE.py cf /hmewald/Autoencoders/ModelCF/ /hmewald/DatasetTraining/VideoCF /hmewald/DatasetTest/VideoCF $num_max
