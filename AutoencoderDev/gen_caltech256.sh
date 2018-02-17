rm -r /home/hmewald/Desktop/Dorsa/AutoencoderResults
mkdir /home/hmewald/Desktop/Dorsa/AutoencoderResults
mkdir /home/hmewald/Desktop/Dorsa/AutoencoderResults/TrainingImages
mkdir /home/hmewald/Desktop/Dorsa/AutoencoderResults/TestImages

python trainAE.py american-flag /home/hmewald/Desktop/Dorsa/AutoencoderResults/ /home/hmewald/Desktop/Dorsa/DatasetTraining /home/hmewald/Desktop/Dorsa/DatasetTest 128
