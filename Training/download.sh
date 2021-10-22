FILE=$1

# download celeba-HQ dataset
# dataset from StarGAN-v2 (https://github.com/clovaai/stargan-v2)
if  [ $FILE == "celeba-hq-dataset" ]; then
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=./data/celeba_hq.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    #rm $ZIP_FILE

# download celebaMask-HQ dataset & create final dataset for training
elif  [ $FILE == "celebaMask-hq-dataset" ]; then
    ZIP_FILE=./data/CelebAMask-HQ.zip
    unzip $ZIP_FILE -d ./data
    mv ./data/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt ./data/anno.txt
    mkdir ./data/celeba_hq/anno
    python create_dataset.py
    rm -r ./data/CelebAMask-HQ
    #rm $ZIP_FILE

else
    echo "Available arguments are celeba-hq-dataset and celebaMask-hq-dataset."
    exit 1

fi