cd /home/tchuang/research/Eyediap

for f in EYEDIAP*.tar.gz
do
    tar -zxvf "$f" --strip-components=4 -C /home/tchuang/research/Eyediap/Data
done
