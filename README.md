VERSION=2.4.5
wget https://github.com/singularityware/singularity/releases/download/$VERSION/singularity-$VERSION.tar.gz
tar xvf singularity-$VERSION.tar.gz
cd singularity-$VERSION
./configure --prefix=/usr/local
make
sudo make install
git clone https://github.com/lkmokadam/tensorflow-singularity.git
cd tensorflow-singularity
sudo singularity build lolcow.simg Singularity 
