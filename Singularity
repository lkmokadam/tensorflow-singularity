####
# Defines a Singularity container with GPU and MPI enabled TensorFlow
# https://www.tensorflow.org/install/install_sources#tested_source_configurations
####

BootStrap: docker
From: ubuntu:xenial

%environment
  export PATH=${PATH-}:/usr/lib/jvm/java-8-openjdk-amd64/bin/:/usr/local/cuda/bin
  export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
  export CUDA_HOME=/usr/local/cuda
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

%post
  apt update
  apt-get install -y software-properties-common
  apt-add-repository universe 
  add-apt-repository ppa:openjdk-r/ppa -y
  apt update
  apt install -y openjdk-8-jdk
  apt install -y build-essential wget curl pkg-config libtool autoconf g++ zip zlib1g-dev unzip git
  apt install -y python-numpy python-scipy python-dev python-pip python-setuptools
  apt install -y python3-numpy python3-scipy python3-dev python3-pip python3-setuptools
  apt install -y openjdk-8-jdk build-essential wget curl pkg-config libtool \
                  autoconf g++ zip zlib1g-dev unzip git \
                  python-numpy python-scipy python-dev python-pip python-setuptools \
                  python3-numpy python3-scipy python3-dev python3-pip python3-setuptools locales
  export LC_ALL="en_US.UTF-8"
  export LC_CTYPE="en_US.UTF-8"
  dpkg-reconfigure locales
  pip install --upgrade pip
  pip3 install --upgrade pip

  # Install CUDA toolkit and driver libraries/binaries
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install -y cuda
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

  # Install cuDNN
  wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
  tar xvfz cudnn-8.0-linux-x64-v6.0.tgz
  cp -P cuda/include/cudnn.h /usr/local/cuda/include
  cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
  chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

  # Clean up CUDA install
  rm -rf cudnn-8.0-linux-x64-v6.0.tgz

  # Install Bazel
  echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
  curl https://bazel.build/bazel-release.pub.gpg | apt-key add - 
  apt-get update 
  apt-get -y --allow-unauthenticated install bazel

  # Make sure no leftover tensorflow artifacts from previous builds
  rm -rf /tmp/tensorflow_pkg
  rm -rf /root/.cache

  # Set tensorflow configure options
  export TF_NEED_MKL=0
  export CC_OPT_FLAGS="-march=native"
  export TF_NEED_JEMALLOC=1
  export TF_NEED_GCP=0
  export TF_NEED_HDFS=0
  export TF_ENABLE_XLA=0
  export TF_NEED_OPENCL=0
  export TF_NEED_CUDA=1
  export TF_CUDA_CLANG=0
  export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
  export TF_CUDA_VERSION="8.0"
  export CUDA_TOOLKIT_PATH="/usr/local/cuda"
  export TF_CUDNN_VERSION="6.0.21"
  export CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
  export TF_CUDA_COMPUTE_CAPABILITIES="3.5"
  export TF_NEED_VERBS=0
  export TF_NEED_MPI=1
  export MPI_HOME=/usr
  export TF_NEED_GDR=0
  export TF_NEED_S3=0

  # Java cert update
  apt install ca-certificates-java
  update-ca-certificates -f

  git config --global user.email "help@olcf.ornl.gov"
  git config --global user.name "OLCF"

  # Build/Install Tensorflow against python 3
  export PYTHON_BIN_PATH=`which python3`
  export PYTHON_LIB_PATH=/usr/lib/python3/dist-packages

  git clone https://github.com/tensorflow/tensorflow.git
  cd tensorflow
  git checkout tags/v1.6.0
  ./configure 

  bazel build --local_resources 2048,2.0,1.0 -c opt --copt=-mavx --copt=-msse4.1 --copt=-msse4.2 --config=cuda tensorflow/tools/pip_package:build_pip_package
  bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

  pip3 install /tmp/tensorflow_pkg/tensorflow-*.whl

  cd /
  rm -rf tensorflow
  rm -rf /tmp/tensorflow_pkg

  # Install Additional deeplearning python packages

  pip3 install keras

  pip3 install scikit-learn

  apt install -y python3-theano

  # Install MPI4PY against mpich(python-mpi4py is built against OpenMPI)
  # GCC/4.8 is too old to acept the compile flags required by mpi4py
  pip3 install mpi4py

  # Install a few plotting libraries
  pip3 install matplotlib

  # Patch container to work on Titan
  wget https://raw.githubusercontent.com/olcf/SingularityTools/master/Titan/TitanBootstrap.sh
  sh TitanBootstrap.sh
  rm TitanBootstrap.sh

  # Make sure bazel is shutdown so it doesn't stop singularity from cleanly exiting
  bazel shutdown
  sleep 10
  pkill -f bazel*
  ps aux | grep bazel
