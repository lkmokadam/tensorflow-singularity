Bootstrap: docker
From: nvidia/cuda:9.0-base-ubuntu16.04
IncludeCmd: yes # Use the CMD as runscript instead of ENTRYPOINT


%runscript

    echo "TF..."

%post
 
   #echo "Here we are installing software and other dependencies for the container!"
   apt-get update
   apt-get install -y git 
   apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-dev-9-0 \
        cuda-cudart-dev-9-0 \
        cuda-cufft-dev-9-0 \
        cuda-curand-dev-9-0 \
        cuda-cusolver-dev-9-0 \
        cuda-cusparse-dev-9-0 \
        curl \
        git \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        libcudnn7-dev=7.0.5.15-1+cuda9.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

        curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py

        
	pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        && \
        python -m ipykernel.kernelspec

	# Set up Bazel.

	# Running bazel inside a `docker build` command causes trouble, cf:
	#   https://github.com/bazelbuild/bazel/issues/134
	# The easiest solution is to set up a bazelrc file forcing --batch.
	echo "startup --batch" >>/etc/bazel.bazelrc
	
	# Similarly, we need to workaround sandboxing issues:
	#   https://github.com/bazelbuild/bazel/issues/418
	echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >>/etc/bazel.bazelrc
	#   Install the most recent bazel release.
	export BAZEL_VERSION 0.8.0
	cd /
	mkdir /bazel && \
        cd /bazel && \
        curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
        curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
        chmod +x bazel-*.sh && \
        ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
        cd / && \
        rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

	# Download and build TensorFlow.
	cd /tensorflow
	git clone --branch=r1.5 --depth=1 https://github.com/tensorflow/tensorflow.git .

	# Configure the build for our CUDA configuration.
	export CI_BUILD_PYTHON python
	export LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
	export TF_NEED_CUDA 1
	export TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1,7.0
	export TF_CUDA_VERSION=9.0
	export TF_CUDNN_VERSION=7
	export TF_NEED_MPI 1

	ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
        LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
        tensorflow/tools/ci_build/builds/configured GPU \
        bazel build -c opt --config=cuda -c opt     --config=mkl \
            --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
            tensorflow/tools/pip_package:build_pip_package && \
        rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
        pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
        rm -rf /tmp/pip && \
        rm -rf /root/.cache
	

        cd /root
