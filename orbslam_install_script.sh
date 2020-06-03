#!/bin/bash

CURRENT_DIR=$(pwd)
MAINDIR=$(pwd)/3rdparty_habitat
CONDA_ENV_NAME=$CONDA_DEFAULT_ENV

# Ensure this is before anaconda's libs
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu

# clean before
rm -rf ${MAINDIR}
mkdir ${MAINDIR}
cd ${MAINDIR}

# TODO add create env from file
# source activate
# Make sure the env is not already activated
# source deactivate
# RUN THIS MANUALLY
# source activate ${CONDA_ENV_NAME}

# These should be already in conda.
# conda install opencv boost -y
# conda install pytorch torchvision -c pytorch -y
# conda install -c conda-forge imageio -y
# conda install ffmpeg -c conda-forge -y

# ========================================================
# Install Eigen
# ========================================================
cd ${MAINDIR}
mkdir eigen3
cd eigen3
wget http://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz
tar -xzf 3.3.5.tar.gz
cd eigen-eigen-b3f3d4950030
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${MAINDIR}/eigen3_installed/
make install

# ========================================================
# Install GLEW
# ========================================================
cd ${MAINDIR}
wget https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.zip
unzip glew-2.1.0.zip
cd glew-2.1.0/
cd build
cmake ./cmake  -DCMAKE_INSTALL_PREFIX=${MAINDIR}/glew_installed
make -j4
make install
cd ${MAINDIR}


# ========================================================
# Install Pangolin (clean)
# ========================================================
rm Pangolin -rf
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=${MAINDIR}/glew_installed/ -DCMAKE_LIBRARY_PATH=${MAINDIR}/glew_installed/lib/ -DCMAKE_INSTALL_PREFIX=${MAINDIR}/pangolin_installed  -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF
make -j8
make install
cd ${MAINDIR}

# ========================================================
# Install ORB SLAM2 (D Mishkin)
# (https://github.com/ducha-aiki/ORB_SLAM2)
# ========================================================
rm ORB_SLAM2 -rf
rm ORB_SLAM2-PythonBindings -rf
git clone https://github.com/ducha-aiki/ORB_SLAM2
git clone https://github.com/ducha-aiki/ORB_SLAM2-PythonBindings
cd ${MAINDIR}/ORB_SLAM2
sed -i "s,cmake .. -DCMAKE_BUILD_TYPE=Release,cmake .. -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=${MAINDIR}/eigen3_installed/include/eigen3/ -DCMAKE_INSTALL_PREFIX=${MAINDIR}/ORBSLAM2_installed ,g" build.sh
ln -s ${MAINDIR}/eigen3_installed/include/eigen3/Eigen ${MAINDIR}/ORB_SLAM2/Thirdparty/g2o/g2o/core/Eigen
./build.sh
cd build
make install
cd ${MAINDIR}
cd ORB_SLAM2-PythonBindings/src
ln -s ${MAINDIR}/eigen3_installed/include/eigen3/Eigen Eigen
# Make the python bindings
cd ${MAINDIR}/ORB_SLAM2-PythonBindings
mkdir build
cd build
sed -i "s,lib/python3.5/dist-packages,${CONDA_PREFIX}/lib/python3.6/site-packages,g" ../CMakeLists.txt
# CMAKE install
cmake .. -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.6m.so -DPYTHON_EXECUTABLE:FILEPATH=`which python` -DCMAKE_LIBRARY_PATH=${MAINDIR}/ORBSLAM2_installed/lib -DCMAKE_INCLUDE_PATH=${MAINDIR}/ORBSLAM2_installed/include;${MAINDIR}/eigen3_installed/include/eigen3 -DCMAKE_INSTALL_PREFIX=${MAINDIR}/pyorbslam2_installed 
make
make install

# TODO: make sure you have boost for conda, and optionally include the boost path (/home/ubuntu/anaconda3/envs/habitat_p36/include/boost) into the -DCMAKE_INCLUDE_PATH argument in the cmake
# TODO: make sure you have opencv version 2 or 3
# TODO: also need to change the CMakeLists.txt for ORB_SLAM2-PythonBindings, change boost required component to python36, not python-py36
# TODO: also have to copy the thirdparty g2o, dbow2 .so files to site-packages if required
# TODO: may need to change the <pyboostcvconverter/pyboostcvconverter.hpp> to "pyboostcvconverter/pyboostcvconverter.hpp"

# Copy the ORB Vocabulary file
cp ${MAINDIR}/ORB_SLAM2/Vocabulary/ORBvoc.txt ${CURRENT_DIR}/data/
