# start BCE-2015-fall from AWS console on a g2.2xlarge
# $0.65/hour; 4 Gb video RAM, 1536 CUDA cores

# make sure to increase space for home directory by requesting more when start instance, e.g. 30 Gb

# set variable holding IP address
# export ip=54-69-106-127

# ssh to the Amazon instance
# ssh -i ~/.ssh/ec2_rsa ubuntu@ec2-${ip}.us-west-2.compute.amazonaws.com

sudo su

# install CUDA
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5-18_amd64.deb
dpkg -i cuda-repo-ubuntu1504_7.5-18_amd64.deb

apt-get update
date >> /tmp/date
apt-get install -y cuda # a bit less than 10 mins
date >> /tmp/date

rm -rf cuda-repo-ubuntu1504_7.5-18_amd64.deb


# set up some utilities for monitoring the GPU
echo "" >> ~ubuntu/.bashrc
echo "export PATH=${PATH}:/usr/local/cuda/bin" >> ~ubuntu/.bashrc
echo "" >> ~ubuntu/.bashrc
echo "alias gtop=\"nvidia-smi -q -d UTILIZATION -l 1\"" >> ~ubuntu/.bashrc
echo "" >> ~ubuntu/.bashrc
echo "alias gmem=\"nvidia-smi -q -d MEMORY -l 1\"" >> ~ubuntu/.bashrc

# set up access to CUDA shared libraries
echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf
ldconfig

exit # back to ubuntu user

source ~/.bashrc

# reboot the instance

gtop

# gtop result without reboot will error:
#modprobe: ERROR: could not insert 'nvidia_352': No such device
#NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running. 

# create deviceQuery executable
sudo /usr/local/cuda/bin/nvcc /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery.cpp -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -o /usr/local/cuda/bin/deviceQuery

deviceQuery

# install PyCUDA
pip install pycuda

# install RCUDA
cd /tmp
git clone https://github.com/duncantl/RCUDA
git clone https://github.com/omegahat/RAutoGenRunTime

cd RCUDA/src
ln -s ../../RAutoGenRunTime/src/RConverters.c .
ln -s ../../RAutoGenRunTime/inst/include/RConverters.h .
ln -s ../../RAutoGenRunTime/inst/include/RError.h .

cd ../..

R CMD build RCUDA
R CMD build RAutoGenRunTime
R CMD INSTALL RAutoGenRunTime_0.3-0.tar.gz 
R CMD INSTALL RCUDA_0.4-0.tar.gz 


#### Create image ##########################

# 1) now save the image in us-west-2 via point and click on VM page under Actions
# 2) make it public

