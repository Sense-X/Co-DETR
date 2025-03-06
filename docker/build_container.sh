docker stop co-detr
docker rm co-detr



#docker pull sorajang/gsam2



# Create a new container
docker run -i -t --gpus all -d \
--shm-size=12gb \
-v $HOME:/mnt/source \
-v /home/soratabata/datadrive:/mnt/source/datadrive \
--name="co-detr" co-detr bash

# Git pull orbslam and compile
docker exec -it co-detr bash -i -c "apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install libgl1-mesa-dev && \
    apt-get install libglib2.0-0 && \
    pip3 install mxnet-mkl==1.6.0 numpy==1.23.1"