FROM pytorch/torchserve:latest

# install dependencies
RUN pip3 install PIL

# copy model artifacts, custom handler and other dependencies
COPY ./custom_handler.py /home/model-server/
COPY ./index_to_name.json /home/model-server/
COPY ./models.py /home/model-server/
COPY ./checkpoints/checkpoint_num_39__21_11_2021__16_33_06.pth.tar /home/model-server
COPY ./vocabulary.py /home/model-server

# create torchserve configuration file
USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f --model-name=caption --version=1.0 --serialized-file=/home/model-server/checkpoint_num_39__21_11_2021__16_33_06.pth.tar --handler=/home/model-server/custom_handler.py --extra-files "/home/model-server/checkpoint_num_39__21_11_2021__16_33_06.pth.tar,/home/model-server/models.py,/home/model-server/vocabulary.py,/home/model-server/index_to_name.json" --export-path=/home/model-server/model-store


# run Torchserve HTTP serve to respond to prediction request
CMD ["torchserve", "--start", "--ts-config=/home/model-server/config.properties", "--models", "caption=caption.mar", "--model-store", "/home/model-server/model-store"]
