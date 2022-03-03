
FROM tensorflow/serving

ENV MODEL_NAME=diamonds
ENV MODEL_FOLDER=/model/stored_model/
ENV MODEL_STORAGE='gs://stored_model/stored_model/1'



RUN apt-get update && apt-get install -y curl gnupg

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

RUN mkdir /models/diamonds

RUN gsutil cp -r $MODEL_STORAGE /models/diamonds

CMD tensorflow_model_server --port=8050 --rest_api_port=8501 --model_base_path=/models/diamonds/model --model_name='DIAMONDS'
