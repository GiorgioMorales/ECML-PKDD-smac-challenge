# Start from a pytorch image
FROM pytorch/pytorch
WORKDIR /usr/local/app

RUN apt-get update
RUN apt-get install -y git


RUN pip install -r requirements.txt

CMD [ "python", "evaluation_phase.py" ]