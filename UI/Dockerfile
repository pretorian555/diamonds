
FROM python:3.8
RUN pip install -U streamlit numpy pandas gcsfs
WORKDIR /app
COPY diamonds.py diamond-434.gif ./
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["diamonds.py"]
