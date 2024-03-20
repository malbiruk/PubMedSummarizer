FROM python:3.10-slim
# RUN mkdir /streamlit
WORKDIR /streamlit
COPY requirements.txt /streamlit
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8505

RUN apt update
RUN apt update && apt install -yqq git g++ && \
    rm -rf /var/lib/apt/lists/*

COPY . /streamlit
CMD ["streamlit", "run", "web_interface.py", "--server.port", "8505", "--server.address", "0.0.0.0"]
