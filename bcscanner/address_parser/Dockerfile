FROM centos:latest

RUN yum -y install curl git make autoconf automake libtool pkgconfig
RUN yum update -y \
    && yum install -y https://centos7.iuscommunity.org/ius-release.rpm \
    && yum install -y python36u python36u-libs python36u-devel python36u-pip \
    && yum install -y gcc-c++ jsoncpp-devel

RUN python3.6 --version
RUN pip3.6 --version

ARG ROOT_DIR=/workdir
RUN mkdir -p ${ROOT_DIR}
WORKDIR ${ROOT_DIR}

RUN mkdir -p ${ROOT_DIR}/data

RUN git clone https://github.com/openvenues/libpostal && \
    cd libpostal && \
    ./bootstrap.sh && \
    ./configure --datadir=${ROOT_DIR}/data && \
    make -j4 && \
    make install && \
    ldconfig

RUN pip3.6 install postal
RUN pip3.6 install flask

COPY address_parser.py .

ENV LD_LIBRARY_PATH=/usr/local/lib

ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["python3.6", "address_parser.py"]

