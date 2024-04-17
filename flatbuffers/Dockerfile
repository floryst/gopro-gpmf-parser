FROM silkeh/clang:16 AS build

ENV VERSION=24.3.25
ENV CORES=8

WORKDIR /flatbuffers
ADD https://github.com/google/flatbuffers/archive/refs/tags/v${VERSION}.tar.gz /flatbuffers/flatbuffers.tar.gz

RUN <<EOF
tar -xf /flatbuffers/flatbuffers.tar.gz
mkdir /flatbuffers/build
EOF

WORKDIR /flatbuffers/build
RUN <<EOF
CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release /flatbuffers/flatbuffers-${VERSION}
make -j${CORES:-4}
mv flatc /usr/bin
EOF

WORKDIR /src
RUN rm -rf /flatbuffers