# ================================
# 빌드 스테이지
# ================================
FROM debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 빌드 도구 설치
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        git \
        dpkg-dev \
        gcc \
        g++ \
        clang \
        libc6-dev \
        make \
        autoconf \
        cmake \
        clang-format \
        python3 \
        python3-pip \
        lsb-release \
        libssl-dev \
        libcurl4-openssl-dev \
        libthrift-dev \
        libboost-dev \
        zlib1g-dev \
        libbz2-dev \
        liblz4-dev \
        libsnappy-dev \
        libzstd-dev \
        libre2-dev \
        libutf8proc-dev \
    ; \
    rm -rf /var/lib/apt/lists/*

# Apache Arrow 빌드 및 설치
RUN set -eux; \
    git clone https://github.com/apache/arrow.git /tmp/arrow; \
    cd /tmp/arrow/cpp; \
    git checkout apache-arrow-21.0.0; \
    mkdir build; \
    cd build; \
    cmake .. \
        -DARROW_IPC=ON \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX=/opt/arrow \
        -DARROW_ACERO="ON" \
        -DARROW_BUILD_STATIC="OFF" \
        -DARROW_COMPUTE="ON" \
        -DARROW_CSV="ON" \
        -DARROW_DATASET="ON" \
        -DARROW_FILESYSTEM="ON" \
        -DARROW_JSON="ON" \
        -DARROW_PARQUET="ON" \
        -DARROW_WITH_BZ2="ON" \
        -DARROW_WITH_RE2="ON" \
        -DARROW_WITH_SNAPPY="ON" \
        -DARROW_WITH_UTF8PROC="ON" \
        -DARROW_WITH_ZLIB="ON" \
        -DARROW_WITH_ZSTD="ON" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS="ON"; \
    make -j$(nproc); \
    make install; \
    cd /tmp; \
    rm -rf arrow

# AWS SDK C++ 빌드 및 설치
RUN set -eux; \
    cd /tmp; \
    git clone --recursive --branch 1.11.555 https://github.com/aws/aws-sdk-cpp.git; \
    cd aws-sdk-cpp; \
    mkdir build; \
    cd build; \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/aws \
        -DBUILD_ONLY="core;s3;transfer" \
        -DENABLE_TESTING=OFF \
        -DCPP_STANDARD=17 \
        -DBUILD_SHARED_LIBS=ON; \
    make -j$(nproc); \
    make install; \
    cd /tmp; \
    rm -rf aws-sdk-cpp

# seahorsedb 소스 복사 및 빌드
COPY . /usr/src/seahorsedb
RUN mkdir -p /usr/src/seahorsedb_build
WORKDIR /usr/src/seahorsedb_build/

# 라이브러리 경로 설정
ENV PKG_CONFIG_PATH="/opt/arrow/lib/pkgconfig:/opt/aws/lib/pkgconfig"
ENV LD_LIBRARY_PATH="/opt/arrow/lib:/opt/aws/lib"
ENV CMAKE_PREFIX_PATH="/opt/arrow:/opt/aws"

RUN set -eux; \
    \
# https://github.com/jemalloc/jemalloc/issues/467 -- we need to patch the "./configure" for the bundled jemalloc to match how Debian compiles, for compatibility
# (also, we do cross-builds, so we need to embed the appropriate "--build=xxx" values to that "./configure" invocation)
    gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
    extraJemallocConfigureFlags="--build=$gnuArch"; \
# https://salsa.debian.org/debian/jemalloc/-/blob/c0a88c37a551be7d12e4863435365c9a6a51525f/debian/rules#L8-23
    dpkgArch="$(dpkg --print-architecture)"; \
    case "${dpkgArch##*-}" in \
        amd64 | i386 | x32) extraJemallocConfigureFlags="$extraJemallocConfigureFlags --with-lg-page=12" ;; \
        *) extraJemallocConfigureFlags="$extraJemallocConfigureFlags --with-lg-page=16" ;; \
    esac; \
    extraJemallocConfigureFlags="$extraJemallocConfigureFlags --with-lg-hugepage=21"; \
    grep -F 'CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure ' /usr/src/seahorsedb/deps/CMakeLists.txt; \
    sed -ri 's!CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure !&'"$extraJemallocConfigureFlags"' !' /usr/src/seahorsedb/deps/CMakeLists.txt; \
    grep -F "CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure $extraJemallocConfigureFlags " /usr/src/seahorsedb/deps/CMakeLists.txt; \
    \
    cmake /usr/src/seahorsedb; \
    export BUILD_TLS=yes; \
    make -j "$(nproc)" all; \
    mv /usr/src/seahorsedb_build/seahorse-server /usr/local/bin/seahorse-server; \
    mv /usr/src/seahorsedb_build/seahorse-cli /usr/local/bin/seahorse-cli; \
    \
    seahorse-cli --version; \
    seahorse-server --version

# ================================
# 런타임 스테이지
# ================================
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# add our user and group first to make sure their IDs get assigned consistently, regardless of whatever dependencies get added
RUN set -eux; \
    groupadd -r -g 999 seahorse; \
    useradd -r -g seahorse -u 999 seahorse

# runtime dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
# add tzdata explicitly for https://github.com/docker-library/redis/issues/138 (see also https://bugs.debian.org/837060 and related)
		tzdata \
        ca-certificates \
        libcurl4 \
        libssl3 \
        libuuid1 \
        zlib1g \
        libbz2-1.0 \
        libsnappy1v5 \
        libzstd1 \
        liblz4-1 \
        libre2-9 \
        libthrift-0.17.0 \
        libutf8proc2 \
    ; \
    rm -rf /var/lib/apt/lists/*

# grab gosu for easy step-down from root
# https://github.com/tianon/gosu/releases
ENV GOSU_VERSION=1.17
RUN set -eux; \
    savedAptMark="$(apt-mark showmanual)"; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates gnupg wget; \
    rm -rf /var/lib/apt/lists/*; \
    arch="$(dpkg --print-architecture | awk -F- '{ print $NF }')"; \
    case "$arch" in \
        'amd64') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-amd64'; sha256='bbc4136d03ab138b1ad66fa4fc051bafc6cc7ffae632b069a53657279a450de3' ;; \
        'arm64') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-arm64'; sha256='c3805a85d17f4454c23d7059bcb97e1ec1af272b90126e79ed002342de08389b' ;; \
        'armel') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-armel'; sha256='f9969910fa141140438c998cfa02f603bf213b11afd466dcde8fa940e700945d' ;; \
        'i386') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-i386'; sha256='087dbb8fe479537e64f9c86fa49ff3b41dee1cbd28739a19aaef83dc8186b1ca' ;; \
        'mips64el') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-mips64el'; sha256='87140029d792595e660be0015341dfa1c02d1181459ae40df9f093e471d75b70' ;; \
        'ppc64el') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-ppc64el'; sha256='1891acdcfa70046818ab6ed3c52b9d42fa10fbb7b340eb429c8c7849691dbd76' ;; \
        'riscv64') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-riscv64'; sha256='38a6444b57adce135c42d5a3689f616fc7803ddc7a07ff6f946f2ebc67a26ba6' ;; \
        's390x') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-s390x'; sha256='69873bab588192f760547ca1f75b27cfcf106e9f7403fee6fd0600bc914979d0' ;; \
        'armhf') url='https://github.com/tianon/gosu/releases/download/1.17/gosu-armhf'; sha256='e5866286277ff2a2159fb9196fea13e0a59d3f1091ea46ddb985160b94b6841b' ;; \
        *) echo >&2 "error: unsupported gosu architecture: '$arch'"; exit 1 ;; \
    esac; \
    wget -O /usr/local/bin/gosu.asc "$url.asc"; \
    wget -O /usr/local/bin/gosu "$url"; \
    echo "$sha256 */usr/local/bin/gosu" | sha256sum -c -; \
    export GNUPGHOME="$(mktemp -d)"; \
    gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4; \
    gpg --batch --verify /usr/local/bin/gosu.asc /usr/local/bin/gosu; \
    gpgconf --kill all; \
    rm -rf "$GNUPGHOME" /usr/local/bin/gosu.asc; \
    apt-mark auto '.*' > /dev/null; \
    [ -z "$savedAptMark" ] || apt-mark manual $savedAptMark > /dev/null; \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
    chmod +x /usr/local/bin/gosu; \
    gosu --version; \
    gosu nobody true

# 빌드된 라이브러리들 복사
COPY --from=builder /opt/arrow /opt/arrow
COPY --from=builder /opt/aws /opt/aws

# 빌드된 바이너리 복사
COPY --from=builder /usr/local/bin/seahorse-server /usr/local/bin/seahorse-server
COPY --from=builder /usr/local/bin/seahorse-cli /usr/local/bin/seahorse-cli

# 라이브러리 경로 설정
ENV PKG_CONFIG_PATH="/opt/arrow/lib/pkgconfig:/opt/aws/lib/pkgconfig"
ENV LD_LIBRARY_PATH="/opt/arrow/lib:/opt/aws/lib"
ENV CMAKE_PREFIX_PATH="/opt/arrow:/opt/aws"

RUN mkdir /data && chown seahorse:seahorse /data
VOLUME /data
WORKDIR /data

# mimalloc memory settings
ENV MIMALLOC_PURGE_DELAY=0 \
    MIMALLOC_EAGER_COMMIT_DELAY=0 \
    MIMALLOC_PAGE_RESET=1

COPY docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["docker-entrypoint.sh"]

EXPOSE 5555
CMD ["seahorse-server"]
