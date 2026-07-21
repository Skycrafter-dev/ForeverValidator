# syntax=docker/dockerfile:1

FROM debian:trixie-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY . .

RUN make -j"$(nproc)"


FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system forevervalidator \
    && useradd --system --gid forevervalidator --no-create-home --shell /usr/sbin/nologin forevervalidator

COPY --from=build /src/build/native/forevervalidator /usr/local/bin/forevervalidator

USER forevervalidator

ENTRYPOINT ["/usr/local/bin/forevervalidator"]
CMD []
