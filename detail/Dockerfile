FROM r-base:4.2.0 AS base
WORKDIR /root/
RUN apt-get update && apt-get install -y \
    build-essential \
    dos2unix \
    libcurl4-openssl-dev \
    libgsl-dev \
    libxml2-dev \
    libssl-dev && \
rm -rf /var/lib/apt/lists/*
COPY texlive.profile /root/
RUN wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz && \
    tar zxf install-tl-unx.tar.gz && \
    rm install-tl-unx.tar.gz && \
    mv texlive.profile install-tl-* && \
    cd install-tl-* && \
    perl install-tl -profile texlive.profile && \
    cd .. && \
    rm -r install-tl-*
RUN tlmgr install \
    beamer \
    booktabs \
    caption \
    luatex85 \
    mathtools \
    multibib \
    multirow \
    setspace \
    standalone \
    pgf \
    titlesec \
    titling \
    xkeyval \
    jknapltx \
    rsfs

FROM base AS builder
WORKDIR /root/
RUN apt-get update && apt-get install -y \
    cmake \
    file \
    git \
    python3-dev \
    python3-pip && \
rm -rf /var/lib/apt/lists/*
RUN Rscript --vanilla -e "install.packages(c(\
    'DBI',\
    'distributional',\
    'egg',\
    'fable',\
    'ggplot2',\
    'ggrepel',\
    'gridExtra',\
    'latex2exp',\
    'lemon',\
    'MASS',\
    'pracma',\
    'Rfast',\
    'RSQLite',\
    'tibble',\
    'tikzDevice',\
    'tsibble',\
    'xptr',\
    'zoo'\
), dependencies = TRUE, repos = 'https://cran.rstudio.com/')"
COPY probabilistic/ probabilistic/
WORKDIR probabilistic/
RUN Rscript --vanilla "download_external_dependencies.R"
RUN BUILD=Release PROBABILISTIC_SILENCE_WARNS=TRUE R CMD INSTALL probabilistic
WORKDIR /root/
RUN rm -rf probabilistic

FROM base
COPY --from=builder /usr/lib/R/site-library/ /usr/lib/R/site-library/
COPY --from=builder /usr/local/lib/R/site-library/ /usr/lib/R/site-library/

