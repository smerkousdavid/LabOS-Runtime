FROM ubuntu:jammy-20260109

ENV     LC_ALL=C.UTF-8     LANG=C.UTF-8     APP_UPDATE=manual     APP_BRANCH=dev     APP_PORT=8080

EXPOSE 8080


VOLUME /home/Shinobi/videos
VOLUME /home/Shinobi/libs/customAutoload
VOLUME /config

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://us.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list

SHELL [ "/bin/bash" , "-c" ]
RUN set -e \
    ; mkdir -p \
        /config \
        /home/Shinobi \
        /customAutoLoad \
    ; true

RUN set -e \
    ; apt-get update -qq \
    ; true

RUN set -e \
    ; DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -yqq -o Dpkg::Options::=--force-unsafe-io \
        gnupg2 apt-transport-https ca-certificates \
    ; true

RUN set -e \
    ; [ command -v curl >/dev/null 2>/dev/null ] || DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -yqq -o Dpkg::Options::=--force-unsafe-io \
        curl \
    ; true

RUN set -e \
    ; curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    ; true

RUN set -e \
    ; apt-get update -qq \
    ; true

RUN set -e \
    ; DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -yqq -o Dpkg::Options::=--force-unsafe-io \
        nodejs \
    ; true

RUN set -e \
    ; DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -yqq -o Dpkg::Options::=--force-unsafe-io \
        jq mysql-client \
    ; true

RUN set -e \
    ; DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -yqq -o Dpkg::Options::=--force-unsafe-io \
        ffmpeg \
    ; true

RUN set -e \
    ; DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -yqq -o Dpkg::Options::=--force-unsafe-io \
        python3 make git \
    ; true

RUN set -e \
    ; DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -yqq -o Dpkg::Options::=--force-unsafe-io \
        git \
    ; true

RUN set -e \
    ; ld=$(pwd) \
    ; cd "/home/Shinobi" \
    ; git clone "https://gitlab.com/Shinobi-Systems/Shinobi.git" "existing-dir.tmp" \
    ; mv 'existing-dir.tmp/.git' "." \
    ; rm -rf "existing-dir.tmp/" \
    ; git reset --hard HEAD \
    ; git checkout 'ea13a4dc58e92c4ab6c9ee192aca3bf9d91fcc16' \
    ; git checkout -b dev \
    ; true

RUN set -e \
    ; ld=$(pwd) \
    ; cd "/home/Shinobi" \
    ; npm install --unsafe-perm \
    ; npm install pm2 -g  \
    ; cd "$ld" \
    ; true

RUN set -e \
    ; ln -s /config/conf.json "/home/Shinobi/conf.json" \
    ; ln -s /config/super.json "/home/Shinobi/super.json" \
    ; true

COPY /docker-assets/docker-entrypoint.sh /home/Shinobi/
COPY /docker-assets/modifyJson.js /home/Shinobi/tools
RUN set -e \
    ; chmod -f +x /home/Shinobi/*.sh \
    ; chmod 777 /home/Shinobi/plugins \
    ; true

CMD ["pm2-docker", "Docker/pm2.yml"]
WORKDIR /home/Shinobi
ENTRYPOINT ["/home/Shinobi/docker-entrypoint.sh"]
