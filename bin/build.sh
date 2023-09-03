#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../ 

. ./.env
export user_name group_name

docker build ./ -f docker/Dockerfile.cpu --build-arg user_name=$user_name --build-arg group_name=$group_name -t render

