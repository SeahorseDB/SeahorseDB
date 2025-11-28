#!/bin/sh
set -e

# first arg is `-f` or `--some-option`
# or first arg is `something.conf`
if [ "${1#-}" != "$1" ] || [ "${1%.conf}" != "$1" ]; then
	set -- seahorse-server "$@"
fi

# allow the container to be started with `--user`
if [ "$1" = 'seahorse-server' -a "$(id -u)" = '0' ]; then
	find . \! -user seahorse -exec chown seahorse '{}' +
# Add call to gosu to drop from root user to redis user
# when running original entrypoint
	exec gosu seahorse "$0" "$@"
fi

# set an appropriate umask (if one isn't set already)
# - https://github.com/docker-library/redis/issues/305
# - https://github.com/redis/redis/blob/bb875603fb7ff3f9d19aad906bd45d7db98d9a39/utils/systemd-redis_server.service#L37
um="$(umask)"
if [ "$um" = '0022' ]; then
	umask 0077
fi

# replace the current pid 1 with original entrypoint
exec "$@"