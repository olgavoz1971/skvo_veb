# skvo_veb
This is the main archive of the skvo_veb application (https://skvo.science.upjs.sk/igebc) of Interactive Gaia Eclipsing Binary Catalog (Pavol Jozef Šafárik University in Košice, Faculty of Science)

The web application is written using Dash Plotly. Сallbacks that work with remote databases are organized as 'background callbacks'. To manage them we use a Redis server and Celery
## Redis and Celery installation
**Disclaimer**: The following instructions apply only to Linux, specifically Ubuntu 22.04.4 LTS.

### How to Install Redis (Community Edition)
https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/install-redis-on-linux/ 
```bash
$ sudo apt-get install lsb-release curl gpg
$ curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o  /usr/share/keyrings/redis-archive-keyring.gpg
$ sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg
$ echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
$ sudo apt-get update
$ sudo apt-get install redis
```
Redis-server will start automatically
Check its status:
```bash
$ sudo systemctl status redis
```
Alternatively, use redis-cli:
```bash
$ redis-cli
127.0.0.1:6379> ping
PONG
127.0.0.1:6379>
```
If Redis has not started, run it:
```bash
$ sudo systemctl start redis-server
```
To ensure Redis restarts at boot time, type:
```bash
$ sudo systemctl enable redis-server
```

### How to install Celery
To manage background callbacks, you need to install Celery in your application's virtual environment.
For more information, visit: https://dash.plotly.com/background-callbacks
https://dash.plotly.com/background-callbacks

```bash
$ source path_to_skvo_veb_project/.venv/bin/activate
(venv) $ pip install dash[celery]
```
#### Run celery in the Terminal (Debug Mode)
```bash
$ cd path_to_skvo_veb_project/
(venv) $ celery -A skvo_veb.celery_app worker --loglevel=DEBUG
```
This command starts Celery workers for the application.
Our project uses .env file to store sensitive information. To access the environmental variables from this file we export them before starting Celery in the go_celery.sh bash script:
```bash
$ cat go_celery.sh 
#!/bin/bash
export $(grep -v '^#' .env | xargs)
celery -A skvo_veb.celery_app worker --loglevel=DEBUG
$ cd path_to_skvo_veb_project/
(venv) $ ./go_celery.sh
```


### Celery Daemonization
In production mode, Celery must be run as a service.

1. Create /etc/systemd/system/celery.service file
```ini
[Unit]
Description=Celery Worker for skvo_veb
After=network.target

[Service]
TimeoutStartSec=120
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/flask
ExecStart=/var/www/flask/skvo_veb/venv/bin/celery -A skvo_veb.celery_app worker --loglevel=DEBUG --logfile=/var/www/flask/skvo_veb/log/celery.log
Environment="PATH=/var/www/flask/skvo_veb/venv/bin"
EnvironmentFile=/var/www/flask/.env
StandardOutput=append:/var/log/celery.log
StandardError=append:/var/log/celery.log

Restart=on-failure

[Install]
WantedBy=multi-user.target
```
Here we specify our environmental file using the keyword _EnvironmentFile_

2. Run the service
In production mode, place the application in the _/var/www/flask_ directory
Start the service from there:

```bash
$ systemctl daemon-reload
$ sudo systemctl start celery.service
```

Check  the service status:
```bash
$ sudo systemctl status celery.service
```
Again, to make sure the service starts on reboot, type:
```bash
sudo systemctl enable celery
```

### Troubleshooting
To view celery logs:
```bash
$ sudo journalctl -xeu celery.service
$ tail -f /var/www/flask/skvo_veb/log/celery.log
$ cat /var/log/celery.log
```
To check the Redis queue:
```bash
$ redis-cli
127.0.0.1:6379> KEYS *
127.0.0.1:6379> LRANGE celery 0 -1
```
