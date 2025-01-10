# skvo_veb
This is the main archive of the skvo_veb application (https://skvo.science.upjs.sk/igebc) of Interactive Gaia Eclipsing Binary Catalog (Pavol Jozef Šafárik University in Košice, Faculty of Science)

The web application is written with Dash Plotly. Сallbacks, working with remote databases, are organised as 'background callbacks'. To arrange them we use Redis server and Celery
## Redis and Celery installation
**Disclaimer**: The following instruction relates only to Linux, namely Ubuntu 22.04.4 LTS
### How to install Redis-community
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
Check it:
```bash
$ sudo systemctl status redis
```
Or this way:
```bash
$ redis-cli
127.0.0.1:6379> ping
PONG
127.0.0.1:6379>
```
If it has not started, run it:
```bash
$ sudo systemctl start redis-server
```
To make it restart at boot time, type:
```bash
$ sudo systemctl enable redis-server
```
### How to install Celery
https://dash.plotly.com/background-callbacks

It's better to do this under your application's virtual environment
```bash
$ source path_to_your_venv/.vennv/bin/activate
(venv) $ pip install dash[celery]
```

### Celery demonization

```bash
sudo systemctl enable celery
```
