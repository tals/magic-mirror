![image](https://user-images.githubusercontent.com/761863/124344565-d83ee300-db87-11eb-8a62-e8b04f2181f3.png)

Made for a party in San Francisco's 20 Misson and The Archive.

It continuesly captures and indexes what it sees, and displays the closeset matches from the recent past :)


# Frontend
This is the web client, based on Svelte + Vite + Tailwind.

```sh
$ cd website
$ npm i
$ npm run dev
```

# Server

```sh
$ cd server
$ pip install -r requirements.txt
$ python app.py
```

## Caddy
Proxies everything. Does SSL termination

```
$ cd server
$ sudo caddy run
```
