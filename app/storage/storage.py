import redis


class RedisStub:
    """
    Заглушка для Redis, если настоящий сервер недоступен.
    Хранит данные в памяти.
    """
    def __init__(self):
        self._data = {}
        self._counters = {}

    def ping(self):
        return True

    def incr(self, key):
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    def set(self, key, value):
        self._data[key] = value

    def get(self, key):
        return self._data.get(key)

    def exists(self, key):
        return key in self._data

    def flushdb(self):
        self._data.clear()
        self._counters.clear()


class Storage:
    """
    Обертка для Redis клиента с fallback на RedisStub.
    """
    def __init__(self, host: str, port: int):
        try:
            self.client = redis.Redis(host=host, port=port, decode_responses=False)
            self.client.ping()
            print("✅ Redis подключен")
        except redis.ConnectionError:
            print("⚠️ Redis недоступен, используется RedisStub")
            self.client = RedisStub()

    def flushdb(self):
        self.client.flushdb()

    def incr(self, key):
        return self.client.incr(key)

    def set(self, key, value):
        self.client.set(key, value)

    def get(self, key):
        return self.client.get(key)

    def exists(self, key):
        # redis.Redis возвращает int (0 или 1), RedisStub — bool
        if isinstance(self.client, redis.Redis):
            return self.client.exists(key) > 0
        return self.client.exists(key)
