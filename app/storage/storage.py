import redis

class RedisStub:
    """
    Заглушка для Redis, если настоящий сервер недоступен.
    Хранит данные в памяти.
    """
    def __init__(self):
        self.data = {}
        self.counter = 0

    def ping(self):
        return True

    def incr(self, key):
        self.counter += 1
        return self.counter

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def flushdb(self):
        self.data.clear()
        self.counter = 0


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
