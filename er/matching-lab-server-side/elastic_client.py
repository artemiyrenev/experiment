from elasticsearch import AsyncElasticsearch

def new_async_es_client(config):
    # Логируем для отладки
    print(f"Connecting to ES at: {config.elastic.url} (Login: {config.elastic.login})")

    # Формируем аргументы для подключения
    client_kwargs = {
        "hosts": [config.elastic.url],
        # Пропускаем проверку SSL, так как мы работаем внутри docker по http
        "verify_certs": False, 
    }

    # Если логин и пароль заданы в конфиге, добавляем авторизацию
    if config.elastic.login and config.elastic.password:
        client_kwargs["basic_auth"] = (
            config.elastic.login,
            config.elastic.password,
            
        )

    es = AsyncElasticsearch(**client_kwargs)

    # Проверка es is None не совсем корректна для конструктора, 
    # он всегда вернет объект. Но можно оставить проверку на инициализацию.
    if not es:
        raise RuntimeError("Не удалось инициализировать клиент Elasticsearch")

    return es