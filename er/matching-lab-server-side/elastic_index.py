from elasticsearch import AsyncElasticsearch

MAPPING = {
    "settings": {
        "index.max_ngram_diff": 10,
        "analysis": {
            # 1. CHAR FILTERS
            "char_filter": {
                # ВАЖНО: Исправленная регулярка для дробных чисел (1.5ml, 0,5%)
                "split_digits_units": {
                    "type": "pattern_replace",
                    "pattern": "(\\d+(?:[\\.,]\\d+)?)([%a-zA-Zа-яА-ЯёЁ]+)",
                    "replacement": "$1 $2"
                },
                "clean_special_chars": {
                    "type": "pattern_replace",
                    "pattern": "[^a-zA-Zа-яА-ЯёЁ0-9\\s\\.\\-%/,]",
                    "replacement": " "
                },
                "yo_replacer": {
                    "type": "mapping",
                    "mappings": ["ё => е", "Ё => Е"]
                }
            },
            
            # 2. NORMALIZERS (Для сортировки и агрегаций)
            "normalizer": {
                "lowercase_normalizer": {
                    "type": "custom",
                    "filter": ["lowercase", "asciifolding"]
                }
            },
            
            # 3. FILTERS
            "filter": {
                "ru_stemmer": { "type": "stemmer", "language": "russian" },
                "en_stemmer": { "type": "stemmer", "language": "english" },
                "latin_cleaner": { "type": "asciifolding", "preserve_original": True },
                "unique_token": { "type": "unique", "only_on_same_position": True },
                
                # Делимитер для ИНДЕКСАЦИИ (простой)
                "delimiter_index": {
                    "type": "word_delimiter",
                    "generate_word_parts": True,
                    "generate_number_parts": True,
                    "split_on_numerics": True,
                    "preserve_original": True,
                    "split_on_case_change": True
                },
                # Делимитер для ПОИСКА (Graph - умный)
                "delimiter_search": {
                    "type": "word_delimiter_graph",
                    "generate_word_parts": True,
                    "generate_number_parts": True,
                    "split_on_numerics": True,
                    "preserve_original": False, # False для поиска - это норма
                    "split_on_case_change": True
                },
                # NGram для поиска по обрывкам слов
                "ngram_filter": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 10
                }
            },
            
            # 4. ANALYZERS
            "analyzer": {
                # Анализатор при ЗАПИСИ (Index time)
                "goods_index_analyzer": {
                    "type": "custom",
                    "char_filter": ["html_strip", "clean_special_chars", "split_digits_units", "yo_replacer"],
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase", 
                        "latin_cleaner", 
                        "delimiter_index", 
                        "en_stemmer", 
                        "ru_stemmer", 
                        "unique_token"
                    ]
                },
                # Анализатор при ПОИСКЕ (Search time)
                "goods_search_analyzer": {
                    "type": "custom",
                    "char_filter": ["clean_special_chars", "split_digits_units", "yo_replacer"],
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase", 
                        "latin_cleaner", 
                        "delimiter_search", # Graph version
                        "en_stemmer", 
                        "ru_stemmer", 
                        "unique_token"
                    ]
                },
                # Анализатор для N-грамм (частичное совпадение)
                "ngram_analyzer": {
                    "type": "custom",
                    "char_filter": ["html_strip", "clean_special_chars", "split_digits_units", "yo_replacer"],
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase", 
                        "latin_cleaner", 
                        "ngram_filter", 
                        "unique_token"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "account_id": { "type": "keyword" },
            "product_vector": { 
                "type": "dense_vector", 
                "dims": 1024, 
                "index": True, 
                "similarity": "cosine" 
            },
            "product": {
                "properties": {
                    "id": { "type": "keyword" },
                    "archived": { "type": "boolean" },
                    "name": { 
                        "type": "text", 
                        "analyzer": "goods_index_analyzer",
                        "search_analyzer": "goods_search_analyzer",
                        "fields": {
                            # Для точных совпадений и сортировок
                            "keyword": { 
                                "type": "keyword", 
                                "ignore_above": 256,
                                "normalizer": "lowercase_normalizer"
                            },
                            # Для поиска без стемминга (только точные формы)
                            "exact": { 
                                "type": "text", 
                                "analyzer": "standard" 
                            },
                            # Для поиска по обрывкам и опечаткам
                            "partial": { 
                                "type": "text", 
                                "analyzer": "ngram_analyzer",
                                "search_analyzer": "standard" 
                            }
                        }
                    },
                    "updated": { "type": "date", "format": "yyyy-MM-dd HH:mm:ss.SSS" }
                }
            }
        }
    }
}

async def create_index_if_not_exists(conf, client: AsyncElasticsearch) -> None:
    index_name = conf.elastic.index_name
    print(f"DEBUG: Index Name: '{index_name}'")
    
    if not index_name or index_name.strip() == "":
        print("CRITICAL ERROR: Index Name is empty!")
        return

    try:
        exists = await client.indices.exists(index=index_name)
    except Exception as e:
        print(f"ERROR calling indices.exists: {e}")
        exists = False

    if exists:
        # ВАЖНО: Если ты хочешь применить изменения анализатора, 
        # индекс нужно УДАЛИТЬ и создать заново. 
        # Автоматически маппинг существующих полей с анализатором не меняется.
        print(f"Index '{index_name}' already exists. To update mapping, delete it first!")
        return

    print(f"Creating index '{index_name}' with numeric_delimiter support...")
    await create_index(index_name, client)

async def create_index(index_name: str, client: AsyncElasticsearch) -> None:
    resp = await client.indices.create(
        index=index_name,
        body=MAPPING,
    )
    if not resp.get("acknowledged", False):
        raise RuntimeError(f"Error creating index {index_name!r}: {resp}")
    print(f"Successfully created index {index_name}")