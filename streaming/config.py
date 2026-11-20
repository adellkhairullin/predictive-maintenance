"""Общие настройки Kafka (переменные окружения)."""
import os

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:29092")
TOPIC = os.getenv("KAFKA_TOPIC", "sensor-readings")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "pdm-consumer")
PARTITIONS = int(os.getenv("KAFKA_PARTITIONS", "3"))
SCHEMA_VERSION = "1"


def ensure_topic(bootstrap: str = BOOTSTRAP, topic: str = TOPIC, partitions: int = PARTITIONS) -> bool:
    """Создаёт топик, если его нет. Возвращает True, если топик создан сейчас."""
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="pdm-admin")
    try:
        admin.create_topics([NewTopic(name=topic, num_partitions=partitions, replication_factor=1)])
        return True
    except TopicAlreadyExistsError:
        return False
    finally:
        admin.close()
