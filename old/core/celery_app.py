from celery import Celery
from core.config import settings

# The Celery app instance
# The first argument is the name of the current module.
# The broker argument specifies the URL of the message broker (Redis).
celery_app = Celery(
    "memory_layer_worker",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1", # Using a different DB for results
    include=["core.consolidation"] # List of modules to import when the worker starts
)

celery_app.conf.update(
    task_track_started=True,
)