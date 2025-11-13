from fastapi import FastAPI
from api.users import router as users_router
from api.conversations import router as conversations_router

app = FastAPI(
    title="Memory Layer API",
    description="A service providing cognitive orchestration and persistent memory for LLMs.",
    version="1.0.0"
)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Memory Layer API is running."}

# Add the new, versioned routers
app.include_router(users_router)
app.include_router(conversations_router)

# The old routers can now be removed or commented out.
# from api.status import router as status_router
# from api.chat import router as chat_router
# from api.tasks import router as tasks_router
# app.include_router(status_router)
# app.include_router(chat_router)
# app.include_router(tasks_router)