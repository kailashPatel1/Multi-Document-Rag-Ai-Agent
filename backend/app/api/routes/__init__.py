from .auth import router as auth_router
from .documents import router as documents_router
from .chat import router as chat_router
from .search import router as search_router
from .tools import router as tools_router
from .evaluation import router as evaluation_router
from .system import router as system_router

__all__ = [
    "auth_router",
    "documents_router",
    "chat_router",
    "search_router",
    "tools_router",
    "evaluation_router",
    "system_router"
]
