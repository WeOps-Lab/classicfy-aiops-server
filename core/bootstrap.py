import uvicorn
from core.server_settings import server_settings
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware


class Bootstrap:
    def __init__(self):
        load_dotenv()
        self.app = FastAPI(title=server_settings.app_name)

    def setup_middlewares(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

    def setup_router(self):
        pass

    def start(self):
        self.setup_middlewares()
        self.setup_router()
        uvicorn.run(self.app, host=server_settings.app_host, port=server_settings.app_port)
