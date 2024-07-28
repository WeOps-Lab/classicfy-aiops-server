import uvicorn
from langserve import add_routes

from core.server_settings import server_settings
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from runnable.anomaly_detection.abod_runnable import AbodRunnable
from runnable.anomaly_detection.iforest_runnable import IForestRunnable
from runnable.anomaly_detection.inne_runnable import InneRunnable
from runnable.anomaly_detection.knn_runnable import KnnRunnable
from runnable.anomaly_detection.kpca_runnable import KpcaRunnable
from runnable.anomaly_detection.mad_runnable import MadRunnable
from runnable.anomaly_detection.ocssvm_runnable import OcsSvmRunnable
from runnable.anomaly_detection.suod_runnable import SuodRunnable
from runnable.anomaly_detection.xgbod_runnable import XgbodRunnable
from runnable.causation.causality_runnable import CausalityRunnable
from runnable.causation.fpgrowth_runnable import FPGrowthRunnable
from runnable.logreduce.drain_runnable import DrainRunnable
from runnable.timeseries.holt_winter_runnable import HoltWinterRunnable
from runnable.timeseries.sarima_runnable import SarimaRunnable


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
        add_routes(self.app, DrainRunnable().instance(), path="/logreduce/drain")

        add_routes(self.app, CausalityRunnable().instance(), path="/causation/causality")
        add_routes(self.app, FPGrowthRunnable().instance(), path="/causation/fpgrowth")

        add_routes(self.app, MadRunnable().instance(), path="/anomaly_detection/mad")
        add_routes(self.app, AbodRunnable().instance(), path="/anomaly_detection/abod")
        add_routes(self.app, SuodRunnable().instance(), path="/anomaly_detection/suod")
        add_routes(self.app, IForestRunnable().instance(), path="/anomaly_detection/iforest")
        add_routes(self.app, KnnRunnable().instance(), path="/anomaly_detection/knn")
        add_routes(self.app, XgbodRunnable().instance(), path="/anomaly_detection/xgbod")
        add_routes(self.app, InneRunnable().instance(), path="/anomaly_detection/inne")
        add_routes(self.app, KpcaRunnable().instance(), path="/anomaly_detection/kpca")
        add_routes(self.app, OcsSvmRunnable().instance(), path="/anomaly_detection/ocssvm")

        add_routes(self.app, HoltWinterRunnable().instance(), path="/timeseries/holt_winter")
        add_routes(self.app, SarimaRunnable().instance(), path="/timeseries/sarima")

    def start(self):
        self.setup_middlewares()
        self.setup_router()
        uvicorn.run(self.app, host=server_settings.app_host, port=server_settings.app_port)
