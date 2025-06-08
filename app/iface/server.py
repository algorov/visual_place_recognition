from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import shutil
import os
import uuid

from ..usecase.vpr.vpr import VPRSystem
from ..usecase.vpe.vpe import VPEProcessor


class VPEServer:
    def __init__(self):
        self.vpr = VPRSystem()
        self.processor = VPEProcessor(self.vpr)
        self.app = FastAPI()
        self._setup_routes()

        # 👇 Подключаем статику (CSS, JS, изображения)
        self.app.mount("/static", StaticFiles(directory="app/static"), name="static")

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            with open("app/static/index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())

        @self.app.post("/process-video/")
        async def process_video(file: UploadFile = File(...)):
            try:
                temp_filename = f"/tmp/{uuid.uuid4()}_{file.filename}"
                with open(temp_filename, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                results = await self.processor.process_video(temp_filename)

                os.remove(temp_filename)

                return JSONResponse(content={"results": results})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

    def get_app(self) -> FastAPI:
        return self.app
