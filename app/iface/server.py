import asyncio

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid

from ..usecase.vpr.vpr import VPRSystem
from ..usecase.vpe.vpe import VPEProcessor


class VPEServer:
    def __init__(self, scenes):
        """
        Инициализирует сервер VPE, включая систему распознавания мест (VPR),
        построение индекса и маршруты API.
        """
        self.vpr = VPRSystem()
        self.vpr.build_index(scenes)

        self.processor = VPEProcessor(self.vpr)
        self.app = FastAPI(title="VPE Server")

        self._setup_routes()

        # Подключение статики (CSS, JS, изображения)
        self.app.mount("/static", StaticFiles(directory="app/static"), name="static")

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """Отображает главную HTML-страницу."""
            try:
                with open("app/templates/index.html", "r", encoding="utf-8") as f:
                    html = f.read()
                return HTMLResponse(content=html)
            except FileNotFoundError:
                return HTMLResponse(status_code=500, content="<h1>500 — Файл index.html не найден</h1>")

        @self.app.post("/process-video/")
        async def process_video(file: UploadFile = File(...)):
            """
            Обрабатывает загруженное видео, выполняет распознавание местоположения по кадрам.
            """
            temp_filename = f"/tmp/{uuid.uuid4()}_{file.filename}"

            try:
                # Сохраняем видео во временный файл
                with open(temp_filename, "wb") as buf:
                    shutil.copyfileobj(file.file, buf)

                # Обрабатываем видео
                results = await asyncio.to_thread(self.processor.process_video, temp_filename)

                # results = self.processor.process_video(temp_filename)
                print("✅ Результаты анализа:", results)

                return JSONResponse(content={"results": results})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

            finally:
                # Удаляем временный файл в любом случае
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

    def get_app(self) -> FastAPI:
        """
        Возвращает экземпляр FastAPI приложения.
        """
        return self.app
