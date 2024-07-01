from typing import Annotated
from fastapi import FastAPI
from PIL import Image
from fastapi import UploadFile, File
import io
from utils import transformer, model_pipeline
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


app = FastAPI()


templates = Jinja2Templates(directory="./templates")
app.mount("/static", StaticFiles(directory="./static"), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/test_server")
def read_root():
    return {"Message": "Server is up :)"}


# exempt from api docs so set include_in_schema to False
@app.post("/prediction", include_in_schema=False)
def pred_func(request: Request, image: Annotated[bytes, File()]):
    try:
        content = Image.open(image).convert("RGB")
        content = transformer(content)
        prediction = model_pipeline(content)
        data = {"prediction": prediction}
        return templates.TemplateResponse(
            "result.html", {"request": request, "data": data}
        )

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return templates.TemplateResponse(
            "error.html", {"request": request, "error_msg": error_msg}
        )


@app.post("/prediction_fastapi_doc")
def pred_func(request: Request, image: UploadFile):
    try:
        content = image.file.read()
        content = Image.open(io.BytesIO(content)).convert("RGB")
        content = transformer(content)
        prediction = model_pipeline(content)
        data = {"prediction": prediction}
        return data

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return templates.TemplateResponse(
            "error.html", {"request": request, "error_msg": error_msg}
        )
