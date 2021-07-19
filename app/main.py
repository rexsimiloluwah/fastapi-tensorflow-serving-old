import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from helpers import read_img_file, predict

app = FastAPI()


@app.post("/predict", response_model=dict)
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1]
    if not extension in ["jpg", "png", "jpeg"]:
        raise HTTPException(
            detail={
                "status": False,
                "error": "Only image files are supported i.e .jpg, .jpeg, .png"
            },
            status_code=422
        )

    flower_img = read_img_file(await file.read())
    print(flower_img)
    predictions = predict(flower_img)

    return {
        "predictions": list(predictions[0]),
        "confidence": round(predictions[1], 2),
        "class": predictions[2]
    }

if __name__ == "__main__":
    uvicorn.run(app, port=5000)
