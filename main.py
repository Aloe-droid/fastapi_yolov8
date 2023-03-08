from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from serializers import Event
from ultralytics import YOLO
from PIL import Image
# import logging
# logging.basicConfig(level=logging.INFO)

app = FastAPI(debug=False)
model = YOLO("fireLargeV8.pt")

@app.post("/api/event/events/")
async def create_event(request: Request):
    try:
        event_header = request.data['EventHeader']
    except KeyError:
        return JSONResponse(status_code=400, content={'Error': 'EventHeader key not found.'})

    try:
        userId = event_header['UserId']
        cameraId = event_header['CameraId']
        created = event_header['Created']
        path = event_header['Path']
    except KeyError:
        return JSONResponse(status_code=400, content={'Error': 'UserId, CameraId, Created, or Path key not found in EventHeader.'})

    try:
        image = Image.open(path)
    except:
        return JSONResponse(status_code=404, content={"Error": "File not found."})

    results = model.predict(image)

    event_bodies = []
    for result in results:
        for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
            left, top, right, bottom = bbox.tolist()
            event_bodies.append({
            'Label': cls.item(),
            'Left': left,
            'Top': top,
            'Right': right,
            'Bottom': bottom
            })

    sendEvent = Event(
        EventHeader={
            'UserId': userId,
            'CameraId': cameraId,
            'Created': created,
            'Path': path,
            'IsRequiredObjectDetection': False
        },
        EventBodies=event_bodies,
    )

    image.close()
    event_dict = sendEvent.dict()
    return JSONResponse(content=event_dict)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
