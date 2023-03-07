from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from serializers import Event
from ultralytics import YOLO
from PIL import Image

app = FastAPI() 

@app.post("/api/event/events")
async def create_event(event: Event, response_model=Event):
    if event.EventHeader is None:
        return {"error": "EventHeader is required"}
    
    model = YOLO("yolov8l.pt")

    try:
        path = event.EventHeader.Path
        image = Image.open(path)

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

        event.EventHeader.IsRequiredObjectDetection = False
        event.EventBodies = event_bodies

        image.close()

        event_dict = event.dict()
        return event_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8888)