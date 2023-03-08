from typing import Optional, List
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image


router = APIRouter(
    prefix='/event',
    tags=['event']
)

model = YOLO("fireLargeV8.pt")


class EventHeader(BaseModel):
    UserId: str
    CameraId: int
    Created: str
    Path: str


class EventBody(BaseModel):
    Label: int
    Left: int
    Top: int
    Right: int
    Bottom: int


class Event(BaseModel):
    EventHeader: EventHeader
    EventBodies: Optional[List[EventBody]]


@router.post('/create', status_code=status.HTTP_200_OK)
def create_event(event: Event, response: Response):
    user_id = event.EventHeader.UserId
    camera_id = event.EventHeader.CameraId
    created = event.EventHeader.Created
    path = event.EventHeader.Path

    try:
        image = Image.open(path)
    except FileNotFoundError:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'Error': 'File not found.'}

    try:
        results = model.predict(image)
    except Exception as exception:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {'Error': str(exception)}

    image.close()

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

    return {
        'EventDTO': {
            'EventHeader': {
                'UserId': user_id,
                'CameraId': camera_id,
                'Created': created,
                'Path': path,
                'IsRequiredObjectDetection': False
            },
            'EventBodies': event_bodies
        }
    }
