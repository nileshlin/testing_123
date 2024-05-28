import logging
import os
from fastapi import FastAPI, Request,HTTPException
import orjson
from timeit import default_timer as timer
from fastapi.middleware.cors import CORSMiddleware
from torch_unit import disable_torch_init
from image_modifiers import load_image
from load_models import LoadVilaImage

llava_model = LoadVilaImage()

model_path = 'C:\\Users\linuxdev\Desktop\Projects\\vila_assistants\VILA1.5-3b'

tokenizer, model, image_processor, context_len = llava_model.load_pretrained_model(model_path)


# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # TODO, move logger to a separate  file to use everywhere
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# @app.get("/api")
# async def index():
#     logger.info('Executing Request : ')
#     return {
#         'status': 'success',
#         'message': 'Hello from main.py',
#     }
#
#
# @app.post("/api/describe_image")
# def query_check(request: IntentQuery):
#     disable_torch_init()
#     image = load_image(request.IMAGE)
#
#     return agents.intentClassifier(user_query)
#
#
# @app.post("/api/load_parcels_from_db")
# def parcel_query_from_db(query: ParcelQueryNoAuth):
#     user_query = query.message
#     start = timer()
#     parcel_data: bytes = orjson.dumps(processUserQueryBulkDataPgVector(parcel_query=user_query))
#     end = timer()
#     logger.info(f"Time Taken to Get Parcel using Bulk Data: {(end - start)}")
#     return parcel_data