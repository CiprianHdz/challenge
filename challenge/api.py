import fastapi
from challenge.model import DelayModel
import pandas as pd
from fastapi import HTTPException
app = fastapi.FastAPI()
model = DelayModel()

VALID_OPERAS = {
    "Latin American Wings",
    "Grupo LATAM",
    "Sky Airline",
    "Copa Air",
    "Aerolineas Argentinas",  # used in tests
}
VALID_TIPOVUELO = {"I", "N"}

REQUIRED_PAYLOAD = {"OPERA", "TIPOVUELO", "MES"}



@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(payload: dict) -> dict:
    """
        Expected payload example:
        {
        "flights": [
            {
                "OPERA": "Aerolineas Argentinas",
                "TIPOVUELO": "N",
                "MES": 3
            }
        ]
        }
    """

    # Convert payload to DataFrame
    data = payload["flights"]
    df = pd.DataFrame(data)

    # -----------------
    #  Payload Validation
    # -----------------
    #if REQUIRED_PAYLOAD - set(df.columns):
    #    raise HTTPException(
    #        status_code=400,
    #        detail=f"Missing required fields: {sorted(list(missing))}"
    #    )

    # Month Validation
    if (~df["MES"].between(1, 12)).any():
        raise HTTPException(status_code=400)

    # FlightType validation
    if (~df["TIPOVUELO"].isin(VALID_TIPOVUELO)).any():
        raise HTTPException(status_code=400)

    # Opera Validation
    if (~df["OPERA"].isin(VALID_OPERAS)).any():
        raise HTTPException(status_code=400)

    # Preprocess and get prediction
    features = model.preprocess(data=df)
    preds = model.predict(features)

    return {"predict": [preds[0]] }